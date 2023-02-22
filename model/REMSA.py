import torch
import torch.nn as nn

from LinearProj import LinearProj as LP
from utils.trunc import trunc_normal_
from einops import repeat
######################### REMSA #########################
#   x belong to R(H, W, Dx)
class REMSA(nn.Module) :
    def __init__(self, dim, win_size, num_heads, qkvp_bias=True, scale=None, bound=8) :
        super(REMSA, self).__init__()

        # 0 - Init of Init
        self.dim        = dim       # dim of one token (1D)
        self.win_size   = win_size  # (Wh, Ww) -> Wh * Ww = number of token in one window 
        self.num_heads  = num_heads # number of heads per dimmension 

        head_dim = dim // num_heads # dim for one head

        # 1 - prepare the 1st step in fig 3 : the Linear Projection
        self.qkvp = LP(dim, num_heads, head_dim, qkvp_bias)

        # 2 - prepare the scale operation in fig 3
        self.scale  = scale or head_dim ** -0.5 # scaling use in fig 3

        # 3 - prepare to add the relative position bias B in fig 3
        # clipping de la position relative into N(o)
        self.relative_position_bias = nn.Parameter(
            torch.zeros((2 * bound - 1) * (2 * bound - 1), num_heads)  
        ) # Tensor size (2*o-1, 2*o-1, nH) with o <--> clipping value of eq9 

        # Prepare the relative position extraction (2nd part of eq9)
        distance = min(win_size[0]-1, 2)
        if distance > 0:
            kernelsize = 2 * distance+1
            padding = distance
        else:
            kernelsize = 1
            padding = 0
        self.Position_Extract = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernelsize, stride=1, 
            padding = padding, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, 
            groups = num_heads)
        )

        # 4 - Simplify the coord recovery using multi-head and with flat vectors instead of matrices
        coords_h    = torch.arange(self.win_size[0]) # [0, ..., Wh-1]
        coords_w    = torch.arange(self.win_size[1]) # [0, ..., Ww-1]
        coords      = torch.stack(torch.meshgrid(coords_h, coords_w)) # Tensor dim  : (2, Wh, Ww)

        coords_flatten  = torch.flatten(coords) # Tensor dim : (2, Wh*Ww) --> (t, k)
        relative_coords = torch.clamp(coords_flatten[:, :, None] - coords_flatten[:, None, :], -bound+1, bound-1)  # 2, Wh*Ww, Wh*Ww 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2 
        # We shift to start at 0
        relative_coords[:, :, 0] += bound - 1 
        relative_coords[:, :, 1] += bound - 1
        relative_coords[:, :, 0] *= 2 * bound - 1

        self.relative_position_index = relative_coords.sum(-1) # Wh*Ww, Wh*Ww --> B' coord
        
        self.proj    = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, attn_kv=None, mask=None) :

        # 0 - Init 
        H, W, Dx= x.shape # H * W = window dim, Dx = token dim

        # 1 - Linear Proj in fig 3.
        q, k, v, p = self.qkvp(x, attn_kv) # attn_kv attenuation factor
        
        # 2 - first matmul in fig 3. with the scale operation
        q = q * self.scale
        A = (q @ k.transpose(-2, -1))

        # 3 - Relative Position Bias in fig 3.
        relative_position_bias = self.relative_position_bias[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1
        ) # dim = Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # dim = nH,Wh*Ww,Wh*Ww
        ratio = A.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        A += relative_position_bias.unsqueeze(0)

        # 4 - softmax operation in fig 3.
        if mask is not None :
            nW = mask.shape
            mask = repeat(mask, 'nW l n -> nW m (n d)', d=ratio)
            A = A.view(H // nW, nW, self.num_heads, W, W * ratio) + mask.unsqueeze(1).unsqueeze(0)
            A = A.view(-1, self.num_heads, W, W * ratio)
        A = self.softmax(A)

        # 5 - Depth-wise conv on tokens and token-wise linear proj from fig 3.
        Position = self.Position_Extract(p).reshape(H, Dx, W).transpose(-2, -1)

        # 6 - Last matmul of fig 3 and Final operation of fig 3. sum the two sums
        x = ((A @ v)).transpose(1, 2).reshape(H, W, Dx) + Position

        x = self.proj(x)
        return x

