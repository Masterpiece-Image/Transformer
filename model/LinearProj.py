import math
import torch.nn as nn

######################### Linear Projection #########################
#   Prepare and calculate the projs q = XWq, k = XWk, v = XWv and p = XWp
class LinearProj(nn.Module) :
    def __init__(self, dim, heads, dim_heads, bias=True) :
        super(LinearProj, self).__init__()
        self.dim        = dim                   # dimension of one token in X
        self.heads      = heads                 # number of heads
        self.inner_dim  = heads * dim_heads     

        # Wq and Wk have the same dim : Dx * Dk and Wv and Wp have the same dim : Dx * Dz
        # But to reduce the calculation time the windows are taken as square so Dk = Dz
        # Moreover, for calculation of k, v and p the originals authors have introduced an
        # attenuation factor that is why we end up with 2 nn.Linear function here instead of one.
        self.linear_to_q    = nn.Linear(dim, self.inner_dim, bias=bias)     # Input of size (H*W)*Dx and output of size (H*W)*inner_dim
        self.linear_to_kvp  = nn.Linear(dim, self.inner_dim*3, bias=bias)   # Input of size (H*W)*Dx and 3 outputs of size (H*W)*inner_dim

    def forward(self, x, attn_kv=None) :
        B_, N, C    = x.shape           # dimention of x with B_*N = number of token and C=dim=dimension of one token
        H           = int(math.sqrt(N)) # use to calculate p 

        attn_kv = x if attn_kv is None else attn_kv

        q   = self.linear_to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kvp = self.linear_to_kvp(attn_kv).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)

        q = q[0]
        k, v, p_temp = kvp[0], kvp[1], kvp[2]
        p = p_temp.transpose(-1, -2).reshape(B_, C, N).view(B_, C, H, H)

        return q, k, v, p