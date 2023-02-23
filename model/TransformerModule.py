import math
import torch
import torch.nn as nn

from REMSA import REMSA
from LeFF import LeFF

from utils.helpers import to_2tuple
from utils.drop import DropPath
from utils.window_operation import window_partition, window_reverse

########### Window-based Transformer Module (WTM) ###########
# Deraining-Transformer fig 4.a
# process :
# 1 - Layer Normalization
# 2 - W-REMSA
# 3 - Layer Normalization
# 4 - LeFF
# skip connection 1 - before the 1st Layer Normalization and after W-REMSA
# skip connection 2 - before the 2nd Layer Normalization and after LeFF
class WTM(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkvp_bias=True, scale=None, bound=8, leff_ratio=4., act_layer=nn.GELU, drop_path=0., shift_size=0) :
        super(WTM, self).__init__()
        # We obviously find the same arguments as in REMSA and LeFF
        # drop_path is for the DropBlock operation
        # shift_size is for the mask configuration in the forward method

        # 0 - INIT
        self.win_size = win_size
        self.shift_size = shift_size
    
        # 1 - Prepare the first Layer Normalization
        self.firstLayerNorm = nn.LayerNorm(dim)

        # 2 - Prepare W-REMSA
        self.WREMSA = REMSA(dim, win_size=to_2tuple(win_size), num_heads=num_heads, qkvp_bias=qkvp_bias, scale=scale, bound=bound)

        # Regularisation method for convolutional networks : DropBlock
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 3 - Prepare the second Layer Normalization
        self.secondLayerNorm = nn.LayerNorm(dim)

        # 4 - Prepare LeFF
        hidden_dim = int(dim * leff_ratio)
        self.LeFF = LeFF(dim, hidden_dim=hidden_dim, act_layer=act_layer)

    def forward(self, x, mask=None) :
        ### 0 - INIT
        H, W, C = x.shape
        # window of size sqrt(W) x sqrt(W)
        window_size = int(math.sqrt(W))
        # use for first skip connection :
        skip1 = x.detach().clone()

        ### 1 - Mask
        # input mask use in REMSA forward
        if mask is not None :
            input_mask = nn.functional.interpolate(mask, size=(window_size, window_size)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            remsa_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            remsa_mask = remsa_mask.unsqueeze(2) * remsa_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            remsa_mask = remsa_mask.masked_fill(remsa_mask != 0, float(-100.0)).masked_fill(remsa_mask == 0, float(0.0))
        else :
            remsa_mask = None

        # shift mask
        if self.shift_size > 0 :
            # calculate attention mask for shifted windows multi-heads self-attention
            # ref : 4.4 - ablation study, 4.4.1 - Relative Position Enhanced Multi-head Self-Attention
            shift_mask = torch.zeros((1, window_size, window_size, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_remsa_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            remsa_mask = remsa_mask or shift_remsa_mask
            remsa_mask = remsa_mask.masked_fill(shift_remsa_mask != 0, float(-100.0))

        ### 2 - First Layer Normalization
        x = self.firstLayerNorm(x)
        x = x.view(H, window_size, window_size, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x   
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        
        ### 3 - WREMSA
        wremsa_windows = self.WREMSA(x_windows, mask=remsa_mask)

        # merge windows
        wremsa_windows = wremsa_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(wremsa_windows, self.win_size, window_size, window_size) 

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(H, window_size * window_size, C)

        ### 4 - Skip connection 1
        x = skip1 + self.drop_path(x)

        ### 5 - second Layer Normalization + LeFF + 2nd skip connection
        x = x + self.drop_path(self.LeFF(self.secondLayerNorm(x)))

        del remsa_mask
        return x

########### Spatial-based Transformer Module (STM) ###########
# Deraining-Transformer fig 4.b
# process :
# 1 - Layer Normalization
# 2 - S-REMSA
# 3 - Layer Normalization
# 4 - LeFF
# skip connection 1 - before the 1st Layer Normalization and after S-REMSA
# skip connection 2 - before the 2nd Layer Normalization and after LeFF
class STM(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkvp_bias=True, scale=None, bound=8, leff_ratio=4., act_layer=nn.GELU) :
        super(STM, self).__init__()
        # We obviously find the same arguments as in REMSA and LeFF

    def forward(x) :
        return x
