import math
import torch.nn as nn

from REMSA import REMSA
from LeFF import LeFF

from utils.helpers import to_2tuple
from utils.drop import DropPath

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
    def __init__(self, dim, win_size, num_heads, qkvp_bias=True, scale=None, bound=8, leff_ratio=4., act_layer=nn.GELU, drop_path=0.) :
        super(WTM, self).__init__()
        # We obviously find the same arguments as in REMSA and LeFF

        # 0 - INIT
        self.win_size = win_size
    
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
        H, W, C = x.shape
        # window of size sqrt(W) x sqrt(W)
        window_size = int(math.sqrt(W))

        # a mask use in REMSA forward
        if mask is not None :
            input_mask = nn.functional.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            remsa_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            remsa_mask = remsa_mask.unsqueeze(2) * remsa_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            remsa_mask = remsa_mask.masked_fill(remsa_mask != 0, float(-100.0)).masked_fill(remsa_mask == 0, float(0.0))
        
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
