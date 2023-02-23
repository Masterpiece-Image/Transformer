import math
import torch.nn as nn

from einops import rearrange

########## Local-enhanced Feed-Forward network (LeFF) ##########
# method taken from :
# "Incorporating convolution designs into vision transformers", Fig 3.
# Proceedings of the IEEE/CVF International Conference on Learning Representations, 2021
 
# Employed before self-attention, applied to ease learning.

# Process :
# Input : tokens
# 1 - Linear Proj on tokens patch
# 2 - Spatial Restoration
# 3 - Depth-wise Convolution
# 4 - Flatten 
# 5 - Linear Proj
# Output : tokens

# contains a 3x3 depth-wise convolution between the two point-wise
# multi-layer perceptrons (MLPs).

class LeFF(nn.Module) :
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU) :
        super(LeFF, self).__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim

        # 1 - Prepare the first Linear Projection
        self.firstLinear = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer)

        # 2 - Prepare the depth-wise conv
        self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim), act_layer)

        # 3 - Prepare the last Linear Projection
        self.secondLinear = nn.Sequential(nn.Linear(hidden_dim, dim))

    # x is a token patch, dim(x) = h x w x c
    def forward(self, x) :
        _, w, _ = x.size()
        # we need to have square windows of the token (fig 3) 
        # so we take a window_size = sqrt(w) x sqrt(w)
        window_size = int(math.sqrt(w))

        # 1 - First Linear Projection
        x = self.firstLinear(x)

        # 2 - Spatial Restoration
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=window_size, w=window_size)

        # 3 - Depth-wise conv
        x = self.conv(x) + x

        # 4 - Flatten
        x = rearrange(x, 'b c h w -> b (h w) c', h=window_size, w=window_size)

        # 5 - Last Linear Projection
        x = self.secondLinear(x)
        return x