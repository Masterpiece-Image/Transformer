import torch
import torch.nn as nn





############################ Input Projection ############################
# Input of the transformer neural network

class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super(InputProj, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        act_layer(inplace=True)) # standard 3x3 convolutionnal layer

        if norm_layer is not None: # if a norm layer is added to the input
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None


    def forward(self, x):
        B, C, H, W = x.shape # dimension of x with H and W the Height and Weight of x, C
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None: # Same : if a there is a norm layer, we apply it before returning x
            x = self.norm(x)
        return x
