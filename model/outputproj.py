import torch
import torch.nn as nn


############################ Output Projection ############################
# Restoration of the features to the original size

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super(OutputProj, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        ) # standard 3x3 convolutionnal layer

        if act_layer is not None: # if there is an activation layer, we add it to the convolutionnal layer
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None: # if there is an normalisation layer
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape # shape of x with B : , L : H*W , C :
        H = int(math.sqrt(L)) # calculate H
        W = int(math.sqrt(L)) # calculate W
        x = x.transpose(1, 2).view(B, C, H, W) # reshaping x
        x = self.proj(x) # convlayer
        if self.norm is not None: # if norm, we apply it
            x = self.norm(x)
        return x

    def flops(self, H, W): # calculate the number of operations
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops
