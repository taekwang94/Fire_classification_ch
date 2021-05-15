import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


import math
from math import log

class eca_layer(nn.Module):
    """Constructs a ECA module. Args: channel: Number of channels of the input feature map k_size: Adaptive selection of kernel size """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class CACSNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None):
        super(CACSNet,self).__init__()
        self.eca_input = eca_layer(1)


        pass
    def forward(self, x_r, x_g, x_b):
        x_r = self.eca_input(x_r)
        x_g = self.eca_input(x_g)
        x_b = self.eca_input(x_b)
        print(x_b)
        x = torch.cat((x_r,x_g,x_b),1)
        print(x.size())
a= torch.rand(size=(1,1,3,3))
b= torch.rand(size=(1,1,3,3))
c= torch.rand(size=(1,1,3,3))
print(b)
CACSNet().forward(a,b,c)