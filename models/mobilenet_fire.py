from torch import nn
import torch
from .utils import load_state_dict_from_url


class Swish(nn.Module):
    def __init__(self, max_value = 10):
        super(Swish,self).__init__()
        self.max_value = 10

    def forward(self,x):
        out = x * torch.sigmoid(x)
        return torch.clamp(out, max=self.max_value)

class depthwise_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1,stride=2, groups=nin)


    def forward(self, x):
        out = self.depthwise(x)
        return out

class pointwise_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(x)
        return out

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d()
        self.swish10 = Swish(10)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.batchnorm(out)
        out = self.swish10(out)
        out = self.pointwise(out)
        out = self.batchnorm(out)
        out = self.swish10(out)
        return out


class MobileNet_Fire(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        input_channel = 32
        last_channel = 1280
        super(MobileNet_Fire, self).__init__()
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        self.d = depthwise_conv(3,32)
        self.dc1 = depthwise_separable_conv(32,16,24)
        self.dc2 = depthwise_separable_conv(24,24,24)