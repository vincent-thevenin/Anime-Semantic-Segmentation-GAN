import torch.nn as nn

#import chainer.links as L
#from spectral_norms import SNConv, SNHookConv

"""
class AtrousConv(L.DilatedConvolution2D):
    def __init__(self, in_channels, out_channels, ksize=None, rate=1, initialW=None):
        super().__init__(in_channels, out_channels, ksize=ksize, stride=1, pad=rate, dilate=rate, initialW=initialW)


class AtrousSNConv(SNConv):
    def __init__(self, in_channels, out_channels, ksize=None, rate=1, initialW=None):
        super().__init__(in_channels, out_channels, ksize=ksize, stride=1, pad=rate, dilate=rate, initialW=initialW)
"""

class AtrousSNHookConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=None, rate=1, initialW=None):
        super(AtrousSNHookConv, self).__init__()

        self.c =nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=1, padding=rate,
            bias=True, dilation=rate)
        self.c = nn.utils.spectral_norm(self.c)

    def forward(self, x):
        return self.c(x)


def define_atrous_conv(opt):
    if opt.conv_norm == 'original':
        return AtrousConv
    
    if opt.conv_norm == 'spectral_norm':
        return AtrousSNConv
    
    if opt.conv_norm == 'spectral_norm_hook':
        return AtrousSNHookConv