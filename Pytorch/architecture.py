import torch
import torch.nn as nn

from Pytorch.spectral_norms import define_conv
from Pytorch.atrous_conv import define_atrous_conv

class ASPP(nn.Module):
    def __init__(self, opt, input_ch, input_resolution=65):
        super(ASPP, self).__init__()

        #get options
        nf = opt.aspp_nf

        #this rate is dilate size based original paper.
        x65_rate = [6, 12, 18]
        rate = [round(x * input_resolution / 65) for x in x65_rate]

        self.x1 = define_conv(opt)(input_ch, nf, ksize=1)
        self.x1_bn = nn.BatchNorm2d(nf)

        self.x3_small = define_atrous_conv(opt)(input_ch, nf, ksize=3, rate=rate[0])
        self.x3_small_bn = nn.BatchNorm2d(nf)
        
        self.x3_middle = define_atrous_conv(opt)(input_ch, nf, ksize=3, rate=rate[1])
        self.x3_middle_bn = nn.BatchNorm2d(nf)

        self.x3_large = define_atrous_conv(opt)(input_ch, nf, ksize=3, rate=rate[2])
        self.x3_large_bn = nn.BatchNorm2d(nf)

        self.sum_func = define_conv(opt)(nf * 4, input_ch, ksize=3, pad=1)

        self.activation = nn.ReLU()

    def forward(self, x):
        h1 = self.x1(x)
        h1 = self.x1_bn(h1)
        h1 = self.activation(h1)

        h2 = self.x3_small(x)
        h2 = self.x3_small_bn(h2)
        h2 = self.activation(h2)

        h3 = self.x3_middle(x)
        h3 = self.x3_middle_bn(h3)
        h3 = self.activation(h3)

        h4 = self.x3_large(x)
        h4 = self.x3_large_bn(h4)
        h4 = self.activation(h4)

        out = torch.cat((h1, h2, h3, h4), dim=1) #dim=1?????
        out = self.sum_func(out)
        out = self.activation(out)

        return out

class PixelShuffler(nn.Module):
    def __init__(self, opt, input_ch, output_ch, rate=2):
        super(PixelShuffler, self).__init__()

        output_ch = output_ch * rate**2

        self.c = define_conv(opt)(input_ch, output_ch, ksize=3, stride=1, pad=1)

        self.ps_func = nn.PixelShuffle(rate)

    def __call__(self, x):
        out = self.c(x)
        out = self.ps_func(out)

        return out