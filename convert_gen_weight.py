import chainer
import importlib
import torch

from generator import ResNetDeepLab as CRes
from options import get_options
from Pytorch.generator import ResNetDeepLab as PRes


gen_npz = 'pretrained/gen.npz'
opt = get_options()

c_gen = CRes(opt)
p_gen = PRes(opt)


chainer.serializers.load_npz(gen_npz, c_gen)
d = dict([(i,torch.from_numpy(chainer.as_array(j))) for i,j in c_gen.namedparams() if 'resnet' not in i])


ordered_layer_list = [
    '/c1/c/b',                  #1
    '/c1/c/W',                  #1  
    '/norm1/gamma',             #2
    '/norm1/beta',              #2
    '/c2/c/b',                  #3
    '/c2/c/W',                  #3
    '/norm2/gamma',             #4
    '/norm2/beta',              #4
    '/aspp/x1/c/b',             #5 1
    '/aspp/x1/c/W',             #5 1
    '/aspp/x1_bn/gamma',        #5 2
    '/aspp/x1_bn/beta',         #5 2
    '/aspp/x3_small/c/b',       #5 3
    '/aspp/x3_small/c/W',       #5 3
    '/aspp/x3_small_bn/gamma',  #5 4
    '/aspp/x3_small_bn/beta',   #5 4
    '/aspp/x3_middle/c/b',      #5 5
    '/aspp/x3_middle/c/W',      #5 5
    '/aspp/x3_middle_bn/gamma', #5 6
    '/aspp/x3_middle_bn/beta',  #5 6
    '/aspp/x3_large/c/b',       #5 7
    '/aspp/x3_large/c/W',       #5 7
    '/aspp/x3_large_bn/gamma',  #5 8
    '/aspp/x3_large_bn/beta',   #5 8
    '/aspp/sum_func/c/b',       #5 9
    '/aspp/sum_func/c/W',       #5 9
    '/up1/c/c/b',               #6
    '/up1/c/c/W',               #6
    '/up2/c/c/b',               #7
    '/up2/c/c/W',               #7
    '/up3/c/c/b',                #8
    '/up3/c/c/W',               #8
    '/to_class/c/b',            #9
    '/to_class/c/W',            #9
]

len_resnet_module = len(
    list(
        list(p_gen.children())[0].modules()
    )
) + 1

with torch.no_grad():
    for i,m in enumerate(p_gen.children()):
        if i != 0: #skip resnet weights
            if i < 5: #skip aspp
                #params = list(m.parameters())
                for j,p in enumerate(m.parameters()):
                    p.data = d[ordered_layer_list[(i-1)*2+j]]
            if i == 5: #aspp
                for i2, mm in enumerate(m.children()):
                    for j,p in enumerate(mm.parameters()):
                        p.data = d[ordered_layer_list[(i-1+i2)*2+j]]
                i2 -= 1 #remove relu
            if i > 5:
                for j,p in enumerate(m.parameters()):
                    p.data = d[ordered_layer_list[(i-1+i2)*2+j]]

torch.save(
    {
        'gen': p_gen.state_dict()
    },
    'pretrained/gen.pth'
)