import matplotlib.pyplot as plt
import numpy as np
import torch

from Pytorch.generator import ResNetDeepLab
from options import get_options

opt = get_options()

gen = ResNetDeepLab(opt)

gen.load_state_dict(torch.load('pretrained/gen.pth')['gen'])

x = plt.imread('predict_from/sample.png')
x = torch.from_numpy(x).transpose(1,2).transpose(0,1).unsqueeze(0)
x = gen(x)

for i in range(5):
    plt.subplot(230+i+1)
    plt.imshow(x[0,i].detach().numpy())
plt.show()

print()