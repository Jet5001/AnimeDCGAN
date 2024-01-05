#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from Generator import Generator
MODELPATH = "AnimeGanTake2/outputs/Models/generatorNet.pth"

manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

NUM_IMGS = 128
generatorNet = Generator(1, 100, 64, 3)
generatorNet = torch.load("AnimeGanTake2/outputs/Models/generatorNet.pth")
generatorNet.eval()
generatorNet.to('cuda')
imgs = []
numCreated = 0
for i in range(NUM_IMGS):
    noise = torch.randn(1, 100, 1, 1, device='cuda')
    imgs.append(generatorNet(noise).squeeze())

plt.axis("off")
plt.tight_layout()
count = 0
for i in imgs:
    plt.imshow(np.transpose(vutils.make_grid(i.to('cuda'), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(f"AnimeGanTake2/outputs/Images/64x64/{count}.png", bbox_inches = 'tight', facecolor = 'black')
    count +=1