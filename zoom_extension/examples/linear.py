import torch
from torch import nn
from zoom import *

DIM = 128

model = nn.Sequential([
    nn.Linear(128, 128),
    nn.Linear(128, 128),
    nn.Linear(128, 5)
])

x = torch.randn(128, device='zoom')
model.to('zoom')

print(model(x))