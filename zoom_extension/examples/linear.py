import torch
from torch import nn
import torch_zoom

torch.utils.rename_privateuse1_backend('zoom')
torch._register_device_module('zoom', torch_zoom)
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)

torch_zoom.init_zoom()
DIM = 128
device = 'zoom'

model = nn.Sequential(
    nn.Linear(128, 128),
    nn.Linear(128, 128),
    nn.Linear(128, 5)
)

x = torch.randn((128), device=device)
model.to(device)
out = model(x)

x = torch.ones(3,3, device=device)
y = torch.ones(3,3, device=device)
z = torch.mm(x,y)
print(z)

# print(out.device)
# print(out)