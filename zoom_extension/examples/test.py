import torch
import torch_zoom

torch.utils.rename_privateuse1_backend('zoom')
torch._register_device_module('zoom', torch_zoom)
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)

torch_zoom.init_zoom()

x = torch.empty(5, device='zoom:0')

x = torch.ones(5, device='cpu') * -1
y = x.to('zoom:0')
z = y.to('zoom:1')

print(y.device)

print(z.device)

# print(z.cpu())
f = z.to('cpu')
print(f)
# z = z.abs()
z = torch.abs(z)
print(z.device)

print(z.cpu())
print(f)
print(y.abs().cpu())