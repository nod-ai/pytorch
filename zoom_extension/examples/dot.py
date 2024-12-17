
import torch
import torch.zoom

torch.utils.rename_privateuse1_backend('zoom')
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)
device='zoom'

n = 1024
x = torch.ones(n, device=device, dtype=torch.float)
y = torch.ones(n, device=device, dtype=torch.float)

print(x.is_contiguous(), y.is_contiguous())

z = torch.dot(x, y)

print(z)

print(x[:10])

# x = torch.rand((n,))
# y = torch.rand((n,))

# z1 = torch.dot(x,y)

# x2 = x.to(device=device)
# y2 = y.to(device=device)

# z2 = torch.dot(x2, y2)

# print(torch.allclose(z1, z2.cpu()))