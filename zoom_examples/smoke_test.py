import torch
import torch.zoom
torch.utils.rename_privateuse1_backend('zoom')
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=None)

x = torch.randn(10, device='zoom')