# import torch
# import torch_zoom

# torch.utils.rename_privateuse1_backend('zoom')
# torch._register_device_module('zoom', torch_zoom)
# # TODO: figure this out
# unsupported_dtypes = None
# torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)

# torch_zoom.init_zoom()

# print(torch._C._get_privateuse1_backend_name())

import torch
import torch.zoom

torch.utils.rename_privateuse1_backend('zoom')
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)
x = torch.empty(5, device='zoom:0')
