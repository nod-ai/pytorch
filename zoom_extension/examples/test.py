import torch
import torch_zoom

torch.utils.rename_privateuse1_backend('zoom')
torch._register_device_module('zoom', torch_zoom)
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)

torch_zoom.init_zoom()

print(torch._C._get_privateuse1_backend_name())

# x = torch.empty(5, device='zoom:0')

def list_mybackend_devices(backend='zoom'):
    devices = []
    i = 0
    while True:
        try:
            # Attempt to create a tensor on the device
            torch.tensor([1], device=f'{backend}:{i}')
            devices.append(f'{backend}:{i}')
            i += 1
        except RuntimeError as e:
            # If we get an error, assume we've reached the end of available devices
            break
    return devices

print(list_mybackend_devices())
print(list_mybackend_devices('privateuseone'))

# for some reason zoom:0 works fine, but using another device index gives memory access errors when running the abs kernel
x = torch.ones(5, device='zoom') #* -1

print(x.cpu())
# y = x.to(torch.device('zoom:1'))
# print(y == y)
# print(y.cpu())
# print(y.abs().cpu())
# print(y.device, y.abs().device)
# z = x.to(torch.device('zoom', 1))
# print(z.cpu())
# print(z.abs().cpu())