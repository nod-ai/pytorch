import torch
import torch_zoom


torch.utils.rename_privateuse1_backend('zoom')
torch._register_device_module('zoom', torch_zoom)
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)

class ZoomTestBase(DeviceTypeTestBase):
    device_type = 'zoom'

TEST_CLASS = ZoomTestBase