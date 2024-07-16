import torch
# import torch_zoom
import torch.zoom


torch.utils.rename_privateuse1_backend('zoom')
# torch._register_device_module('zoom', torch_zoom)
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)
# torch_zoom.init_zoom()

class ZoomTestBase(DeviceTypeTestBase):
    device_type = 'privateuseone'

TEST_CLASS = ZoomTestBase