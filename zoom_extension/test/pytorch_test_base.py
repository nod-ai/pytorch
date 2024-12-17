import torch
import torch.zoom
from typing import ClassVar

torch.utils.rename_privateuse1_backend('zoom')
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)

class ZoomTestBase(DeviceTypeTestBase):
    device_type = 'privateuseone'
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device
    

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.zoom.device_count()

        prim_device = cls.get_primary_device()
        zoom_str = 'zoom:{0}'
        non_primary_devices = [zoom_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices
    
    @classmethod
    def setUpClass(cls):
        # Force Zoom Init
        t = torch.ones(1, device='zoom')
        # Acquires the current device as the primary (test) device
        cls.primary_device = f'zoom:{torch.zoom.current_device()}'

TEST_CLASS = ZoomTestBase