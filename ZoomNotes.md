# CMake

Using the `USE_ZOOM` flag with CMake will enable building with HIP for ROCm without requiring any of the "HIPify" scripts in order to build. This will include HIP libraries and population `torch.version.hip` appropriately. This flag is NOT yet entered into the `setup.py` script, so for now it needs to be added manually via `cmake` or `ccmake`.

# Setup.py
First use `env.sh` to set up the environment, you may need to change the `PYTORCH_ROCM_ARCH` variable based on what you get when running `rocminfo`, under "Name" there should be an architecture name like `gfx90a`.

Running `python setup.py install` inside `zoom_extension/` will install the `torch_zoom` package, to use this in a python process you must `import torch` before `import torch_zoom`, otherwise certain necessary shared libraries will not be available.

Programs using the zoom backend must be prefaced with this stub until we register a proper dispatch key in pytorch

```python
import torch
import torch_zoom

torch.utils.rename_privateuse1_backend('zoom')
torch._register_device_module('zoom', torch_zoom)
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)
```

# Running Device Type Tests
Set up the environment using `env.sh`. You may have to edit these variables if cloning. `TORCH_TEST_DEVICES` should point to `test/pytorch_test_base.py`.

Then you can run `pytorch/test/test_torch.py` to run all device tests.

TODO List:


- Impl kernels and functionality w/ AOT triton kernels
- Properly support multiple devices, this involves managing contexts appropriately between threads. We should have a bank of streams corresponding to each device as well as a bank of allocators corresponding to each device.