import glob
import torch
import torch.utils.cpp_extension

zoom_sources = glob.glob("src/**/*.cpp", recursive=True)
zoom_include_dirs = ["src", "src/kernels"]

# dynamically build zoom backend
torch_zoom = torch.utils.cpp_extension.load(
    name="zoom_device",
    sources=zoom_sources,
    with_cuda=True,
    extra_include_paths=zoom_include_dirs,
    extra_cflags=["-g"],
    verbose=True,
)

# add zoom module to torch
torch.utils.rename_privateuse1_backend('zoom')
torch._register_device_module('zoom', torch_zoom)

# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)