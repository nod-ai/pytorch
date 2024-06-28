import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

zoom_sources = glob.glob("src/**/*.cpp", recursive=True) + glob.glob("src/**/*.cu", recursive=True)
zoom_include_dirs = ["src", "src/kernels"]

setup(
    name="torch_zoom",
    ext_modules=[
        CUDAExtension(
            name="torch_zoom",
            sources=zoom_sources,
            extra_include_paths=zoom_include_dirs,
            extra_cflags=["-g"],
            verbose=True,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)