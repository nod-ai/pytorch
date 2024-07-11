from torch.utils.hipify import hipify_python
import os
import glob

# sources = glob.glob("src/cub*", recursive=False) + ["src/kernels/KernelUtils.cuh"]
sources = ["src/kernels/" + x for x in ["LegacyThrustHelpers.cu", "SortingCommon.cuh"]]
print("Targetting:")
print(sources)

include_dirs = []

build_dir = os.getcwd()
hipify_result = hipify_python.hipify(
    project_directory=build_dir,
    output_directory=build_dir,
    header_include_dirs=include_dirs,
    includes=[os.path.join(build_dir, '*')],  # limit scope to build_dir only
    extra_files=[os.path.abspath(s) for s in sources],
    show_detailed=True,
    is_pytorch_extension=True,
    hipify_extra_files_only=True,  # don't hipify everything in includes path
)

hipified_sources = set()
for source in sources:
    s_abs = os.path.abspath(source)
    hipified_s_abs = (hipify_result[s_abs].hipified_path if (s_abs in hipify_result and
                        hipify_result[s_abs].hipified_path is not None) else s_abs)
    # setup() arguments must *always* be /-separated paths relative to the setup.py directory,
    # *never* absolute paths
    hipified_sources.add(os.path.relpath(hipified_s_abs, build_dir))

sources = list(hipified_sources)