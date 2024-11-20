# CMake

Using the `USE_ZOOM` flag with CMake will enable building with HIP for ROCm without requiring any of the "HIPify" scripts in order to build. This will include HIP libraries and populate `torch.version.hip` appropriately. This flag is NOT yet entered into the `setup.py` script, so for now it needs to be added manually via `cmake` or `ccmake`.

You'll need to set `ROCM_PATH` and `HIP_ROOT_DIR` appropriately, by default on linux these should be `/opt/rocm/` and `/opt/rocm/hip` respectively.

For now, I've added a Macro in `Allocator.h` that registers a functor that retrieves the `ZoomCachingAllocator` for us since we're currently implemented as an external backend (e.g. using PU1 dispatch key). Once, we're in the main repo we can replace it with the proper logic when retrieving the allocator for the Zoom backend.

# Setup.py for torch.zoom
First use `zoom_extension/env.sh` to set up the environment, you may need to change the `PYTORCH_ROCM_ARCH` variable based on what you get when running `rocminfo`, under "Name" there should be an architecture name like `gfx90a`.

Running `python setup.py install` inside root will build torch with zoom (currently still using the `PrivateUse` dispatch key).

Programs using the zoom backend must be prefaced with this stub until we register a proper dispatch key in pytorch

```python
import torch

torch.utils.rename_privateuse1_backend('zoom')
# TODO: figure this out
unsupported_dtypes = None
torch.utils.generate_methods_for_privateuse1_backend(unsupported_dtype=unsupported_dtypes)
```

# Running Device Type Tests
Set up the environment using `env.sh`. You may have to edit these variables if cloning. `TORCH_TEST_DEVICES` should point to `zoom_extension/test/pytorch_test_base.py`.

Then you can run `test.sh` to run the pytorch device test suite. This script will have a few output artifacts, one will be `test.log` with a verbose log of the `unittest` output from the test suite. Another is `zoom_unimplemented_operators.log` which will contain a list of unimplemented operators in the zoom backend, as well as the frequency with which this operator was called in the test suite. Finally, it will output a list of test failures (i.e. `AssertionError`) that were encountered in the test suite in `zoom_test_errors.log`.

The unimplemented operator log should not be considered exhaustive as additional operator failures may occur once the offending operator is implemented. This is just meant to be a tool to drive development.

# HIP Library Dependencies
For these running on ROCm, this also means that we take a dependency on the 'roc*' equivalent (e.g. hipBLAS requires rocBLAS)

* HIP - runtime, dtypes
* hipBLAS
* hipBLASLt
* hipRand
* hipSparse
* hipFFT
* rccl - TODO: add this in lieue of NCCL functionality
* hipThrust
* hipCub
* hipSolver

# HIPBlasLt

This is temporarily disabled via the macro `DISABLE_HIPBLASLT` in `ZoomContextLight.h`, we can reenable it by undef'ing that macro. This means that right now `scaledgemm` and `intmm` dont work, but we can implement hipblas versions of them and/or just enable hipblaslt.

# JITerator Notes:
https://dev-discuss.pytorch.org/t/keeping-pytorchs-ops-maintainable-the-jiterator/468



Dot kernel notes:
empty{} -> SIGBUS
zeros ({} or {1}) -> SIGBUS
zeros ({10}) -> SIGSEV??
Somehow not moving the pointer appropriately??
Blas_test.cu demonstrates that raw kernel code works and the pointer is not messed up in any way
-the exact same kernel when jitted with hiprtc throws sigbus though??

TODO List:

- Add RCCL
- Determine rocBLAS determinism requirements as far as config and versions (necessary to throw determinism errors when appropriate)

Note on error in test suite: `RuntimeError: t.use_count() <= 1`
This error is thrown in the `test_parallel_cow_materialize_error` test in the torch device type tests because
of many parallel references being held on the same tensor. This will only throw in debug mode. I think we can ignore this since 
this same error is thrown on the CPU backend in debug mode, and passes in release.

Note on error in `test_grad_scaling_state_dict`, this error occurs in the instance check `isinstance(s1._scale, torch.FloatTensor)`
because, despite their datatypes being equal, the PU1 dispatch key is a mismatch with the CPU dispatch key of the `FloatTensor` class.
These tensor types are deprecated anyways, and the rest of the test works so we can just ignore - if we want to we can add a
`torch.zoom.FloatTensor` (though this is a deprecated design pattern and likely frowned upon). The real correct thing to do is to refactor the instance check. See `python_tensor.cpp:Tensor_instancecheck`