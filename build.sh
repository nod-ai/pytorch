#!/bin/bash

rm -rf build
git clean -fdx -e .idea
git clean -fdX -e .idea


export USE_ZOOM=1
export USE_ROCM=0
export USE_CUDA=0
#export USE_PER_OPERATOR_HEADERS=1
export USE_CCACHE=1
export BUILD_PYTHON=1
export USE_NUMPY=1
export USE_FLASH_ATTENTION=0
#export BUILD_SHARED_LIBS=ON

export BUILD_AOT_INDUCTOR_TEST=0
#export BUILD_BINARY=0
#export BUILD_CUSTOM_PROTOBUF=1
export BUILD_DOCS=0
export BUILD_EXECUTORCH=0
export BUILD_FUNCTORCH=0
export BUILD_JNI=0
#export BUILD_LAZY_TS_BACKEND=1
#export BUILD_LIBTORCH_CPU_WITH_DEBUG=0
export BUILD_LITE_INTERPRETER=0
export BUILD_MOBILE_AUTOGRAD=0
export BUILD_MOBILE_BENCHMARK=0
export BUILD_MOBILE_TEST=0
export BUILD_ONNX_PYTHON=0
export BUILD_STATIC_RUNTIME_BENCHMARK=0
export BUILD_TEST=0
export USE_ASAN=0
export USE_C10D_GLOO=0
export USE_C10D_MPI=0
export USE_C10D_NCCL=0
export USE_COLORIZE_OUTPUT=0
export USE_COREML_DELEGATE=0
export USE_CPP_CODE_COVERAGE=0
export USE_CUDA=0
export USE_CUDNN=0
export USE_CUPTI_SO=0
export USE_CUSPARSELT=0
export USE_DISTRIBUTED=1
export USE_FAKELOWP=0
export USE_FBGEMM=0
export USE_FLASH_ATTENTI0=0
export USE_GFLAGS=0
export USE_GLOG=0
export USE_GLOO=0
export USE_GLOO_WITH_OPENSSL=0
export USE_GNU_SOURCE=0
export USE_GOLD_LINKER=0
export USE_IBVERBS=0
export USE_INTERNAL_PTHREADPOOL_IMPL=0
export USE_ITT=0
export USE_KINETO=0
export USE_LIBUV=0
export USE_LIGHTWEIGHT_DISPATCH=0
export USE_LITE_INTERPRETER_PROFILER=0
export USE_LITE_PROTO=0
export USE_MAGMA=0
export USE_MIMALLOC=0
export USE_MKLDNN=0
export USE_MKLDNN_CBLAS=0
export USE_MPI=0
export USE_NATIVE_ARCH=0
export USE_NCCL=0
export USE_NNAPI=0
export USE_NNPACK=0
export USE_NUMA=0
export USE_NVRTC=0
export USE_OBSERVERS=0
export USE_OPENCL=0
export USE_OPENMP=0
export USE_PRECOMPILED_HEADERS=0
export USE_PROF=0
export USE_PTHREADPOOL=0
export USE_PYTORCH_METAL=0
export USE_PYTORCH_METAL_EXPORT=0
export USE_PYTORCH_QNNPACK=0
export USE_QNNPACK=0
#export USE_RCCL=0
export USE_REDIS=0
#export USE_ROCM_KERNEL_ASSERT=0
export USE_SANITIZER=0
export USE_SLEEF_FOR_ARM_VEC256=0
export USE_SNPE=0
export USE_SOURCE_DEBUG_0_MOBILE=0
export USE_STATIC_CUDNN=0
export USE_STATIC_MKL=0
export USE_STATIC_NCCL=0
export USE_SYSTEM_BENCHMARK=0
export USE_SYSTEM_CPUINFO=0
export USE_SYSTEM_EIGEN_INSTALL=0
export USE_SYSTEM_FP16=0
export USE_SYSTEM_FXDIV=0
export USE_SYSTEM_GLOO=0
export USE_SYSTEM_GOOGLEBENCHMARK=0
export USE_SYSTEM_GOOGLETEST=0
export USE_SYSTEM_LIBS=0
export USE_SYSTEM_NCCL=0
export USE_SYSTEM_0NX=0
export USE_SYSTEM_PSIMD=0
export USE_SYSTEM_PTHREADPOOL=0
export USE_SYSTEM_PYBIND11=0
export USE_SYSTEM_SLEEF=0
export USE_SYSTEM_XNNPACK=0
export USE_TBB=0
export USE_TCP_OPENSSL_LINK=0
export USE_TCP_OPENSSL_LOAD=0
export USE_TENSORPIPE=1
export USE_TSAN=0
export USE_UCC=0
export USE_VALGRIND=0
export USE_VULKAN_FP16_INFERENCE=0
export USE_VULKAN_RELAXED_PRECISI0=0
export USE_XNNPACK=0
export USE_XPU=0

# for the ligerllama example we need distributed and tensorpipe, only because
# huggingface model.generate insists on querying torch.distributed and distributed relies on tensorpipe
# this could be a factor of nod-pytorch being out of date with upstream:
# https://github.com/pytorch/pytorch/issues/97397

python setup.py develop
python zoom_extension/examples/test.py
PYTORCH_TEST_WITH_SLOW=1 TORCH_TEST_DEVICES=zoom_extension/test/pytorch_test_base.py ./test.sh
python setup.py bdist_wheel