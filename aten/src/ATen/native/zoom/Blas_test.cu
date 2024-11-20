#include <hip/hip_runtime.h>
#include <ATen/native/zoom/Blas_test.cuh>
#include <iostream>

#define HIP_ENABLE_PRINTF

__global__ void test_kernel(float* a) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid==0) {
        printf("KERNELPRINT\n");
        printf("%p\n", a);
        printf("%i\n", a != nullptr);
        if(a){
            printf("inbranch\n");
            a[0] = 3.0f;
        }
    }
}


// expects device ptr
void launch_test(at::Tensor result) {
    std::cout << result.options() << std::endl;
    std::cout << result.numel() << std::endl;
    hipPointerAttribute_t attributes;
    hipError_t err = hipPointerGetAttributes(&attributes, result.mutable_data_ptr<float>());
    if (!result.mutable_data_ptr<float>()) {
        printf("NULL RESULT\n");
    }
    else {
        printf("%p\n", result.mutable_data_ptr<float>());
    }

    if (err == hipSuccess) {
        if (attributes.type == hipMemoryTypeDevice) {
            printf("Pointer 'result' is a valid device pointer.\n");
            printf("Device: %d\n", attributes.device);
            printf("Device Pointer: %p\n", attributes.devicePointer);
            printf("Host Pointer: %p\n", attributes.hostPointer);
            printf("Is Managed Memory: %s\n", attributes.isManaged ? "Yes" : "No");
        } else {
            printf("Pointer 'result' is not a device pointer. Memory type: %d\n", attributes.type);
        }
    } else {
        printf("Error getting pointer attributes: %s\n", hipGetErrorString(err));
    }

    test_kernel<<<1,1>>>(result.data_ptr<float>());
}