#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <c10/core/ScalarType.h>

namespace at::native {

    // Helper function to convert hip_bfloat16 to float
    __device__ float bfloat16_to_float(hip_bfloat16 a) {
        union {
            uint32_t int32;
            float float32;
        } u = {uint32_t(a.data) << 16};
        return u.float32;
    }

    // Helper function to convert float to hip_bfloat16
    __device__ hip_bfloat16 float_to_bfloat16(float a) {
        union {
            float float32;
            uint32_t int32;
        } u = {a};
        hip_bfloat16 b;
        b.data = uint16_t(u.int32 >> 16);
        return b;
    }

    template <typename T>
    __device__ float convert_to_float(T a) {
        return a;
    }

    template <>
    __device__ float convert_to_float<hip_bfloat16>(hip_bfloat16 a) {
        return bfloat16_to_float(a);
    }

    template <>
    __device__ float convert_to_float<__half>( __half a) {
        return __half2float(a);
    }

    template <typename T>
    __device__ T convert_from_float(float a) {
        return static_cast<T>(a);
    }

    template <>
    __device__ hip_bfloat16 convert_from_float<hip_bfloat16>(float a) {
        return float_to_bfloat16(a);
    }

    template <>
    __device__ __half convert_from_float<__half>(float a) {
        return __float2half(a);
    }


    template <typename T>
    __global__ void batched_matmul_kernel(const T* A, const T* B, T* C, 
                                        int M, int N, int K, int batch_size) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int batch = blockIdx.z;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += convert_to_float(A[batch * M * K + row * K + k]) * 
                    convert_to_float(B[batch * K * N + k * N + col]);
            }
            C[batch * M * N + row * N + col] = convert_from_float<T>(sum);
        }
    }

    template <typename T>
    void batched_matmul(const T* A, const T* B, T* C, 
                        int M, int N, int K, int batch_size) {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    batch_size);

        hipLaunchKernelGGL(batched_matmul_kernel<T>, numBlocks, threadsPerBlock, 0, 0,
                        A, B, C, M, N, K, batch_size);
    }

    // Specialization for at::Half
    template <>
    void batched_matmul<at::Half>(const at::Half* A, const at::Half* B, at::Half* C,
                                        int M, int N, int K, int batch_size) {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    batch_size);

        hipLaunchKernelGGL(batched_matmul_kernel<__half>, numBlocks, threadsPerBlock, 0, 0,
            reinterpret_cast<const __half*>(A),
            reinterpret_cast<const __half*>(B),
            reinterpret_cast<__half*>(C),
            M, N, K, batch_size);
    }

    // Specialization for at::BFloat16
    template <>
    void batched_matmul<at::BFloat16>(const at::BFloat16* A, const at::BFloat16* B, at::BFloat16* C,
                                            int M, int N, int K, int batch_size) {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    batch_size);

        hipLaunchKernelGGL(batched_matmul_kernel<hip_bfloat16>, numBlocks, threadsPerBlock, 0, 0,
            reinterpret_cast<const hip_bfloat16*>(A),
            reinterpret_cast<const hip_bfloat16*>(B),
            reinterpret_cast<hip_bfloat16*>(C),
            M, N, K, batch_size);
    }

    // Explicit instantiations for supported types
    template void batched_matmul<float>(const float*, const float*, float*, int, int, int, int);
    template void batched_matmul<double>(const double*, const double*, double*, int, int, int, int);
    template void batched_matmul<half>(const half*, const half*, half*, int, int, int, int);
    template void batched_matmul<hip_bfloat16>(const hip_bfloat16*, const hip_bfloat16*, hip_bfloat16*, int, int, int, int);

} // at::native