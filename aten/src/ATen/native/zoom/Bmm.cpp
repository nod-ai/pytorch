#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/mm_native.h>
#endif


namespace at::native {
    // Forward decl, defined in HIPbmm.cu
    template <typename T>
    void batched_matmul(const T* A, const T* B, T* C, int M, int N, int K, int batch_size);

    const Tensor& bmm_out_hip_impl(const Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2) {
        // handle pathological cases
        if (result.numel() == 0) {
            return result;
        } else if (batch1.size(2) == 0) {
            return result.zero_();
        }
        TORCH_CHECK(batch1.sizes()[2] == batch2.sizes()[1], "batch1 dim 2 must match batch2 dim 1");

        c10::MaybeOwned<Tensor> result_ = c10::MaybeOwned<Tensor>::borrowed(result);
        IntArrayRef result_strides = result.strides();
        IntArrayRef result_sizes = result.sizes();

        int m = batch1.sizes()[1];
        int n = batch1.sizes()[2];
        int k = batch2.sizes()[2];
        int num_batches = result_->sizes()[0];

        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "bmm_hip", [&] {
            const scalar_t* batch1_ptr = batch1.const_data_ptr<scalar_t>();
            const scalar_t* batch2_ptr = batch2.const_data_ptr<scalar_t>();
            scalar_t* result_ptr = result_->mutable_data_ptr<scalar_t>();
           
           batched_matmul<scalar_t>(batch1_ptr, batch2_ptr, result_ptr, m, n, k, num_batches);
        });
        if (!result.is_same(*result_)) {
            result.copy_(*result_);
        }
        return result;

    }

    TORCH_IMPL_FUNC(bmm_out_zoom)(const Tensor& batch1, const Tensor& batch2, const Tensor &result)
    {
        NoNamesGuard guard;
        bmm_out_hip_impl(result, result, batch1, batch2);
    }

    Tensor& mm_out_hip_impl(Tensor& result, const Tensor& mat1, const Tensor& mat2) {
        // Make sure to keep addmm_hip below in sync with this code; it
        // preflights a check to try to avoid actually needing to call
        // expand().
        TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
        TORCH_CHECK(
            mat1.dtype() == mat2.dtype(),
            "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
        )

        TensorArg targs[]{{result, "out", 0}, {mat1, "mat1", 1}, {mat2, "mat2", 2}};
        checkAllSameGPU(__func__, targs);

        IntArrayRef mat1_sizes = mat1.sizes();
        IntArrayRef mat2_sizes = mat2.sizes();
        at::ScalarType scalar_type = mat1.scalar_type();
        TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
        TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "mat1 dim 1 must match mat2 dim 0");

        // resize result tensor
        at::native::resize_output(result, {mat1_sizes[0], mat2_sizes[1]});
        IntArrayRef result_sizes = result.sizes();
        if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
            return result;
        }

        if (mat1.numel() == 0) {
            // By definition, values in self should be ignored. nans and infs
            // should not propagate
            return result.zero_();
        }

        int m = mat1_sizes[0];
        int n = mat1_sizes[1];
        int k = mat2_sizes[1];

        // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result.is_conj());
        
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            scalar_type,
            "mm_zoom",
            [&] {
                const scalar_t* mat1_ptr = mat1.const_data_ptr<scalar_t>();
                const scalar_t* mat2_ptr = mat2.const_data_ptr<scalar_t>();
                scalar_t* result_ptr = result.mutable_data_ptr<scalar_t>();
                batched_matmul<scalar_t>(mat1_ptr, mat2_ptr, result_ptr, m, n, k, 1);
            });

        return result;
    }

    TORCH_IMPL_FUNC(mm_out_zoom)(const Tensor& self, const Tensor& mat2, const Tensor& result) 
    {
        mm_out_hip_impl(const_cast<Tensor&>(result), self, mat2);
    }

} // at::native


