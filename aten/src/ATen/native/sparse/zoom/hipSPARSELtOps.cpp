// !!! This is a file automatically generated by hipify!!!
#include <ATen/zoom/ZoomContext.h>
#include <ATen/zoom/ZoomDataType.h>
#include <ATen/zoom/HIPSparse.h>
#include <ATen/zoom/HIPConfig.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>
#include <c10/zoom/ZoomCachingAllocator.h>
#include <c10/util/Half.h>
#include <hipsparse/hipsparse.h>
#include <cstdint>

#if AT_HIPSPARSELT_ENABLED()

#include <hipsparselt/hipsparselt.h>

namespace at::native {

// Ideally we would use the same DeviceThreadHandlePool mechanism as used in aten/src/ATen/cuda/CuSparseHandlePool.cpp
// which would handle this for us. However, the hipSPARSELt handle signature is different from that of cuSPARSE/cuBLAS,
// so it's not possible to reuse the existing pooling mechanism. Instead we have to handle our handles ourselves, which
// is why these variables are thread local. Once hipSPARSELt updates their handle signature to be consistent with the rest
// of CUDA, we can switch to using DeviceThreadHandlePool.
thread_local hipsparseLtHandle_t handle;
thread_local bool handle_initialized = false;

at::Tensor _cslt_compress(const Tensor& sparse_input)
{
    if (!handle_initialized){
        TORCH_HIPSPARSE_CHECK(hipsparseLtInit(&handle));
        handle_initialized = true;
    }
    // create sparse descriptor, dtype
    hipsparseLtMatDescriptor_t sparse_input_descriptor;
    hipDataType type;
    auto compression_factor = 9;

    switch(
        sparse_input.scalar_type()
    )
    {
        case at::ScalarType::Char:
            type = HIP_R_8I;
            compression_factor = 10;
            break;
        case at::ScalarType::Half:
            type = HIP_R_16F;
            break;
        case at::ScalarType::BFloat16:
            type = HIP_R_16BF;
            break;
        case at::ScalarType::Float:
            type = HIP_R_32F;
            break;
        default:
            TORCH_CHECK(false, "Unsupported dtype for hipSPARSELt compressed matrix");
            break;
    }

    // create a new compressed tensor with the same dtype as
    auto compressed_tensor = sparse_input.new_empty(sparse_input.numel() * compression_factor / 16);

    TORCH_HIPSPARSE_CHECK(hipsparseLtStructuredDescriptorInit(
        &handle,
        &sparse_input_descriptor,
        sparse_input.size(0),
        sparse_input.size(1),
        sparse_input.size(1),
        16,
        type,
        HIPSPARSE_ORDER_ROW,
        HIPSPARSELT_SPARSITY_50_PERCENT));

    // compress input
    //--------------------------------------------------------------------------
    size_t compressed_size, compressed_buffer_size;
    TORCH_HIPSPARSE_CHECK(hipsparseLtSpMMACompressedSize2(
        &handle,
        &sparse_input_descriptor,
        &compressed_size,
        &compressed_buffer_size));

    auto& allocator = *::c10::zoom::ZoomCachingAllocator::get();
    auto compressedBufferPtr = allocator.allocate(compressed_buffer_size);
    hipStream_t stream = c10::zoom::getCurrentZoomStream();

    TORCH_HIPSPARSE_CHECK(hipsparseLtSpMMACompress2(
        &handle,
        &sparse_input_descriptor,
        true,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        sparse_input.data_ptr(),
        compressed_tensor.data_ptr(),
        compressedBufferPtr.get(),
        stream));

    return compressed_tensor;
}

std::tuple<int64_t, at::Tensor> _cslt_sparse_mm_impl(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int alg_id,
    bool search_alg_id
)
{
  if (!handle_initialized){
      TORCH_HIPSPARSE_CHECK(hipsparseLtInit(&handle));
      handle_initialized = true;
  }
  // cupsarselt constructs
  hipsparseLtMatmulDescriptor_t matmul;
  hipsparseLtMatmulPlan_t plan;
  hipsparseLtMatmulAlgSelection_t alg_sel;

  int tensor_alpha_mode = 0;
  float alpha = 1.0;
  float beta = 0.0;
  hipsparseLtDatatype_t input_type;
  hipDataType output_type;
  hipsparseLtComputetype_t compute_type;
  auto compression_factor = 9;


  switch(compressed_A.scalar_type())
  {
    case at::ScalarType::Char:
        input_type = HIPSPARSELT_R_8I;
        output_type = HIP_R_8I;
        compute_type = HIPSPARSELT_COMPUTE_32I;
        compression_factor = 10;
        break;
    case at::ScalarType::Half:
        input_type = HIPSPARSELT_R_16F;
        output_type = HIP_R_16F;
        compute_type = HIPSPARSELT_COMPUTE_32F;
        break;
    case at::ScalarType::BFloat16:
        input_type = HIPSPARSELT_R_16BF;
        output_type = HIP_R_16BF;
        compute_type = HIPSPARSELT_COMPUTE_32F;
        break;
    case at::ScalarType::Float:
        input_type = HIPSPARSELT_R_32F;
        output_type = HIP_R_32F;
        compute_type = HIPSPARSELT_COMPUTE_32F;
        break;
    default:
        TORCH_CHECK(false, "Unsupported dtype for hipSPARSELt compressed matrix multiplication.");
        break;
  }
  ScalarType out_dtype = dense_B.scalar_type();
  // special check for mixed dtype int8 int8 -> {fp16, bf16, int32} support
  if (out_dtype_opt.has_value()) {
    out_dtype = out_dtype_opt.value();
    TORCH_CHECK(input_type == HIPSPARSELT_R_8I, "out_dtype support only available for int8 inputs");
    switch (out_dtype)
    {
        case at::ScalarType::Half:
            output_type = HIP_R_16F;
            break;
        case at::ScalarType::BFloat16:
            output_type = HIP_R_16BF;
            break;
        case at::ScalarType::Int:
            output_type = HIP_R_32I;
            break;
        default:
            TORCH_CHECK(false, "Unsupported out_dtype passed, must be one of {fp16, bf16, int32}");
            break;
    }
  }

  int64_t k = dense_B.size(0);
  int64_t n = dense_B.size(1);
  int64_t m = (compressed_A.numel() * 16 / compression_factor  ) / k;

  //initialize sparse descriptor
  hipsparseLtMatDescriptor_t sparse_input_descriptor;
  TORCH_HIPSPARSE_CHECK(hipsparseLtStructuredDescriptorInit(
      &handle,
      &sparse_input_descriptor,
      m,
      k,
      k,
      16,
      input_type,
      HIPSPARSE_ORDER_ROW,
      HIPSPARSELT_SPARSITY_50_PERCENT));

  // initialize dense input descriptor
  hipsparseLtMatDescriptor_t dense_input_descriptor;
  TORCH_HIPSPARSE_CHECK(hipsparseLtDenseDescriptorInit(
      &handle,
      &dense_input_descriptor,
      (dense_B.is_contiguous()) ? k : n,
      (dense_B.is_contiguous()) ? n : k,
      (dense_B.is_contiguous()) ? n : k,
      16,
      input_type,
      HIPSPARSE_ORDER_ROW));

  // create result tensor
  auto res_tensor_options = c10::TensorOptions().dtype(out_dtype).device(dense_B.device());
  at::Tensor res = (transpose_result) ? at::empty({n, m}, res_tensor_options)
                                      : at::empty({m, n}, res_tensor_options);

  hipsparseLtMatDescriptor_t res_descriptor;
  TORCH_HIPSPARSE_CHECK(hipsparseLtDenseDescriptorInit(
      &handle,
      &res_descriptor,
      m,
      n,
      (transpose_result) ? m: n,
      16,
      output_type,
      (transpose_result) ? HIPSPARSE_ORDER_COLUMN : HIPSPARSE_ORDER_ROW));

  // initialize matmul
  TORCH_HIPSPARSE_CHECK(hipsparseLtMatmulDescriptorInit(
      &handle,
      &matmul,
      HIPSPARSE_OPERATION_NON_TRANSPOSE,
      (dense_B.is_contiguous()) ? HIPSPARSE_OPERATION_NON_TRANSPOSE : HIPSPARSE_OPERATION_TRANSPOSE,
      &sparse_input_descriptor,
      &dense_input_descriptor,
      &res_descriptor,
      &res_descriptor,
      compute_type));

  // set bias pointer for matmul, need to assign to get location
  if (bias_opt.has_value()) {
    auto& bias = bias_opt.value();
    void* dBias = bias.data_ptr();
    TORCH_HIPSPARSE_CHECK(hipsparseLtMatmulDescSetAttribute(
        &handle, &matmul, HIPSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias)));
  }

  TORCH_HIPSPARSE_CHECK(hipsparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT));

  // set alg_id
  TORCH_HIPSPARSE_CHECK(hipsparseLtMatmulAlgSetAttribute(
      &handle, &alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));

  // set tensor_alpha_mode and alpha pointer for matmul
  const auto alpha_tensor = alpha_opt.has_value() ? *alpha_opt: Tensor{};
  const auto alpha_ptr = alpha_opt.has_value() ? alpha_tensor.data_ptr(): &alpha;
  if (alpha_opt.has_value()) {
    tensor_alpha_mode = 1;
    TORCH_HIPSPARSE_CHECK(hipsparseLtMatmulDescSetAttribute(
        &handle, &matmul, HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &tensor_alpha_mode, sizeof(tensor_alpha_mode)));
  }

  TORCH_HIPSPARSE_CHECK(
      hipsparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size;
  TORCH_HIPSPARSE_CHECK(
      hipsparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  auto& allocator = *::c10::zoom::ZoomCachingAllocator::get();
  auto workspacePtr = allocator.allocate(workspace_size);
  hipStream_t stream = c10::zoom::getCurrentZoomStream();

  if(search_alg_id){
    // run matmul search
    TORCH_HIPSPARSE_CHECK(hipsparseLtMatmulSearch(
        &handle,
        &plan,
        alpha_ptr,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        // jank because of the way we want this to be an array of streams
        &stream,
        1));

    // get alg_id used
    TORCH_HIPSPARSE_CHECK(hipsparseLtMatmulAlgGetAttribute(
        &handle, &alg_sel, HIPSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)));
  }
  else {
    // do normal matmul
    TORCH_HIPSPARSE_CHECK(hipsparseLtMatmul(
        &handle,
        &plan,
        alpha_ptr,
        compressed_A.data_ptr(),
        dense_B.data_ptr(),
        &beta,
        res.data_ptr(),
        res.data_ptr(),
        workspacePtr.get(),
        // jank because of the way we want this to be an array of streams
        &stream,
        1));
  }

  //destroy descriptors
  TORCH_HIPSPARSE_CHECK(
      hipsparseLtMatDescriptorDestroy(&sparse_input_descriptor));
  TORCH_HIPSPARSE_CHECK(
      hipsparseLtMatDescriptorDestroy(&dense_input_descriptor));
  TORCH_HIPSPARSE_CHECK(hipsparseLtMatDescriptorDestroy(&res_descriptor));
  // destroy plan
  TORCH_HIPSPARSE_CHECK(hipsparseLtMatmulPlanDestroy(&plan));

  return {alg_id, res};
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int64_t alg_id
)
{
    auto result = _cslt_sparse_mm_impl(
        compressed_A,
        dense_B,
        bias_opt,
        alpha_opt,
        out_dtype_opt,
        transpose_result,
        (int) alg_id,
        false);
    return std::get<1>(result);
}

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result
)
{
    int alg_id_int = 0;
    auto result = _cslt_sparse_mm_impl(
        compressed_A,
        dense_B,
        bias_opt,
        alpha_opt,
        out_dtype_opt,
        transpose_result,
        alg_id_int,
        true);
    return (int64_t) std::get<0>(result);
}


} // namespace at::native

#else // No hipSPARSELt support, throw error if these functions are called.

namespace at::native {

at::Tensor _cslt_compress(const Tensor& sparse_input){
    TORCH_CHECK(false, "hipSPARSELt not supported on your machine.");
}

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result,
    int64_t alg_id)
{
    TORCH_CHECK(false, "hipSPARSELt not supported on your machine.");
}

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype,
    bool transpose_result
)
{
    TORCH_CHECK(false, "hipSPARSELt not supported on your machine.");
}

} // namespace at::native

#endif
