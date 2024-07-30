// !!! This is a file automatically generated by hipify!!!
#include <hip/hip_runtime.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/zoom/UniqueCub.cuh>

#include <ATen/zoom/ZoomContext.h>
#include <ATen/zoom/detail/KernelUtils.h>
#include <ATen/zoom/ZoomApplyUtils.cuh>
#include <ATen/zoom/cub.cuh>

#include <c10/core/DeviceArray.h>
#include <c10/util/Load.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native::internal {

namespace {

template <typename InputIteratorT>
__global__ void adjacent_difference_kernel(
    int64_t n,
    InputIteratorT input,
    int* output) {
  HIP_KERNEL_LOOP(i, n) {
    output[i] = i > 0 ? input[i] != input[i - 1] : 0;
  }
}

__global__ void scatter_kernel(
    int64_t n,
    const int64_t* input,
    const int64_t* indices,
    int64_t* output) {
  HIP_KERNEL_LOOP(i, n) {
    output[indices[i]] = input[i];
  }
}

template <typename scalar_t>
const scalar_t * wrap_input_iterator(const scalar_t *data) {
  return data;
}

struct LoadBoolOp {
  __device__ bool operator()(uint8_t x) const {
    return static_cast<bool>(x);
  }
};

auto wrap_input_iterator(const bool *data) {
  // See NOTE [Loading boolean values]
  LoadBoolOp op;
  return NO_ROCM(at_zoom_detail)::hipcub::TransformInputIterator<bool, LoadBoolOp, const uint8_t*, int>(
      reinterpret_cast<const uint8_t*>(data), op);
}

// A variation of compute_unique (defined in Unique.cu) that doesn't allow
// customizing equal and not_equal (CUB doesn't allow them).
template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> compute_unique(
    const Tensor& sorted,
    const Tensor& sorted_indices,
    const bool return_inverse,
    const bool return_counts,
    const bool consecutive) {
  int64_t num_inp = sorted.numel();
  auto options = sorted.options().dtype(kLong);
  auto data = wrap_input_iterator(sorted.const_data_ptr<scalar_t>());
  hipStream_t stream = c10::zoom::getCurrentZoomStream();

  // inverse indices
  Tensor inverse_indices;
  if (!return_inverse) {
    inverse_indices = at::empty({0}, options);
  } else {
    inverse_indices = at::empty(sorted.sizes(), options);
    Tensor inv_loc = consecutive ? at::empty({num_inp}, options.dtype(kInt))
                                 : inverse_indices;
    int* inv_loc_ptr = static_cast<int*>(inv_loc.mutable_data_ptr());
    const dim3 block =
        dim3(::min(static_cast<int64_t>(zoom::getApplyBlock().x), num_inp));
    dim3 grid;
    c10::DeviceIndex curDevice = -1;
    c10::zoom::GetDevice(&curDevice);
    zoom::getApplyGrid(num_inp, grid, curDevice);
   hipLaunchKernelGGL(( adjacent_difference_kernel), dim3(grid), dim3(block), 0, stream, 
        num_inp, data, inv_loc_ptr);
    C10_ZOOM_KERNEL_LAUNCH_CHECK();

    Tensor inv_loc_out =
        consecutive ? inverse_indices : at::empty({num_inp}, options);
    at::zoom::hipcub::inclusive_sum_truncating(
        inv_loc_ptr,
        inv_loc_out.mutable_data_ptr<int64_t>(),
        num_inp);

    if (!consecutive) {
      TORCH_INTERNAL_ASSERT(
          sorted_indices.defined(),
          "return_inverse is set to true, but sorted_indices is undefined. Send a bug report!");
     hipLaunchKernelGGL(( scatter_kernel), dim3(grid), dim3(block), 0, stream, 
          num_inp,
          inv_loc_out.const_data_ptr<int64_t>(),
          sorted_indices.const_data_ptr<int64_t>(),
          inverse_indices.mutable_data_ptr<int64_t>());
      C10_ZOOM_KERNEL_LAUNCH_CHECK();
    }
  }

  // unique and count
  Tensor data_out = at::empty({num_inp}, sorted.options());
  Tensor counts = at::empty({0}, options);
  Tensor length = at::empty({1}, options);
  int64_t num_out;
  if (!return_counts) {
    zoom::hipcub::unique(data, data_out.mutable_data_ptr<scalar_t>(), length.mutable_data_ptr<int64_t>(), num_inp);
    num_out = length.item<int64_t>();
  } else {
    counts.resize_(num_inp);
    at::zoom::hipcub::run_length_encode(
        data,
        data_out.mutable_data_ptr<scalar_t>(),
        counts.mutable_data_ptr<int64_t>(),
        length.mutable_data_ptr<int64_t>(),
        num_inp);
    num_out = length.item<int64_t>();
    counts.resize_(num_out);
  }

  data_out.resize_(num_out);
  return std::tuple<Tensor, Tensor, Tensor>(
      data_out, inverse_indices, counts);
}

} // namespace

// This function (and compute_unique above) are defined in a separate file from
// Unique.cu because for now ATen/cuda/cub.cuh can't be used together with
// thrust in the same compilation unit.

template <typename scalar_t>
struct UniqueCub {
  std::tuple<Tensor, Tensor, Tensor> operator() (
      const Tensor& self,
      const bool consecutive,
      const bool return_inverse,
      const bool return_counts) {
    hipStream_t stream = c10::zoom::getCurrentZoomStream();

    int64_t num_inp = self.numel();
    Tensor sorted;
    if (consecutive) {
      sorted = self;
    } else {
      sorted = at::empty(self.sizes(), self.options());
    }

    Tensor sorted_indices;
    if (!return_inverse) {
      if (!consecutive) {
        zoom::hipcub::radix_sort_keys(
          self.const_data_ptr<scalar_t>(),
          sorted.mutable_data_ptr<scalar_t>(),
          num_inp);
      }
    } else {
      if (!consecutive) {
        auto options = self.options().dtype(kLong);
        Tensor range = at::arange(0, num_inp, options);
        sorted_indices = at::empty({num_inp}, options);
        zoom::hipcub::radix_sort_pairs(
            self.const_data_ptr<scalar_t>(),
            sorted.mutable_data_ptr<scalar_t>(),
            range.const_data_ptr<int64_t>(),
            sorted_indices.mutable_data_ptr<int64_t>(),
            num_inp);
      }
    }

    return compute_unique<scalar_t>(
        sorted, sorted_indices, return_inverse, return_counts, consecutive);
  }
};

struct MapNumberOfTrueValues {
  __device__ int operator()(uint8_t x) const {
    return static_cast<bool>(x);
  }
};

C10_LAUNCH_BOUNDS_1(at::zoom::detail::HIP_NUM_THREADS)
__global__ void unique_bool_write_inverse_indices(
    const int numel,
    const int *num_true_p,
    const bool *self,
    int64_t *inverse_indices_out) {
  constexpr int false_idx = 0;
  const int num_true = *num_true_p;
  const int num_false = numel - num_true;
  const int true_idx = num_false > 0;

  HIP_KERNEL_LOOP(i, numel) {
    const auto value = c10::load(&self[i]);
    inverse_indices_out[i] = value ? true_idx : false_idx;
  }
}

C10_LAUNCH_BOUNDS_1(1)
__global__ void unique_bool_write_output(
    const int numel,
    const int *num_true_p,
    bool *values_out,
    int64_t *counts_out) {
  constexpr int false_idx = 0;
  const int num_true = *num_true_p;
  const int num_false = numel - num_true;
  const int true_idx = num_false > 0;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (num_false > 0) {
      values_out[false_idx] = false;
      counts_out[false_idx] = num_false;
    }
    if (num_true > 0) {
      values_out[true_idx] = true;
      counts_out[true_idx] = num_true;
    }
  }
}

template <>
struct UniqueCub<bool> {

  std::tuple<Tensor, Tensor, Tensor> operator() (
      const Tensor& self,
      const bool consecutive,
      const bool return_inverse,
      const bool return_counts) {
    auto stream = c10::zoom::getCurrentZoomStream();

    int64_t num_inp = self.numel();

    Tensor output, inverse_indices, counts;
    if (consecutive) {
      Tensor sorted_indices;
      return compute_unique<bool>(
          self, sorted_indices, return_inverse, return_counts, consecutive);
    }

    // Instead of sorting, we use a reduction to find the number of
    // true values and from that we can infer the number of false.
    // If either has a count of zero, we omit it from the output.
    auto allocator = zoom::getZoomDeviceAllocator();
    c10::DeviceArray<int> tmp_num_true(*allocator, 1);

    const bool* self_data = self.const_data_ptr<bool>();
    MapNumberOfTrueValues op;
    NO_ROCM(at_zoom_detail)::hipcub::TransformInputIterator<int, MapNumberOfTrueValues, const uint8_t*, int>
        data_iter(reinterpret_cast<const uint8_t*>(self_data), op);
    at::zoom::hipcub::reduce(data_iter, tmp_num_true.get(), num_inp,
                          NO_ROCM(at_zoom_detail)::hipcub::Sum{}, 0);

    auto options = self.options();
    output = at::empty({2}, self.options());
    counts = at::empty({2}, options.dtype(kLong));

   hipLaunchKernelGGL(( unique_bool_write_output), dim3(1), dim3(1), 0, stream, 
        num_inp,
        tmp_num_true.get(),
        output.mutable_data_ptr<bool>(),
        counts.mutable_data_ptr<int64_t>());
    C10_ZOOM_KERNEL_LAUNCH_CHECK();

    if (return_inverse) {
      using namespace at::zoom::detail;
      inverse_indices = at::empty(self.sizes(), options.dtype(kLong));
      dim3 block = HIP_NUM_THREADS;
      dim3 grid = GET_BLOCKS(num_inp);
     hipLaunchKernelGGL(( unique_bool_write_inverse_indices), dim3(grid), dim3(block), 0, stream, 
          num_inp,
          tmp_num_true.get(),
          self_data,
          inverse_indices.mutable_data_ptr<int64_t>());
      C10_ZOOM_KERNEL_LAUNCH_CHECK();
    }

    // Final sync to fix the output tensors shape
    int num_true = 0;
    c10::zoom::memcpy_and_sync(&num_true, tmp_num_true.get(), sizeof(int),
                              hipMemcpyDeviceToHost, stream);
    const int num_false = num_inp - num_true;
    const int num_out = ((num_true > 0) + (num_false > 0));
    output.resize_({num_out});
    counts.resize_({num_out});

    return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
  }
};

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_zoom_template(
    const Tensor& self,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts) {
  auto num_inp = self.numel();
  TORCH_CHECK(
      num_inp <= INT_MAX, "num_inp ", num_inp, " is too big to for CUB");
  if (num_inp == 0) {
    Tensor output = at::empty({0}, self.options());
    Tensor inverse_indices = at::empty(self.sizes(), self.options().dtype(kLong));
    Tensor counts = at::empty({0}, self.options().dtype(kLong));
    return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
  }

  auto self_c = self.expect_contiguous();
  return UniqueCub<scalar_t>{}(*self_c, consecutive, return_inverse, return_counts);
}

#define INSTANTIATE_UNIQUE_HIP_TEMPLATE(TYPE)                            \
  template std::tuple<Tensor, Tensor, Tensor> unique_zoom_template<TYPE>( \
      const Tensor& self,                                                 \
      const bool consecutive,                                             \
      const bool return_inverse,                                          \
      const bool return_counts)

INSTANTIATE_UNIQUE_HIP_TEMPLATE(uint8_t);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(int8_t);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(double);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(float);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(int32_t);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(int64_t);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(int16_t);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(uint32_t);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(uint64_t);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(uint16_t);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(bool);
INSTANTIATE_UNIQUE_HIP_TEMPLATE(at::Half);

#undef INSTANTIATE

} // namespace at::native::internal
