#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/zoom/HIPGraphsUtils.hpp>
#include <ATen/Utils.h>

#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

namespace {

// See note [Algorithm of randperm]
template<typename T, typename scalar_t>
__global__ void randperm_handle_duplicate_keys_kernel(T *keys, scalar_t *data, T mask, int n, at::PhiloxHIPState philox_args) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  // find the beginning of islands
  if (tid >= n - 1) return;  // out of range
  if ((keys[tid] & mask) != (keys[tid + 1] & mask)) return;  // not in an island
  if (tid != 0 && (keys[tid] & mask) == (keys[tid - 1] & mask)) return;  // not the beginning of an island

  // find the size of islands
  int island_size = 0;
  do { island_size++; }
  while ((tid + island_size < n) && (keys[tid + island_size] & mask) == (keys[tid] & mask));

  // do random permutation inside each island.
  data += tid;
  auto seeds = at::zoom::philox::unpack(philox_args);
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(std::get<0>(seeds), tid, std::get<1>(seeds), &state);
  for (int i = island_size - 1; i > 0; i--) {
    unsigned int r = hiprand(&state) % (i + 1);
    if (i != r) {
      scalar_t tmp = data[i];
      data[i] = data[r];
      data[r] = tmp;
    }
  }
}

// See note [Algorithm of randperm]
template<typename T, typename scalar_t>
void randperm_handle_duplicate_keys(T *keys, scalar_t *data, int bits, int64_t n, c10::optional<at::Generator> &gen_) {
  auto gen = at::get_generator_or_default<at::ZoomGeneratorImpl>(gen_, at::zoom::detail::getDefaultZoomGenerator());
  int64_t counter_offset = n;
  at::PhiloxHIPState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_hip_state(counter_offset);
  }
  T mask = static_cast<T>((1UL << bits) - 1);
  randperm_handle_duplicate_keys_kernel<<<(n + 511) / 512, 512, 0, c10::zoom::getCurrentZoomStream()>>>(
    keys, data, mask, n, rng_engine_inputs);
  C10_ZOOM_KERNEL_LAUNCH_CHECK();
}

}