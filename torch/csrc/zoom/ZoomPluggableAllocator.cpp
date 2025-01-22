#include <c10/zoom/ZoomCachingAllocator.h>
#include <c10/zoom/ZoomGuard.h>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <torch/csrc/zoom/ZoomPluggableAllocator.h>

namespace torch::zoom::ZoomPluggableAllocator {

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata()
    : size(0), device_idx(-1), stream{} {}

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    c10::DeviceIndex device_idx,
    hipStream_t stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
ZoomPluggableAllocator::ZoomPluggableAllocator(
    std::function<void*(size_t, int, hipStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, hipStream_t)> free_fn)
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

ZoomPluggableAllocator::ZoomPluggableAllocator(ZoomPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) {}

void ZoomPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  init_fn_ = std::move(init_fn);
}

void ZoomPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  reset_fn_ = std::move(reset_fn);
}

void ZoomPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn) {
  memory_fraction_fn_ = std::move(memory_fraction_fn);
}

void ZoomPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn) {
  base_alloc_fn_ = std::move(base_alloc_fn);
}

void ZoomPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, hipStream_t stream)> record_stream_fn) {
  record_stream_fn_ = std::move(record_stream_fn);
}

void ZoomPluggableAllocator::set_begin_allocate_to_pool(
    std::function<
        void(int, c10::zoom::MempoolId_t, std::function<bool(hipStream_t)>)>
        capture_begin_fn) {
  begin_allocate_to_pool_fn_ = std::move(capture_begin_fn);
}

void ZoomPluggableAllocator::set_end_allocate_to_pool_fn(
    std::function<void(int, c10::zoom::MempoolId_t)> capture_about_to_end_fn) {
  end_allocate_to_pool_fn_ = std::move(capture_about_to_end_fn);
}

void ZoomPluggableAllocator::set_release_pool(
    std::function<void(int, c10::zoom::MempoolId_t)> capture_destroy_fn) {
  relase_pool_fn_ = std::move(capture_destroy_fn);
}

void* ZoomPluggableAllocator::malloc(
    size_t size,
    c10::DeviceIndex device,
    hipStream_t stream) {
  void* r = alloc_fn_(size, device, stream);
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  return r;
}

c10::DataPtr ZoomPluggableAllocator::allocate(size_t size) {
  c10::DeviceIndex device = -1;
  C10_ZOOM_CHECK(c10::zoom::GetDevice(&device));
  hipStream_t stream = c10::zoom::getCurrentZoomStream(device);
  void* r = this->malloc(size, device, stream);
  c10::DataPtr data_ptr = {
      r, r, raw_deleter(), c10::Device(c10::DeviceType::PrivateUse1, device)};
  return data_ptr;
}

c10::DeleterFnPtr ZoomPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void* ZoomPluggableAllocator::raw_alloc(size_t nbytes) {
  c10::DeviceIndex device = -1;
  C10_ZOOM_CHECK(c10::zoom::GetDevice(&device));
  hipStream_t stream = c10::zoom::getCurrentZoomStream(device);
  return malloc(nbytes, device, stream);
}

void* ZoomPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
    hipStream_t stream) {
  c10::DeviceIndex device = -1;
  C10_ZOOM_CHECK(c10::zoom::GetDevice(&device));
  return malloc(nbytes, device, stream);
}

void ZoomPluggableAllocator::raw_delete(void* ptr) {
  hipStream_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, stream);
}

void ZoomPluggableAllocator::init(int device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool ZoomPluggableAllocator::initialized() {
  return initialized_;
}

void ZoomPluggableAllocator::setMemoryFraction(
    double fraction,
    c10::DeviceIndex device) {
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

void ZoomPluggableAllocator::emptyCache() {
  if (reset_fn_) {
    return reset_fn_();
  }
}

void ZoomPluggableAllocator::cacheInfo(
    c10::DeviceIndex device,
    size_t* largestBlock) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* ZoomPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

void ZoomPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    streamType stream) {
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

c10::zoom::ZoomCachingAllocator::DeviceStats ZoomPluggableAllocator::
    getDeviceStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void ZoomPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void ZoomPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

c10::zoom::ZoomCachingAllocator::SnapshotInfo ZoomPluggableAllocator::
    snapshot() {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<void> ZoomPluggableAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// HIPGraph interactions
void ZoomPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    c10::zoom::MempoolId_t mempool_id,
    std::function<bool(hipStream_t)> filter) {
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

void ZoomPluggableAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    c10::zoom::MempoolId_t mempool_id) {
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

void ZoomPluggableAllocator::releasePool(
    c10::DeviceIndex device,
    c10::zoom::MempoolId_t mempool_id) {
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

void ZoomPluggableAllocator::recordHistory(
    bool enabled,
    c10::zoom::ZoomCachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10::zoom::ZoomCachingAllocator::RecordContext when) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void ZoomPluggableAllocator::attachOutOfMemoryObserver(
    c10::zoom::ZoomCachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

void ZoomPluggableAllocator::attachAllocatorTraceTracker(
    c10::zoom::ZoomCachingAllocator::AllocatorTraceTracker tracker) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not support attachAllocatorTraceTracker. "
      "attachAllocatorTraceTracker is only used inside Pytorch.");
}

std::shared_ptr<c10::zoom::ZoomCachingAllocator::AllocatorState>
ZoomPluggableAllocator::getCheckpointState(
    c10::DeviceIndex device,
    c10::zoom::MempoolId_t id) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support getCheckpointState. "
      "If you need it, please file an issue describing your use case.");
}

c10::zoom::ZoomCachingAllocator::CheckpointDelta ZoomPluggableAllocator::
    setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<c10::zoom::ZoomCachingAllocator::AllocatorState> pps) {
  TORCH_CHECK(
      false,
      "ZoomPluggableAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void ZoomPluggableAllocator::enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  c10::zoom::ZoomGuard device_guard(dev);
  hipError_t err = hipDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == hipErrorPeerAccessAlreadyEnabled) {
    // ignore and clear the error if access was already enabled
    (void)hipGetLastError();
  } else {
    C10_ZOOM_CHECK(err);
  }
}

hipError_t ZoomPluggableAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    hipStream_t stream,
    bool p2p_enabled) {
  return hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream);
}

std::string ZoomPluggableAllocator::name() {
  return "pluggable";
}

void ZoomPluggableAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  C10_ZOOM_CHECK(
      hipMemcpy(dest, src, count, hipMemcpyKind::hipMemcpyDeviceToDevice));
}

std::shared_ptr<c10::zoom::ZoomCachingAllocator::ZoomAllocator>
    current_custom_allocator;

std::shared_ptr<c10::zoom::ZoomCachingAllocator::ZoomAllocator>
getCurrentAllocator() {
  return current_custom_allocator;
}

// TODO: add more functions in the argument
std::shared_ptr<c10::zoom::ZoomCachingAllocator::ZoomAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, hipStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, hipStream_t)> free_fn) {
  std::shared_ptr<ZoomPluggableAllocator> allocator(
      new ZoomPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
  allocator->init(device_count);
  return allocator;
}

void changeCurrentAllocator(
    const std::shared_ptr<c10::zoom::ZoomCachingAllocator::ZoomAllocator>&
        allocator) {
  TORCH_CHECK(
      !c10::zoom::ZoomCachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  c10::zoom::ZoomCachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr) {
  current_custom_allocator->raw_delete(ptr);
}

} // namespace torch::zoom::ZoomPluggableAllocator
