#include "CachingHostAllocator.h"

#include <ATen/DeviceGuard.h>
#include <ATen/zoom/ZoomEvent.h>
#include <c10/core/thread_pool.h>
#include <c10/zoom/ZoomAllocatorConfig.h>

#include <hip/hip_runtime.h>
#include <future>

namespace at::zoom {
namespace {

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<
      at::zoom::ZoomEvent,
      std::function<void(at::zoom::ZoomEvent*)>>;
  EventPool() : pools_(c10::zoom::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](at::zoom::ZoomEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<at::zoom::ZoomEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<at::zoom::ZoomEvent>(hipEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<at::zoom::ZoomEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

using Block = HostBlock<c10::zoom::ZoomStream>;

struct ZoomCachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<c10::zoom::ZoomStream, EventPool::Event> {
 private:
  void allocate_host_memory(size_t size, void** ptr) override {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::zoom::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::PrivateUse1, *primary_ctx_device_index));
    }

    if (c10::zoom::ZoomCachingAllocator::ZoomAllocatorConfig::
            pinned_use_zoom_host_register()) {
      allocWithZoomHostRegister(ptr, size);
    } else {
      // Use hipHostMalloc for allocating pinned memory (global lock in driver)
      C10_ZOOM_CHECK(hipHostMalloc(ptr, size, hipHostMallocDefault));
    }
  }

  void free_block(Block* block) override {
    if (c10::zoom::ZoomCachingAllocator::ZoomAllocatorConfig::
            pinned_use_zoom_host_register()) {
      void* ptr = block->ptr_;
      C10_ZOOM_CHECK(hipHostUnregister(ptr));
      free(ptr);
    } else {
      C10_ZOOM_CHECK(hipHostFree(block->ptr_));
    }
  }

  void record_stream(
      std::optional<std::vector<EventPool::Event>>& events,
      c10::zoom::ZoomStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(EventPool::Event& event) override {
    hipError_t err = hipEventQuery(*event);
    if (err == hipErrorNotReady) {
      (void)hipGetLastError(); // clear CUDA error
      return false;
    } else if (err != hipSuccess) {
      C10_ZOOM_CHECK(err);
    }
    return true;
  }

  EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  TaskThreadPool* getThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(
        c10::zoom::ZoomCachingAllocator::ZoomAllocatorConfig::
            pinned_max_register_threads());
    return pool;
  }

  void mapPagesForRegister(
      const void* ptr,
      size_t size,
      size_t i,
      size_t numThreads,
      size_t pageSize) {
    uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
    uintptr_t end = (uintptr_t)start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart =
        (((uintptr_t)start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < ((uintptr_t)end); p += pageSize) {
      memset((void*)p, 0, 1);
    }
  }

  void registerPages(const void* ptr, size_t size) {
    C10_ZOOM_CHECK(
        hipHostRegister((void*)ptr, (size_t)size, hipHostRegisterDefault));

    // If host and device pointer don't match, give a warning and exit
    void* devptr;
    C10_ZOOM_CHECK(hipHostGetDevicePointer(&devptr, (void*)ptr, 0));
    TORCH_CHECK(
        (void*)devptr == (void*)ptr,
        "Host and device pointer dont match with hipHostRegister. "
        "Please dont use this feature by setting "
        "PYTORCH_ZOOM_ALLOC_CONF=use_zoom_host_register:False (default)",
        "");
  }

  void allocWithZoomHostRegister(void** ptr, size_t roundSize) {
    // Here we do regular allocation, pre-fault/map the pages, and then do
    // cudaHostRegister with GPU mapping flags to lock the pages, so we
    // can minimize the cost for the cuda global lock.
    *ptr = malloc(roundSize);

    // Parallelize the mapping/registering of pages to reduce wall time
    size_t pageSize = (1 << 12); // 4kB pages
    size_t numMapThreads = c10::zoom::ZoomCachingAllocator::
        ZoomAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // parallelize the mapping of pages with a threadpool
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this,
                     i,
                     ptr,
                     roundSize,
                     numMapThreads,
                     pageSize,
                     &promises]() mutable {
          mapPagesForRegister(
              *ptr,
              roundSize,
              i, // thread task-id
              numMapThreads,
              pageSize);
          // set the promise when mapping pages are done
          promises[i].set_value();
        };
        pool->run(task);
      }
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // Map pages in the same thread
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

    // Register the mapped pages using cudaHostRegister
    registerPages(*ptr, roundSize);
  }
};

void raw_local_deleter(void* ptr);

struct ZoomCachingHostAllocator final
    : public CachingHostAllocatorInterface<ZoomCachingHostAllocatorImpl> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

ZoomCachingHostAllocator caching_host_allocator;

static inline ZoomCachingHostAllocator& getZoomCachingHostAllocator() {
  return caching_host_allocator;
}

void raw_local_deleter(void* ptr) {
  getZoomCachingHostAllocator().free(ptr);
}

} // anonymous namespace

bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::zoom::ZoomStream stream) {
  return getZoomCachingHostAllocator().record_event(ptr, ctx, stream);
}

// Releases cached pinned memory allocations via cudaHostFree
void CachingHostAllocator_emptyCache() {
  getZoomCachingHostAllocator().empty_cache();
}

at::Allocator* getCachingHostAllocator() {
  return &getZoomCachingHostAllocator();
}

} // namespace at::zoom