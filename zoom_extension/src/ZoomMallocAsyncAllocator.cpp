#include "ZoomCachingAllocator.h"
#include "ZoomException.h"
#include "ZoomFunctions.h"
#include "ZoomGuard.h"
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>

#include <unordered_set>
#include <vector>

namespace c10::zoom::ZoomCachingAllocator::ZoomMallocAsync {

// CUDA device allocator that uses hipMallocAsync to implement
// the same interface as ZoomCachingAllocator.cpp.

// Designed to be safe for CUDA graph capture.
// Interactions with CUDA graph capture are mediated by
// notifyCaptureBegin
// notifyCaptureAboutToEnd
// notifyCaptureEnded
// notifyCaptureDestroy

// Implementation details, not declared in ZoomCachingAllocator.h
namespace {

// General helpers

struct UsageStream {
  hipStream_t stream;
  c10::DeviceIndex device;
  UsageStream() = default;
  UsageStream(hipStream_t s, c10::DeviceIndex d) : stream(s), device(d) {}
  UsageStream(const UsageStream& us) = default;
  UsageStream(const UsageStream&& us) noexcept
      : stream(us.stream), device(us.device) {}
  UsageStream& operator=(UsageStream other) {
    stream = other.stream;
    device = other.device;
    return *this;
  }
};

bool operator==(const UsageStream& lhs, const UsageStream& rhs) {
  return (lhs.stream == rhs.stream) && (lhs.device == rhs.device);
}

struct UsageStreamHash {
  size_t operator()(const UsageStream& us) const noexcept {
    return std::hash<void*>{}(us.stream) + size_t(us.device);
  }
};

struct PtrUsage {
  // recorded_streams holds side usage streams added by record_stream calls.
  // In other words, it does NOT include the original creation stream.
  ska::flat_hash_set<UsageStream, UsageStreamHash> recorded_streams;
  UsageStream creation_stream{};
  uint64_t size;
  bool captured;
  PtrUsage(uint64_t s, bool c) : size(s), captured(c) {}
};

int device_count = 0;
// these don't need to be c10::once_flags as in CUDAGeneratorImpl.cpp
// because they'll only be flipped by functions that have locked the mutex.
std::vector<bool> devs_initialized_flags;
std::vector<UsageStream> dummy_unifying_free_streams;

// Possible micro-optimization:
// Some accesses to ptr_info are read-only.
// We could let those be concurrent with a shared_mutex and
// have concurrent calls take a shared_lock.
// Keeping it simple with an ordinary mutex for now.
std::mutex general_mutex;

/**
 * Note [Avoid freeing uncaptured ptrs during CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * During CUDA graph capture, it's illegal to call hipFreeAsync
 * on a pointer that came from a non-captured hipMallocAsync.
 * Unfortunately, Python being what it is, it's impossible to be
 * sure no uncaptured tensor will ever have its destructor called
 * in a capturing region.
 * We avoid errors by
 *  1. remembering if allocated pointers were captured or uncaptured
 *  2. during capture, if we detect an attempt to free an uncaptured
 *     allocation on a capturing stream, don't free it immediately,
 *     just remember it and defer its hipFreeAsync call to after
 *     the end of capture (specifically, to notifyCaptureEnded).
 */

using PtrInfo = ska::flat_hash_map<void*, PtrUsage>;
PtrInfo ptr_info;
std::vector<void*> ungraphed_ptrs_defer_free_until_no_capture;

// These two help setMemoryFraction limit the amount of memory
// used by PyTorch in particular (as opposed to other libraries
// in the same process that might be sharing the same hipMemPool_t).
std::vector<size_t> pytorch_used_bytes;
std::vector<size_t> pytorch_memory_limits;

// Graph-specific helpers

/**
 * Note [Avoid dangling free streams during CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * During capture, all stream dependencies must branch out from
 * the stream on which capture began and rejoin this initial stream
 * before capture ends.
 * The user rigs desired forking and joining with event waits.
 * But it's hard to be sure when tensor destructors get called relative
 * to the final joins.
 * For example, suppose a user
 *   forks work stream B from initial capture stream A
 *   creates a tensor T in B
 *   joins by syncing A with B
 *   ends capture.
 * All well and good, right? Maybe not: maybe T went out of scope
 * and its destructor got called AFTER the rejoin, leaving the graph with
 * "unjoined work": a dangling hipFreeAsync node in stream B.
 * Ensuring that all tensor destructors for all side stream tensors
 * are called before side streams rejoin the main stream is
 * difficult. The user might have to add a bunch of explicit
 * "del"s at the right spots in code that was fine for ordinary
 * eager execution.
 * Fortunately, we can spare the user this burden:
 * during capture, we remember _all_ free streams,
 * and manually rejoin them with the capture stream during
 * notifyCaptureAboutToEnd.
 * This approach is heavy-handed, but hopefully capture only needs to
 * happen once, so we don't mind being heavy-handed.
 *
 * TODO: If, someday, we augment the graph bindings to support recapture
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#whole-graph-update
 * (eg, as a way to accommodate dynamic params) we should think more
 * carefully about the CPU overhead of remembering and rejoining
 * all free streams during capture. Maybe it's not a big deal.
 */
std::unordered_set<UsageStream, UsageStreamHash> capture_free_streams;
bool capture_underway = false;

// Implementation functions

// Assumes the caller holds general_mutex
inline void lazy_init_device(c10::DeviceIndex device) {
  if (!devs_initialized_flags[device]) {
    ZoomGuard g(device);

    // See "Retaining memory in the pool" here:
    // https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
    hipMemPool_t mempool = nullptr;
    C10_ZOOM_CHECK(hipDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    C10_ZOOM_CHECK(hipMemPoolSetAttribute(
        mempool, hipMemPoolAttrReleaseThreshold, &threshold));

    // I think all these are on by default, but I want to enable them
    // explicitly to ensure awareness.
    int enable = 1;
    C10_ZOOM_CHECK(hipMemPoolSetAttribute(
        mempool, hipMemPoolReuseFollowEventDependencies, &enable));
    C10_ZOOM_CHECK(hipMemPoolSetAttribute(
        mempool, hipMemPoolReuseAllowOpportunistic, &enable));
    C10_ZOOM_CHECK(hipMemPoolSetAttribute(
        mempool, hipMemPoolReuseAllowInternalDependencies, &enable));

    // Grabs a stream from the current device to use as the "unifier" free
    // stream for allocations that end up used on multiple streams.
    const auto dufs = getStreamFromPool();
    dummy_unifying_free_streams[device] =
        UsageStream(dufs.stream(), dufs.device_index());

    pytorch_used_bytes[device] = 0;
    pytorch_memory_limits[device] = UINT64_MAX;

    devs_initialized_flags[device] = true;
  }
}

inline void sync_raw(hipStream_t dependency, hipStream_t dependent) {
  // ZoomCachingAllocator.cpp uses raw hip events, as do we.
  hipEvent_t event = nullptr;
  C10_ZOOM_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  C10_ZOOM_CHECK(hipEventRecord(event, dependency));
  C10_ZOOM_CHECK(hipStreamWaitEvent(dependent, event, 0));
  C10_ZOOM_CHECK(hipEventDestroy(event));
}

// Assumes the caller holds general_mutex
inline void free_impl(PtrInfo::iterator& it) {
  // Possible micro-optimization: If we did a value-copy here, we could move
  // ptr_info.erase(it) up here and drop the lock immediately.
  const auto& recorded_streams = it->second.recorded_streams;
  const auto& creation_stream = it->second.creation_stream;

  // If the usage stream is a null (default) stream,
  // hipFreeAsync infers the device from the ambient context,
  // so we need to set the right ambient context.
  ZoomGuard g(creation_stream.device);

  if (recorded_streams.empty()) {
    // ptr was only used on one stream, which must have been
    // the original allocation stream.
    // Frees ptr in the original allocation stream.

    C10_ZOOM_CHECK(hipFreeAsync(it->first, creation_stream.stream));

    if (C10_UNLIKELY(capture_underway)) {
      // See Note [Avoid dangling free streams during CUDA graph capture]
      capture_free_streams.insert(creation_stream);
    }
  } else {
    // ptr was used on many streams. We don't know which was the most recent.
    // There could even have been multiple most recent usage streams acting
    // on different regions of the memory.
    // But hipFreeAsync only accepts a single most recent usage stream.
    // We can still safely free ptr with a trick:
    // Use a dummy "unifying stream", sync the unifying stream with all of
    // ptr's usage streams, and pass the dummy stream to hipFreeAsync.

    // Retrieves the dummy "unifier" stream from the device
    // on which the pointer was originally allocated.
    auto dummy_unifying_free_stream =
        dummy_unifying_free_streams[creation_stream.device];
    TORCH_INTERNAL_ASSERT(
        dummy_unifying_free_stream.device == creation_stream.device);

    // we're already on creation_stream.device, no need to re-guard
    sync_raw(creation_stream.stream, dummy_unifying_free_stream.stream);

    // The number of usage streams is typically small (low single digits)
    for (const auto& recorded_stream : recorded_streams) {
      // Logic here accommodates the chance some of the usage streams were on
      // other devices, which is possible if some usage kernels accessed the
      // memory via p2p.

      // hipEventRecord requires that the input event and stream are on the
      // same device.
      ZoomGuard g_usage(recorded_stream.device);

      sync_raw(recorded_stream.stream, dummy_unifying_free_stream.stream);
    }

    // Frees ptr in the dummy "unifier" stream.
    C10_ZOOM_CHECK(hipFreeAsync(it->first, dummy_unifying_free_stream.stream));
    // At this point, unless dummy_unifying_free_stream happens to alias some
    // future user stream, the allocation is only available for "opportunistic"
    // reuse, ie, if the CPU sees dummy_unifying_free_stream has reached the
    // point that all events recorded on all usage streams have resolved from
    // the CPU's perspective. In theory, we could remove the need for the driver
    // to do this tracking by e.g. replacing
    // hipStreamWaitEvent(dummy_unifying_free_stream.stream, event);
    // with
    // hipStreamWaitEvent(creation_stream.stream, event);
    // then hipFreeAsyncing straight back into creation_stream.stream,
    // but this forces a potentially false dependency of creation_stream.stream
    // on all the recorded_streams.

    if (C10_UNLIKELY(capture_underway)) {
      // See Note [Avoid dangling free streams during CUDA graph capture]
      capture_free_streams.emplace(
          dummy_unifying_free_stream.stream, dummy_unifying_free_stream.device);
    }
  }

  pytorch_used_bytes[creation_stream.device] -= it->second.size;

  ptr_info.erase(it);
}

void freeAsync(void* ptr) {
  std::lock_guard<std::mutex> lk(general_mutex);

  auto err = hipGetLastError();
  C10_ZOOM_CHECK(err);
  auto it = ptr_info.find(ptr);
  TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");

  if (C10_UNLIKELY(capture_underway)) {
    if (!it->second.captured) {
      TORCH_WARN_ONCE(
          "freeAsync() was called on an uncaptured allocation during graph capture "
          "(address = ",
          ptr,
          "). This may be benign, for example, a Python tensor in the capture "
          "might happen to shadow (use the same name as) an unrelated temporary "
          "tensor from somewhere before capture, pushing the earlier tensor "
          "out of scope. "
          "However, if the tensor we're freeing here IS used by the capture, "
          "freeing it is an error, and may cause illegal memory accesses or "
          "memory corruption during graph replay.");
      // See Note [Avoid freeing uncaptured ptrs during CUDA graph capture]
      // Remembers the raw pointer, not the iterator.
      // This forces notifyCaptureEnded to do another lookup,
      // but avoids the risk the iterator might be invalidated
      // between now and then.
      ungraphed_ptrs_defer_free_until_no_capture.push_back(ptr);
      return;
    }
  } else if (C10_UNLIKELY(it->second.captured)) {
    TORCH_WARN(
        "Attempting uncaptured free of a captured allocation with address ",
        ptr,
        "\nThis is technically allowed, but may indicate you are losing "
        "the last user-visible tensor through which the allocation can "
        "be accessed, so you'll have no way to view the data after "
        "future replays of the owning graph.");
  }

  free_impl(it);
}

// Symmetric with NativeCachingAllocator::malloc for now,
// although I don't think we absolutely need the symmetry.
void mallocAsync(
    void** devPtr,
    c10::DeviceIndex device,
    size_t size,
    hipStream_t stream) {
  TORCH_INTERNAL_ASSERT(
      0 <= device && device < device_count,
      "Invalid device index ",
      device,
      ": did you call init?");

  // If stream is a null (default) stream,
  // hipMallocAsync infers the device from the ambient context,
  // so we need to set the right ambient context.
  ZoomGuard g(device);

  std::lock_guard<std::mutex> lk(general_mutex);

  if (!capture_underway &&
      !ungraphed_ptrs_defer_free_until_no_capture.empty()) {
    // See Note [Avoid freeing uncaptured ptrs during CUDA graph capture]
    for (const auto ptr : ungraphed_ptrs_defer_free_until_no_capture) {
      auto it = ptr_info.find(ptr);
      TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");
      free_impl(it);
    }

    ungraphed_ptrs_defer_free_until_no_capture.clear();
  }

  lazy_init_device(device);

  // Defensively checks for preexisting CUDA error state.
  auto err = hipGetLastError();
  C10_ZOOM_CHECK(err);

  // TODO: Could we avoid calling hipMallocAsync while holding general_mutex,
  // perhaps by letting lazy_init_device use separate once_flags or an internal
  // static initializer?
  if (pytorch_used_bytes[device] + size > pytorch_memory_limits[device]) {
    err = hipErrorMemoryAllocation;
  } else {
    err = hipMallocAsync(devPtr, size, stream);
  }

  if (err == hipErrorMemoryAllocation) {
    // Clears CUDA's internal error state so the user, if desired, can catch the
    // OOM exception, free some stuff on the script side, and retry the
    // allocation. This aligns with the behavior of alloc_block in
    // ZoomCachingAllocator.cpp.
    (void)hipGetLastError(); // clear CUDA error
    size_t device_free = 0;
    size_t device_total = 0;
    C10_ZOOM_CHECK(hipMemGetInfo(&device_free, &device_total));
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        false,
        "Allocation on device ",
        device,
        " would exceed allowed memory. (out of memory)",
        "\nCurrently allocated     : ",
        format_size(pytorch_used_bytes[device]),
        "\nRequested               : ",
        format_size(size),
        "\nDevice limit            : ",
        format_size(device_total),
        "\nFree (according to CUDA): ",
        format_size(device_free),
        "\nPyTorch limit (set by user-supplied memory fraction)"
        "\n                        : ",
        format_size(pytorch_memory_limits[device]));
  } else {
    C10_ZOOM_CHECK(err);
  }

  auto inserted = ptr_info.emplace(*devPtr, PtrUsage(size, capture_underway));
  TORCH_INTERNAL_ASSERT(
      inserted.second,
      "address returned by hipMallocAsync already exists "
      "in ptr_info");

  inserted.first->second.creation_stream = {stream, device};

  pytorch_used_bytes[device] += size;
}

} // anonymous namespace

void local_raw_delete(void* ptr);

// Same pattern as ZoomCachingAllocator.cpp.
struct ZoomMallocAsyncAllocator : public ZoomAllocator {
  DataPtr allocate(size_t size) override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "HIP out of memory. Tried to allocate more than 1EB memory.");
    c10::DeviceIndex device = 0;
    C10_ZOOM_CHECK(c10::zoom::GetDevice(&device));
    void* r = nullptr;
    if (size != 0) {
      mallocAsync(&r, device, size, zoom::getCurrentZoomStream(device));
    }
    return {r, r, &local_raw_delete, Device(DeviceType::PrivateUse1, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    return &local_raw_delete;
  }

  // This function should not issue any context-creating calls,
  // just set up for later calls to init per-device pools based
  // on the current device each later call sees.
  void init(int dev_count) override {
    static bool called = [](int dev_count) {
      ;
      // Are there external guarantees init will be called before
      // any of the allocator's other functions?
      // std::lock_guard<std::mutex> lk(general_mutex);
      device_count = dev_count;
      devs_initialized_flags.resize(dev_count, false);
      dummy_unifying_free_streams.resize(dev_count);
      pytorch_used_bytes.resize(dev_count);
      pytorch_memory_limits.resize(dev_count);
      return true;
    }(dev_count);
    (void)called;
  }

  bool initialized() override {
    return !devs_initialized_flags.empty();
  }

  static inline void assertValidDevice(c10::DeviceIndex device) {
    TORCH_CHECK(
        0 <= device && device < device_count, "Invalid device argument.");
  }

  void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");

    std::lock_guard<std::mutex> lk(general_mutex);
    assertValidDevice(device);
    ZoomGuard g(device);
    // Should setMemoryFraction be allowed to trigger a full device context and
    // pool-creating lazy_init_device, or should we simply assert this device is
    // already initialized, ie
    // TORCH_CHECK(devs_initialized_flags[device], ...)?
    lazy_init_device(device);

    size_t device_free = 0;
    size_t device_total = 0;
    C10_ZOOM_CHECK(hipMemGetInfo(&device_free, &device_total));
    pytorch_memory_limits[device] =
        static_cast<uint64_t>(fraction * static_cast<double>(device_total));

    // Alternative: Instead of a manual hard limit, we could use
    // hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold,
    // &threshold); This is a soft hint: The driver allows the pool's reserved
    // memory to spike above threshold in regions of high hipMallocAsync
    // demand, but opportunistically trims reserved memory back to threshold
    // when the memory in use is < threshold. I don't like this because it
    // introduces performance nondeterminism.
  }

  void emptyCache() override {
    std::lock_guard<std::mutex> lk(general_mutex);

    for (int dev = 0; dev < device_count; dev++) {
      if (devs_initialized_flags[dev]) {
        ZoomGuard g(static_cast<c10::DeviceIndex>(dev));

        hipMemPool_t mempool = nullptr;
        hipDeviceGetDefaultMemPool(&mempool, dev);
        hipDeviceSynchronize();
        hipMemPoolTrimTo(mempool, 0);
      }
    }
  }

  void cacheInfo(c10::DeviceIndex device, size_t* maxWorkspaceGuess) override {
    // The only consumer of cacheInfo is getMaxWorkspaceSize in Conv_v7.cpp.
    // Afaict, the role of cacheInfo is to give getMaxWorkspaceSize a reasonable
    // maximum workspace size to use for an upcoming cudnnFind call.
    //
    // The native allocator's cacheInfo chooses to return the size of its
    // largest unused block (which is the largest allocation the native
    // allocator can service immediately and asynchronously without a
    // hipMalloc.
    //
    // Here, we use a different heuristic: figure out the max usable workspace
    // size with a bit of educated trial and error. It's ok to be
    // perf-inefficient because cacheInfo is a prelude to cudnnFind.
    //
    // The algo cache then stores the best-performing algo with workspace <=
    // maxWorkspaceGuess. Later calls with the same param set hit in cache and
    // try to allocate the same workspace. If, in one of those future calls,
    // workspace allocation fails (ie because less ambient memory is available),
    // the bindings rerun cudnnFind, including calling cacheInfo again
    // beforehand to estimate a new (smaller) largest-available workspace. Over
    // a few such calls, the cache should settle to the algo with a workspace
    // size that's small enough to succeed every time (for that param set).
    //
    // So the strategy here is to return a rough, largeish guess and let the
    // bindings retry to trim as needed over time.
    //
    // The only caveat is, even if a workspace is allocated without OOM errors
    // now and in future calls, it's hard to be sure those later error-free
    // hipMallocAsyncs are fast and come straight from the pool (ie,
    // hipMallocAsync didn't need to reserve more memory from the system).
    // Hopefully, after repeated workspace requests, the pool's reserved memory
    // also stabilizes to a point where they all come straight from the pool.
    std::lock_guard<std::mutex> lk(general_mutex);
    assertValidDevice(device);
    ZoomGuard g(device);
    lazy_init_device(device);

    size_t free_upper_bound = 0;
    size_t device_total = 0;
    C10_ZOOM_CHECK(hipMemGetInfo(&free_upper_bound, &device_total));
    TORCH_INTERNAL_ASSERT(
        free_upper_bound + pytorch_used_bytes[device] <= device_total);
    size_t guess = std::min(
        free_upper_bound,
        pytorch_memory_limits[device] - pytorch_used_bytes[device]);
    auto stream = c10::zoom::getCurrentZoomStream();
    void* dummy = nullptr;

    // Defensively checks for preexisting CUDA error state.
    auto err = hipGetLastError();
    C10_ZOOM_CHECK(err);

    while (true) {
      // Duplicates some logic from mallocAsync to work with the error state
      // directly instead of repeatedly catching an exception thrown by
      // mallocAsync.
      if (pytorch_used_bytes[device] + guess > pytorch_memory_limits[device]) {
        err = hipErrorMemoryAllocation;
      } else {
        err = hipMallocAsync(&dummy, guess, stream);
      }

      if (err == hipSuccess) {
        hipFreeAsync(dummy, stream);
        *maxWorkspaceGuess = guess;
        return;
      } else if (err == hipErrorMemoryAllocation) {
        (void)hipGetLastError(); // clear CUDA error
        guess >>= 1; // quick and dirty: try half the size next iteration
      } else {
        C10_ZOOM_CHECK(err);
      }
    }
  }

  void* getBaseAllocation(void* ptr, size_t* size) override {
    std::lock_guard<std::mutex> lk(general_mutex);

    auto it = ptr_info.find(ptr);
    TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");

    if (size) {
      *size = it->second.size;
    }

    return ptr;
  }

  void recordStream(const DataPtr& ptr, zoom::ZoomStream stream) override {
    std::lock_guard<std::mutex> lk(general_mutex);
    auto ptr_val = ptr.get();
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr_val) {
      return;
    }

    // The pointer should exist in the map already.
    auto it = ptr_info.find(ptr_val);
    TORCH_INTERNAL_ASSERT(it != ptr_info.end(), "ptr not found in ptr_info");

    UsageStream to_record{stream.stream(), stream.device_index()};
    if (to_record == it->second.creation_stream) {
      TORCH_WARN_ONCE(
          "Called record_stream on tensor whose original creation stream "
          "matches the recorded stream. This is unnecessary and has no effect.");
    } else {
      it->second.recorded_streams.insert(to_record);
    }
  }

  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    TORCH_CHECK(
        false,
        "hipMallocAsync does not yet support getIpcDevPtr. "
        "If you need it, please file an issue describing your use case.");
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) override {
    TORCH_CHECK(
        false,
        "hipMallocAsync does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    TORCH_CHECK(
        false,
        "hipMallocAsync does not yet support attachOutOfMemoryObserver. "
        "If you need it, please file an issue describing your use case.");
  }

  void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
    TORCH_CHECK(
        false,
        "hipMallocAsync does not yet support attachAllocatorTraceTracker. "
        "If you need it, please file an issue describing your use case.");
  }

  std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) override {
    TORCH_CHECK(
        false,
        "hipMallocAsync does not yet support getCheckpointState. "
        "If you need it, please file an issue describing your use case.");
  }

  CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> pps) override {
    TORCH_CHECK(
        false,
        "hipMallocAsync does not yet support setCheckpointPoolState. "
        "If you need it, please file an issue describing your use case.");
  }

  // Collects stats for device.
  // If device hasn't been used yet, returns 0s without creating a context.
  DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    assertValidDevice(device);

    // Memory currently reserved by the mempool
    uint64_t reserved_mem_current = 0;
    // High-water mark of memory reserved by the mempool since last reset
    uint64_t reserved_mem_peak = 0;
    // Memory currently in use by the mempool
    uint64_t used_mem_current = 0;
    // High-water mark of memory
    uint64_t used_mem_peak = 0;

    std::lock_guard<std::mutex> lk(general_mutex);

    if (devs_initialized_flags[device]) {
      ZoomGuard g(device);

      hipMemPool_t mempool = nullptr;
      C10_ZOOM_CHECK(hipDeviceGetDefaultMemPool(&mempool, device));
      C10_ZOOM_CHECK(hipMemPoolGetAttribute(
          mempool, hipMemPoolAttrReservedMemCurrent, &reserved_mem_current));

      C10_ZOOM_CHECK(hipMemPoolGetAttribute(
          mempool, hipMemPoolAttrReservedMemHigh, &reserved_mem_peak));

      C10_ZOOM_CHECK(hipMemPoolGetAttribute(
          mempool, hipMemPoolAttrUsedMemCurrent, &used_mem_current));

      C10_ZOOM_CHECK(hipMemPoolGetAttribute(
          mempool, hipMemPoolAttrUsedMemHigh, &used_mem_peak));
    }

    // Many stat types are specific to the native allocator. We leave these
    // untouched. Their "struct Stat"s will contain zeroed values.
    DeviceStats stats;

    // In the native allocator:
    // allocated_bytes is the total bytes of blocks that have been malloc()ed
    // and not yet free()d.
    // active_bytes is the total bytes of blocks that have been malloc()ed but
    // not yet released back into a free pool. In other words, it includes all
    // allocated_bytes, as well as the bytes of "limbo state" blocks had have
    // already been free()ed but not yet free_block()ed back into a pool due to
    // outstanding stream_uses.
    //
    // Here, in the hipMallocAsync allocator:
    // We simply ask the driver's opinion about active memory.
    // We don't bother distinguishing between allocated_bytes and active_bytes.
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        static_cast<int64_t>(used_mem_current);
    stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        static_cast<int64_t>(used_mem_peak);
    stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        static_cast<int64_t>(used_mem_current);
    stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        static_cast<int64_t>(used_mem_peak);
    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current =
        static_cast<int64_t>(reserved_mem_current);
    stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak =
        static_cast<int64_t>(reserved_mem_peak);

    return stats;
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    assertValidDevice(device);
    TORCH_WARN_ONCE(
        "For backend:hipMallocAsync, resetAccumulatedStats has no effect.");
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    assertValidDevice(device);

    ZoomGuard g(device);
    hipMemPool_t mempool = nullptr;
    C10_ZOOM_CHECK(hipDeviceGetDefaultMemPool(&mempool, device));
    // Using zero as the reset value is the method recommended by Cuda driver
    // team. Vivek Kini says:
    //   "Resetting to zero (which is the only valid value when setting
    //    ReservedMemHigh) resets it to ReservedMemCurrent inside the driver
    //   (same goes for UsedMemHigh/UsedMemCurrent)"
    uint64_t zero = 0;
    C10_ZOOM_CHECK(hipMemPoolSetAttribute(
        mempool, hipMemPoolAttrReservedMemHigh, &zero));
    C10_ZOOM_CHECK(
        hipMemPoolSetAttribute(mempool, hipMemPoolAttrUsedMemHigh, &zero));
  }

  SnapshotInfo snapshot() override {
    TORCH_CHECK(
        false,
        "Calling snapshot with backend:hipMallocAsync is not meaningful. "
        "(For backend:native, snapshot returns a detailed summary of all "
        "blocks tracked by the allocator, but the hipMallocAsync backend "
        "does not track individual blocks.)");
    // Alternative: TORCH_WARN
    return {};
  }

  // CUDAGraph interactions
  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(hipStream_t)>) override {
    std::lock_guard<std::mutex> lk(general_mutex);

    TORCH_INTERNAL_ASSERT(capture_free_streams.empty());
    TORCH_CHECK(
        !capture_underway,
        "Only one capture at a time is allowed in a process.")
    capture_underway = true;
  }

  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id)
      override {
    assertValidDevice(device);

    std::lock_guard<std::mutex> lk(general_mutex);

    TORCH_CHECK(
        capture_underway,
        "hipMallocAsync::notifyCaptureAboutToEnd called, "
        "but hipMallocAsync::capture_underway is false.");

    auto capture_stream = zoom::getCurrentZoomStream(device);

    // See Note [Avoid dangling free streams during CUDA graph capture]
    for (const auto& free_stream : capture_free_streams) {
      // hipEventRecord requires that the input event and stream are on the
      // same device.
      ZoomGuard g(free_stream.device);

      // ZoomCachingAllocator.cpp uses raw hip events, as do we.
      hipEvent_t event = nullptr;
      C10_ZOOM_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
      C10_ZOOM_CHECK(hipEventRecord(event, free_stream.stream));
      C10_ZOOM_CHECK(hipStreamWaitEvent(capture_stream.stream(), event, 0));
      C10_ZOOM_CHECK(hipEventDestroy(event));
    }

    capture_free_streams.clear();
    TORCH_CHECK(
        capture_underway,
        "hipMallocAsync::notifyCaptureEnded called, "
        "but hipMallocAsync::capture_underway is false.");
    capture_underway = false;
  }

  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    // Q: Do we need to do anything special here, like clear long-lived
    //    pointers created during the original capture (for example,
    //    tensors intended as the graph's I/O surface) that might still
    //    be resident in ptr_info?
    // A: I don't think so.
    //    Those allocations survived capture because the user held
    //    explicit tensor references to them,
    //    Those tensors' destructors will call freeAsync() on each pointer
    //    when the user is done with them.
    //    The freeAsync()s will probably incur
    //    TORCH_WARN("Attempting uncaptured free of a captured allocation..."
    //    but stale ptrs will not permanently leak into ptr_info.
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    c10::DeviceIndex device = 0;
    C10_ZOOM_CHECK(c10::zoom::GetDevice(&device));
    void* r = nullptr;
    mallocAsync(&r, device, nbytes, zoom::getCurrentZoomStream(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, hipStream_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    c10::DeviceIndex device = 0;
    C10_ZOOM_CHECK(c10::zoom::GetDevice(&device));
    void* r = nullptr;
    mallocAsync(&r, device, nbytes, stream);
    return r;
  }
  void raw_delete(void* ptr) override {
    freeAsync(ptr);
  }
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override {
    // Double-checks allocator backend hasn't changed, which would definitely be
    // an error. hipMallocAsync pools are unaffected by
    // hipDeviceEnablePeerAccess. We need pool-specific enablement. See
    // https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/
    c10::zoom::ZoomGuard device_guard(dev);
    hipMemPool_t mempool = nullptr;
    C10_ZOOM_CHECK(hipDeviceGetDefaultMemPool(&mempool, dev_to_access));
    hipMemAccessDesc desc = {};
    desc.location.type = hipMemLocationTypeDevice;
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    desc.location.id = dev;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    C10_ZOOM_CHECK(hipMemPoolSetAccess(mempool, &desc, 1 /* numDescs */));
  }
  hipError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      hipStream_t stream,
      bool p2p_enabled) override {
    if (p2p_enabled || dstDevice == srcDevice) {
      return hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream);
    } else {
      return hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
    }
  }
  std::string name() override {
    return "hipMallocAsync";
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    C10_ZOOM_CHECK(
        hipMemcpy(dest, src, count, hipMemcpyKind::hipMemcpyDeviceToDevice));
  }
};

ZoomMallocAsyncAllocator device_allocator;

void local_raw_delete(void* ptr) {
  freeAsync(ptr);
}
ZoomAllocator* allocator() {
  return &device_allocator;
}


} // namespace c10::zoom::ZoomCachingAllocator::ZoomMallocAsync