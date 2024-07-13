#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/zoom/HIPGraph.h>
#include <c10/zoom/ZoomException.h>
#include <ATen/Functions.h>
#include <c10/zoom/ZoomCachingAllocator.h>
#include <c10/zoom/ZoomFunctions.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace at::zoom {

static bool _hip_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

MempoolId_t graph_pool_handle() {
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by graph_pool_handle".
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uid++};
}


// Get the expected id of a capture sequence so that we can call beginAllocateStreamToPool
// before starting a graph capture
CaptureId_t capture_sequence_id() {
  // id starts at 1:
  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by cudaStreamGetCaptureInfo".
  // (But how do we know GetCaptureInfo never sets id_ to 0? Because that's the current behavior,
  // and I asked cuda devs to keep it that way, and they agreed.)
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in ZoomGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in ZoomCachingAllocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> HIPGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void HIPGraph::inc_pending_event_queries() {
  pending_event_queries++;
}

void HIPGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
    "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int HIPGraph::num_pending_event_queries() {
  return pending_event_queries;
}

HIPGraph::HIPGraph()
  // CUDAStreams may not be default-constructed.
  : capture_stream_(c10::zoom::getCurrentZoomStream()) {
}

void HIPGraph::register_generator_state(
    c10::intrusive_ptr<at::ZoomGeneratorState> state) {
  captured_generator_states_[std::move(state)] = 0;
}

void HIPGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<ZoomGeneratorImpl> zoom_gen =
      dynamic_intrusive_pointer_cast<ZoomGeneratorImpl>(
          generator.getIntrusivePtr());
  zoom_gen->register_graph(this);
}

void HIPGraph::capture_begin(MempoolId_t pool/*=0*/, hipStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_,
              "This HIPGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // default generator is always registered
  auto* gen = get_generator_or_default<ZoomGeneratorImpl>(
      c10::nullopt, zoom::detail::getDefaultZoomGenerator());
  gen->register_graph(this);

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = c10::zoom::getCurrentZoomStream();

  TORCH_CHECK(stream != c10::zoom::getDefaultZoomStream(),
              "HIP graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::zoom::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our mempool_id_.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::zoom::ZoomCachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](hipStream_t stream) {
      hipStreamCaptureStatus status;
      CaptureId_t stream_capture_id;
      C10_ZOOM_CHECK(hipStreamGetCaptureInfo(stream, &status, &stream_capture_id));
      return status == hipStreamCaptureStatus::hipStreamCaptureStatusActive && stream_capture_id == capture_id_;
  });

  // At this point, any NCCL watchdogs should be aware that we are in capture mode
  // and therefore should not enqueue any additional work that could be event-queried.
  // We still must wait on any existing work that has not been cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
    std::this_thread::sleep_for(
      std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  C10_ZOOM_CHECK(hipStreamBeginCapture(capture_stream_, capture_mode));

  hipStreamCaptureStatus status;
  C10_ZOOM_CHECK(hipStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == hipStreamCaptureStatus::hipStreamCaptureStatusActive);

  TORCH_INTERNAL_ASSERT(id_ > 0);
}

void HIPGraph::capture_end() {
  auto stream = c10::zoom::getCurrentZoomStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  C10_ZOOM_CHECK(hipStreamEndCapture(capture_stream_, &graph_));

  c10::zoom::ZoomCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
  // between replays.
  // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
  // is cudaMallocAsync.
  // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
  // the graph's internal bookkeeping requires that we instantiate with
  // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
  // cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233

    // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
    // who prefer not to report error message through these arguments moving forward
    // (they prefer return value, or errors on api calls internal to the capture)

    C10_ZOOM_CHECK(hipGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));


  has_graph_exec_ = true;

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numHIPGraphNodes = 0;
  C10_ZOOM_CHECK(hipGraphGetNodes(graph_, NULL, &numHIPGraphNodes));
  if (numHIPGraphNodes == 0) {
      TORCH_WARN("The HIP Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  // check if debug path is set
  if (!_hip_graphs_debug) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    C10_ZOOM_CHECK(hipGraphDestroy(graph_));
    has_graph_ = false;
  } else {
    TORCH_WARN("DEBUG: TORCH_HIPGRAPHS_DEBUG_PATH detected. graph_ will not be freed until debug_dump is called.");
  }
}

void HIPGraph::replay() {
  TORCH_CHECK(has_graph_exec_,
              "Called HIPGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  // graph_exec_ may be replayed in any stream.
  C10_ZOOM_CHECK(hipGraphLaunch(graph_exec_, c10::zoom::getCurrentZoomStream()));

// cuda does this sync for certain versions, we're ignoring it here
//   int version;
//   C10_ZOOM_CHECK(cudaDriverGetVersion(&version));
//   if (version < 11040) {
//     // Workaround for bug in libcuda.so that causes replayed graphs with
//     // certain topologies to be corrupted (kernels elided, internal syncs
//     // ignored) when replayed back to back without a sync in between.
//     // The bug is fixed in CUDA 11.4+.
//     C10_ZOOM_CHECK(cudaDeviceSynchronize());
//   }
}

void HIPGraph::enable_debug_mode() {
  _hip_graphs_debug = true;
}

void HIPGraph::debug_dump(const std::string& debug_path) {
  if (_hip_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling hipGraphDebugDotPrint() with ", debug_path);
      C10_ZOOM_CHECK_WARN(hipGraphDebugDotPrint(graph_, debug_path.c_str(), 1<<10)); // most verbose output
      C10_ZOOM_CHECK(hipGraphDestroy(graph_));
    }
  } else {
    // TODO (Arham): technically false right now, need to add this functionality to the Zoom PyBind module
    TORCH_WARN("HIP Graphs debug not enabled, set with torch._C._zoom_enable_graphs_debug_mode");
  }

}

void HIPGraph::reset() {
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this HIPGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::zoom::ZoomCachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_ZOOM_CHECK_WARN(hipGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_ZOOM_CHECK_WARN(hipGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t HIPGraph::pool() {
TORCH_CHECK(has_graph_exec_,
              "Called HIPGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

HIPGraph::~HIPGraph() {
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->unregister_graph(this);
  }
  reset();
}

} // namespace at::zoom