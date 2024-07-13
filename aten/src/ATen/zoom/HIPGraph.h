#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/zoom/HIPGraphsC10Utils.h>
#include <c10/zoom/ZoomStream.h>
#include <c10/util/flat_hash_map.h>

namespace at {

struct Generator;
struct ZoomGeneratorImpl;
struct ZoomGeneratorState;

using MempoolId_t = c10::zoom::MempoolId_t;
using CaptureId_t = c10::zoom::CaptureId_t;

namespace zoom {

// Standalone way to get a unique mempool id usable as a pool=... argument
// to HIPGraph::capture_begin
MempoolId_t graph_pool_handle();

struct HIPGraph {
  HIPGraph();
  ~HIPGraph();

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  // See Note [Explicit Registration of Generators to the CUDA Graph]
  void register_generator_state(c10::intrusive_ptr<at::ZoomGeneratorState> state);
  void register_generator_state(const at::Generator& generator);
  void capture_begin(
      MempoolId_t pool = {0, 0},
      hipStreamCaptureMode capture_mode = hipStreamCaptureModeGlobal);
  void capture_end();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);

 protected:
  hipGraph_t graph_ = NULL;
  hipGraphExec_t graph_exec_ = NULL;

  static std::atomic<int> pending_event_queries;

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if hipStreamEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by hipGraphInstantiate
  // to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if hipGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, used to
  // specify the pool.
  CaptureId_t id_;

  // the ID assigned by hip during graph capture,
  // used to identify when a stream is participating in capture
  CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from CUDACachingAllocator.
  // By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's mempool_id_
  // will be set to the other graph's mempool_id_, and therefore share a mempool with the
  // other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from graph_pool_handle(),
  // it will share a mempool with any other captures that used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // Stream on which capture began
  c10::zoom::ZoomStream capture_stream_;

  // multiple generator states and their wholegraph_increments in this graph
  // that are managed by the CUDA Graph
  ska::flat_hash_map<c10::intrusive_ptr<at::ZoomGeneratorState>, uint64_t>
      captured_generator_states_;

  // Device where capture occurred. Right now, for simplicity, we require all ops
  // in a capture to run on the same device, but this is a limitation of HIPGraph,
  // not CUDA itself.  We can straightforwardly modify HIPGraph to support multi-device
  // captures if needed.
  int capture_dev_;
};

} // namespace cuda
} // namespace at