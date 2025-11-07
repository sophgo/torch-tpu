#pragma once

#include "torch_tpu/csrc/core/TPUCachingAllocator.h"

namespace at_tpu {


// Standalone way to get a unique mempool id usable as a pool=... argument
// to TPUGraph::capture_begin
c10_tpu::MempoolId_t graph_pool_handle();

#ifdef BACKEND_SG2260E
using Graph_t = tpuGraph_t;
using GraphExec_t = tpuGraphExec_t;
#else
using Graph_t = tpudnnGraph_t;
using GraphExec_t = tpudnnGraphExec_t;
#endif

struct TPUGraph {
  TPUGraph(bool keep_graph=false);
  ~TPUGraph();

  void capture_begin(c10_tpu::MempoolId_t pool = {0, 0});
  void capture_end();
  void instantiate();
  void replay();
  void reset();
  c10_tpu::MempoolId_t pool();
  Graph_t raw_tpu_graph();

 protected:
  Graph_t graph_ = nullptr;
  GraphExec_t graph_exec_ = nullptr;

  // internal states so reset() can do its best cleaning up

  // Set to true in capture_end if tpuStreamEndCapture succeeded
  // Set back to false after instantiate() unless keep_graph=True
  // was called on any TPUGraph instance.
  bool has_graph_ = false;
  // Set to true in capture_end if tpuStreamEndCapture succeeded
  bool capture_ended_ = false;
  // Set to true in capture_end if tpuGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // the ID assigned by tpu during graph capture,
  // used to identify when a stream is participating in capture
  c10_tpu::CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from TPUCachingAllocator.
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
  c10_tpu::MempoolId_t mempool_id_;

  // Stream on which capture began
  TPUStream capture_stream_;

  // Device where capture occurred. Right now, for simplicity, we require all ops
  // in a capture to run on the same device, but this is a limitation of TPUGraph,
  // not TPU itself.  We can straightforwardly modify TPUGraph to support multi-device
  // captures if needed.
  // init capture_dev_ as UNDEFINED_DEVICE to check that it stores the real device id in the destructor
  static constexpr c10::DeviceIndex UNDEFINED_DEVICE = -1;
  c10::DeviceIndex capture_dev_{UNDEFINED_DEVICE};

  bool keep_graph_;
};

} // namespace at_tpu
