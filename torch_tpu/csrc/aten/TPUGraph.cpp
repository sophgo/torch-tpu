#include "TPUGraph.h"
#include "TPUException.h"
#include "TPUFunction.h"
#include "tpu_runtime_api.h"

#include <cstddef>

namespace at_tpu {

#ifdef BACKEND_SG2260E
auto& StreamBeginCapture = tpuStreamBeginCapture;
auto& StreamEndCapture = tpuStreamEndCapture;
auto& GraphDestroy = tpuGraphDestroy;
auto& GraphInstantiate = tpuGraphInstantiate;
auto& GraphExecDestroy = tpuGraphExecDestroy;
auto& GraphLaunch = tpuGraphLaunch;
#else
auto& StreamBeginCapture = tpudnnStreamBeginCapture;
auto& StreamEndCapture = tpudnnStreamEndCapture;
auto& GraphDestroy = tpudnnGraphDestroy;
auto& GraphInstantiate = tpudnnGraphInstantiate;
auto& GraphExecDestroy = tpudnnGraphExecDestroy;
auto& GraphLaunch = tpudnnGraphLaunch;
#endif

c10_tpu::MempoolId_t graph_pool_handle() {
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // tpuStreamGetCaptureInfo id_s in capture_begin.
  return c10_tpu::MemPool::graph_pool_handle();
}

/**
 * Note [TPU Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Note [Interaction with TPU graph capture] in TPUCachingAllocator.cpp
 * describes memory management for captures.
 */

TPUGraph::TPUGraph(bool keep_graph)
  // TPUStreams may not be default-constructed.
  : capture_stream_(c10_tpu::getCurrentTPUStream()), keep_graph_(keep_graph) {
}

void TPUGraph::capture_begin(c10_tpu::MempoolId_t pool/*=0*/) {
  TORCH_CHECK(!has_graph_exec_,
              "This TPUGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  capture_dev_ = c10_tpu::current_device();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Create graph pool handle using is_user_created=false.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = c10_tpu::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // Addendum: beginAllocateStreamToPool is now called before tpuStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid tpuEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10_tpu::TPUCachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](tpuStream_t stream) {
      // tpuStreamCaptureStatus status{};
      // CaptureId_t stream_capture_id = 0;
      // AT_TPU_CHECK(tpuStreamGetCaptureInfo(stream, &status, &stream_capture_id));
      // return status == tpuStreamCaptureStatus::tpuStreamCaptureStatusActive && stream_capture_id == capture_id_;
      return true;
  });

  AT_TPU_CHECK(StreamBeginCapture(capture_stream_));

  // tpuStreamCaptureStatus status{};
  // AT_TPU_CHECK(tpuStreamGetCaptureInfo(stream, &status, &capture_id_));
  // TORCH_INTERNAL_ASSERT(status == tpuStreamCaptureStatus::tpuStreamCaptureStatusActive);

}

void TPUGraph::capture_end() {
  // auto stream = c10_tpu::getCurrentTPUStream().stream();

  // TORCH_CHECK(stream == capture_stream_,
  //             "Capture must end on the same stream it began on.");

  AT_TPU_CHECK(StreamEndCapture(capture_stream_, &graph_));

  c10_tpu::TPUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  // TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

  // size_t numTPUGraphNodes = 0;
  // AT_TPU_CHECK(tpuGraphGetNodes(graph_, nullptr, &numTPUGraphNodes));
  // if (numTPUGraphNodes == 0) {
  //     TORCH_WARN("The TPU Graph is empty. This usually means that the graph was ",
  //                "attempted to be captured on wrong device or stream.");
  // }

  capture_ended_ = true;
  has_graph_ = true;
  if (!keep_graph_) {
    instantiate();
    AT_TPU_CHECK(GraphDestroy(graph_));
    has_graph_ = false;
  }
}

void TPUGraph::instantiate() {
  TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    TORCH_CHECK(keep_graph_, "instantiate() is intended to be called by the user only when keep_graph=true");
    AT_TPU_CHECK(GraphExecDestroy(graph_exec_));
  }
  AT_TPU_CHECK(GraphInstantiate(&graph_exec_, graph_, nullptr));
  has_graph_exec_ = true;
}

void TPUGraph::replay() {
  TORCH_CHECK(capture_ended_,
              "Called TPUGraph::replay without a preceding successful capture.");

  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  // graph_exec_ may be replayed in any stream.
  AT_TPU_CHECK(GraphLaunch(graph_exec_, capture_stream_));

  // TODO: need sync ?
  // AT_TPU_CHECK(tpuDeviceSynchronize());
}

Graph_t TPUGraph::raw_tpu_graph() {
  TORCH_CHECK(keep_graph_, "You cannot access the raw tpuGraph_t instance unless TPUGraph was initialized with keep_graph=true");
  TORCH_CHECK(has_graph_, "You cannot access the raw tpuGraph_t instance until capture_end() has been called");
  return graph_;
}

void TPUGraph::reset() {
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
  // If capture_begin, the capture, or capture_end failed at some point, this TPUGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (capture_ended_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10_tpu::TPUCachingAllocator::releasePool(capture_dev_, mempool_id_);
    capture_ended_ = false;
  }
  if (has_graph_exec_) {
    AT_TPU_CHECK(GraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
  if (has_graph_) {
    AT_TPU_CHECK(GraphDestroy(graph_));
    has_graph_ = false;
  }

}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
c10_tpu::MempoolId_t TPUGraph::pool() {
TORCH_CHECK(capture_ended_,
              "Called TPUGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

TPUGraph::~TPUGraph() {
  reset();
  // TODO: need sync?
  if (capture_dev_ != UNDEFINED_DEVICE) // check if capture_dev_ contains the real device id
  {
    AT_TPU_CHECK(tpuSetDevice(capture_dev_));
    // AT_TPU_CHECK(tpuStreamSynchronize(capture_stream_));  // TODO: device sync or stream sync ?
  }
}

} // namespace at_tpu
