#pragma once

#include <c10/util/hash.h>
#include <pybind11/chrono.h>
#include <sophon/algorithm.h>
#include <sophon/common/error.h>
#include <sophon/transport/device.h>
#include <torch/python.h>
#include "tpuDNN.h"
#include "TPUStream.h"
#include "sccl.h"

#include <condition_variable>
#include <deque>
#include <mutex>

#include <c10/util/hash.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <unordered_map>
#include <vector>

namespace c10d {

constexpr const char *SCCL_BACKEND_NAME = "SCCL";

class TORCH_API ProcessGroupSCCL : public ProcessGroup {
public:
  class TORCH_API WorkSCCL final: public Work {
  public:
    explicit WorkSCCL(at::Tensor outputTensor);

    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    bool isCompleted() override;

  protected:
    friend class ProcessGroupSCCL;

  private:
    const at::Tensor outputTensor_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    // Tensors used for barrier op
    std::vector<at::Tensor> barrierTensor_;
  };

  struct TORCH_API Options : public ProcessGroup::Options {
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options>
    create(std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }

    std::vector<std::shared_ptr<::sophon::transport::Device>> devices;
    std::vector<int> chip_map;
  };

  const std::string getBackendName() const override {
    return std::string(SCCL_BACKEND_NAME);
  }

  explicit ProcessGroupSCCL(
      const c10::intrusive_ptr<Store> &store, int rank, int size,
      c10::intrusive_ptr<Options> options = Options::create());

  ~ProcessGroupSCCL() override;

  c10::intrusive_ptr<Options> getOptions() { return options_; }

  c10::intrusive_ptr<Work>
  broadcast(std::vector<at::Tensor> &tensors,
            const BroadcastOptions &opts = BroadcastOptions()) override;
  c10::intrusive_ptr<Work>
  allreduce(std::vector<at::Tensor> &tensors,
            const AllreduceOptions &opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work>
  reduce(std::vector<at::Tensor> &tensors,
         const ReduceOptions &opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work>
  allgather(std::vector<std::vector<at::Tensor>> &outputs,
            std::vector<at::Tensor> &inputs,
            const AllgatherOptions &opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work>
  _allgather_base(at::Tensor &outputBuffer, at::Tensor &inputBuffer,
                  const AllgatherOptions &opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work>
  gather(std::vector<std::vector<at::Tensor>> &outputs,
         std::vector<at::Tensor> &inputs,
         const GatherOptions &opts = GatherOptions()) override;

  c10::intrusive_ptr<Work>
  scatter(std::vector<at::Tensor> &outputs,
          std::vector<std::vector<at::Tensor>> &inputs,
          const ScatterOptions &opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor> &outputs,
      std::vector<std::vector<at::Tensor>> &inputs,
      const ReduceScatterOptions &opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work>
  alltoall_base(at::Tensor &outputTensor, at::Tensor &inputTensor,
                std::vector<int64_t> &outputCounts,
                std::vector<int64_t> &inputCounts,
                const AllToAllOptions &opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dstRank,
                                int tag) override;

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int srcRank,
                                int tag) override;

  c10::intrusive_ptr<Work>barrier(
    const BarrierOptions &opts = BarrierOptions()) override;

  const c10::intrusive_ptr<Store> &_getStore() const {
    return store_;
  }

  void setSequenceNumberForGroup() override;

  uint64_t getSequenceNumberForGroup() override;

  void broadcastUniqueSCCLID(scclUniqueId *scclID, int rank);

  void getSCCLcomm(scclComm_t *comm);

  void insertUsedDeviceIdx(int idx);

  void collectiveCounter();

protected:
  c10::intrusive_ptr<Store> store_;

  const c10::intrusive_ptr<Options> options_;

  scclUniqueId scclID_;

  // Counting for the sequential number of SCCL collective call.
  uint64_t seqCollective_{0};

  bool stop_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  tpudnnHandle_t dev_handle_;

public:
  static c10::intrusive_ptr<c10d::ProcessGroup>
  createProcessGroupSCCL(const c10d::DistributedBackendOptions &dis_opts,
                         Options &options);
};

} // namespace c10d