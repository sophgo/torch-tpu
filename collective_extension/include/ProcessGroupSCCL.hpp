#pragma once

#include <c10/util/hash.h>
#include <pybind11/chrono.h>
#include <sophon/algorithm.h>
#include <sophon/common/error.h>
#include <sophon/context.h>
#include <sophon/rendezvous/store.h>
#include <sophon/transport/device.h>
#include <torch/python.h>
#include "tpuDNN.h"
#include "TPUStream.h"

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

  class TORCH_API SophonStore : public ::sophon::rendezvous::Store {
  public:
    SophonStore(const c10::intrusive_ptr<::c10d::Store> &store)
        : store_(store) {}

    void setUint(const std::string &key, const std::vector<uint8_t> &value) {
      store_->set(key, value);
    }

    void set(const std::string &key, const std::vector<char> &value) override {
      std::vector<uint8_t> tmp(value.begin(), value.end());
      store_->set(key, tmp);
    }

    std::vector<uint8_t> getUint(const std::string &key) {
      auto value = store_->get(key);
      return value;
    }

    std::vector<char> get(const std::string &key) override {
      auto value = store_->get(key);
      return std::vector<char>(value.begin(), value.end());
    }

    void wait(const std::vector<std::string> &keys) override {
      store_->wait(keys, Store::kDefaultTimeout);
    }

    void wait(const std::vector<std::string> &keys,
              const std::chrono::milliseconds &timeout) override {
      store_->wait(keys, timeout);
    }

  protected:
    c10::intrusive_ptr<::c10d::Store> store_;
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

  static std::shared_ptr<::sophon::transport::Device>
  createDeviceForInterface(const std::string &interface);

  static std::shared_ptr<::sophon::transport::Device>
  createDeviceForHostname(const std::string &hostname);

  static std::shared_ptr<::sophon::transport::Device> createDefaultDevice();

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

  c10::intrusive_ptr<Work> recvAnysource(std::vector<at::Tensor> &tensors,
                                         int tag) override;

  c10::intrusive_ptr<Work>barrier(
    const BarrierOptions &opts = BarrierOptions()) override;

  const std::unique_ptr<::sophon::rendezvous::Store> &_getStore() const {
    return store_;
  }

  void setSequenceNumberForGroup() override;

  uint64_t getSequenceNumberForGroup() override;

  void broadcastUniqueSCCLID(sophon::scclUniqueId *scclID, int rank);

  void insertUsedDeviceIdx(int idx);

protected:
  std::unique_ptr<::sophon::rendezvous::Store> store_;
  const c10::intrusive_ptr<Options> options_;

  std::vector<std::shared_ptr<::sophon::Context>> contexts_;
  bool stop_;

  uint32_t collectiveCounter_;

  uint32_t nextTag();

  std::shared_ptr<::sophon::Context> getContext(uint32_t tag);

  std::deque<c10::intrusive_ptr<WorkSCCL>> workQueue_;
  std::vector<c10::intrusive_ptr<WorkSCCL>> workInProgress_;
  std::mutex workMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  tpudnnHandle_t dev_handle_;

public:
  static c10::intrusive_ptr<c10d::ProcessGroup>
  createProcessGroupSCCL(const c10d::DistributedBackendOptions &dis_opts,
                         Options &options);

  static void ProcessGroupSCCLConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("SCCL", py::cpp_function(createProcessGroupSCCL), true);
  }
};

} // namespace c10d