#include <c10/util/Exception.h>

#include <chrono>
#include <exception>
#include <ratio>
#include <tuple>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <ATen/SparseCsrTensorUtils.h>
#include <c10/util/StringUtil.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <sys/types.h>
#include <type_traits>

#include "ProcessGroupSCCL.hpp"
#include "TPUDeviceManager.h"
#include "TPUAddrHelper.h"
#include "sccl.h"

namespace c10d {

static const std::string SCCL_SOCKET_IFNAME_ENV = "SCCL_SOCKET_IFNAME";
constexpr int kBytes = 8;
std::vector<std::string> split0(char separator, const std::string &string) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (std::getline(ss, item, separator)) {
    pieces.push_back(std::move(item));
  }
  return pieces;
}

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;

std::chrono::milliseconds
getRemainingTime(steady_clock_time_point startTime,
                 const std::chrono::milliseconds &timeout, bool waitAllRanks) {
  if (waitAllRanks) {
    // See Note in monitoredBarrier
    return timeout;
  }
  auto elapsedTime = std::chrono::steady_clock::now() - startTime;
  auto remainingMillis =
      timeout -
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime);

  // If no more remaining time, return -1 to indicate to caller.
  if (remainingMillis.count() <= 0) {
    return std::chrono::milliseconds(-1);
  }

  return remainingMillis;
}

void logAndThrow(const std::string &logMessage,
                 const std::string &errorMessage) {
  LOG(ERROR) << logMessage;
  TORCH_CHECK(false, errorMessage);
}

void checkRemainingTime(
    const std::chrono::milliseconds &monitoredBarrierTimeout,
    const std::chrono::milliseconds &remainingTime,
    const std::vector<int> &processedRanks, int currentRank) {
  const std::string kNoRemainingTimeError =
      c10::str("Rank ", currentRank, " timed out in monitoredBarrier after ",
               monitoredBarrierTimeout.count(), " ms.");
  if (remainingTime.count() < 0) {
    std::string rankInfo;
    if (processedRanks.size() > 0) {
      rankInfo = c10::str("Successfully processed ranks: ",
                          c10::Join(", ", processedRanks));
    } else {
      rankInfo = "No ranks successfully processed in monitoredBarrier.";
    }
    auto error = c10::str(kNoRemainingTimeError, "\n", rankInfo);
    logAndThrow(error, error);
  }
}

scclDataType_t toSCCLDtype(::at::ScalarType type) {
  scclDataType_t dtype = SCCL_DTYPE_FP32;
  switch (type) {
  case ::at::ScalarType::Float:
    dtype = SCCL_DTYPE_FP32;
    break;
  case ::at::ScalarType::Half:
    dtype = SCCL_DTYPE_FP16;
    break;
  case ::at::ScalarType::BFloat16:
    dtype = SCCL_DTYPE_BF16;
    break;
  case ::at::ScalarType::Char:
    dtype = SCCL_DTYPE_INT8;
    break;
  case ::at::ScalarType::Byte:
    dtype = SCCL_DTYPE_UINT8;
    break;
  case ::at::ScalarType::Int:
    dtype = SCCL_DTYPE_INT32;
    break;
  case ::at::ScalarType::Long:
    dtype = SCCL_DTYPE_INT64;
    break;
  case ::at::ScalarType::Bool:
    dtype = SCCL_DTYPE_BOOL;
    break;
  default:
    TORCH_CHECK(false, "Invalid scalar type");
  }
  return dtype;
}

typedef void (*ReduceFunc)(void *, const void *, const void *, size_t);


void ProcessGroupSCCL::insertUsedDeviceIdx(int idx) {
  usedDeviceIdxs_.insert(idx);
}

void ProcessGroupSCCL::collectiveCounter(){
  seqCollective_++;
}

template <typename F>
c10::intrusive_ptr<ProcessGroupSCCL::WorkSCCL> collective(
    ProcessGroupSCCL& group,
    at::Tensor &input, at::Tensor &output, F func) {
  // Bump collective counter
  group.collectiveCounter();

  auto device = input.device();
  group.insertUsedDeviceIdx((int)device.index());
  c10_tpu::TPUStream stream = c10_tpu::getCurrentTPUStream();

  scclComm_t comm;
  group.getSCCLcomm(&comm);
  auto ret = func(input, output, comm, stream);

  TORCH_CHECK(scclCommDestroy(comm) == scclSuccess,
              "sccl comm destroy rank failed\n");
  TORCH_CHECK(ret == scclSuccess);

  return c10::make_intrusive<ProcessGroupSCCL::WorkSCCL>(output);
}

std::vector<at::Tensor> ProcessGroupSCCL::WorkSCCL::result() {
  TORCH_CHECK(isCompleted(),
              "Work needs to be completed before calling result(). "
              "Should call wait() before result().");
  return std::vector<at::Tensor> {outputTensor_};
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupSCCL::WorkSCCL::getFuture() {
  return future_;
}

c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
    const at::Tensor &outputTensor) {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(std::vector<at::Tensor> {outputTensor}));
  return future;
}

ProcessGroupSCCL::WorkSCCL::WorkSCCL(at::Tensor outputTensor)
  : outputTensor_(std::move(outputTensor)),
    future_(createFutureAsOutput(outputTensor_)) {
}

bool ProcessGroupSCCL::WorkSCCL::isCompleted() {
  // to do
  return true;
}

bool ProcessGroupSCCL::WorkSCCL::wait(std::chrono::milliseconds timeout) {
  // to do
  return true;
}

ProcessGroupSCCL::Options::Options(std::chrono::milliseconds timeout)
    : ProcessGroup::Options(SCCL_BACKEND_NAME, timeout) {}

void ProcessGroupSCCL::broadcastUniqueSCCLID(scclHandle_t handle,
                                              scclUniqueId *scclID,
                                              int rank) {
  const char* GROUP_IDX_env = getenv("TPU_GROUP_IDX");
  std::string GROUP_IDX =  GROUP_IDX_env ? GROUP_IDX_env : "";
  const std::string key = "ProcessGroupSCCL" + GROUP_IDX;
  memset(scclID, 0x0, sizeof(scclUniqueId));
  if (rank == 0) {
    TORCH_CHECK(scclGetUniqueId(handle, scclID) ==
                    scclSuccess,
                "sccl get unique ID failed\n");
    auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t *>(scclID),
                                 reinterpret_cast<uint8_t *>(scclID) +
                                 SCCL_UNIQUE_ID_BYTES);
    store_->set(key, vec);
  } else {
    auto vec = store_->get(key);
    TORCH_CHECK(vec.size() == SCCL_UNIQUE_ID_BYTES,
                "Invalid size for scclUniqueID\n");
    memcpy(scclID, vec.data(), sizeof(vec.size()));
  }
}

ProcessGroupSCCL::ProcessGroupSCCL(const c10::intrusive_ptr<Store> &store,
                                       int rank, int size,
                                       c10::intrusive_ptr<Options> options)
    : ProcessGroup(rank, size), store_(store),
      options_(options), stop_(false) {
  c10_tpu::TPUStream stream = c10_tpu::getCurrentTPUStream();
  if (rank == 0) {
    scclSetupC2CTopology(stream);
  }
  broadcastUniqueSCCLID(stream, &scclID_, rank);
  init();
}

ProcessGroupSCCL::~ProcessGroupSCCL() {
}

void ProcessGroupSCCL::getSCCLcomm(scclComm_t *comm){
  int ranks, rank;
  ranks = getSize();
  rank = getRank();

  c10::intrusive_ptr<Options> options = getOptions();

  TORCH_CHECK(scclCommInitRank(
              comm, ranks, scclID_, rank,
              options->chip_map.data()) == scclSuccess,
          "sccl comm init rank failed\n");
}

static inline at::Tensor &checkSingleTensor(std::vector<at::Tensor> &tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(false, "ProcessGroupSCCL::send takes a single tensor");
  }
  auto &tensor = tensors[0];
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }
  return tensor;
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::send(std::vector<at::Tensor> &inputs,
                              int dstRank,
                              int /* unused */) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::send: " + msg);
  };
  assertRootRank(invalidArgument, dstRank, size_);
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  auto& tensor = checkSingleTensor(inputs);
  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  return collective(*this, tensor, tensor,
    [&](at::Tensor &input, at::Tensor &output,
        scclComm_t comm, scclHandle_t handle) {
      void *sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      return scclSend(
          sendBuff,
          input.numel(), toSCCLDtype(input.scalar_type()), dstRank,
          comm, handle);
    });
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::recv(std::vector<at::Tensor> &inputs,
                              int srcRank,
                              int /* unused */) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::recv: " + msg);
  };
  assertRootRank(invalidArgument, srcRank, size_);
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  auto& tensor = checkSingleTensor(inputs);
  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  return collective(*this, tensor, tensor,
    [&](at::Tensor &input, at::Tensor &output,
        scclComm_t comm, scclHandle_t handle) {
      void *recvBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      return scclRecv(
          recvBuff,
          input.numel(), toSCCLDtype(input.scalar_type()), srcRank,
          comm, handle);
    });
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::broadcast(std::vector<at::Tensor> &inputs,
                              const BroadcastOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::broadcast: " + msg);
  };
  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(invalidArgument, opts.rootTensor, inputs.size());
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  return collective(*this, inputs[opts.rootTensor], inputs[opts.rootTensor],
    [&](at::Tensor &input, at::Tensor &output,
        scclComm_t comm, scclHandle_t handle) {
      void *sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      return scclBroadcast(
          sendBuff,
          input.numel(), toSCCLDtype(input.scalar_type()), opts.rootRank,
          comm, handle);
    });
}

inline scclReduceType_t toSCCLReduceOp(const ReduceOp reduceOp){
  scclReduceType_t reduceMethod = SCCL_REDUCE_SUM;

    switch (reduceOp) {
    case ReduceOp::SUM:
      reduceMethod = SCCL_REDUCE_SUM;
      break;
    case ReduceOp::PRODUCT:
      reduceMethod = SCCL_REDUCE_PROD;
      break;
    case ReduceOp::MIN:
      reduceMethod = SCCL_REDUCE_MIN;
      break;
    case ReduceOp::MAX:
      reduceMethod = SCCL_REDUCE_MAX;
      break;
    case ReduceOp::BAND:
      TORCH_CHECK(false, "Cannot use ReduceOp.BAND with SCCL");
      break;
    case ReduceOp::BOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BOR with SCCL");
      break;
    case ReduceOp::BXOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with SCCL");
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with SCCL");
      break;
    case ReduceOp::AVG:
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with SCCL");
      break;
    case ReduceOp::UNUSED:
      break;
    }
  return reduceMethod;
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::allreduce(std::vector<at::Tensor> &inputs,
                              const AllreduceOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::allreduce: " + msg);
  };
  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto &layout = inputs[0].layout();
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  if (layout != c10::kStrided) {
    invalidArgument("unsupported layout");
  }

  return collective(
    *this, inputs[0], inputs[0],
    [&](at::Tensor &input, at::Tensor &output, scclComm_t comm,
        scclHandle_t handle) {
      scclReduceType_t reduceMethod = toSCCLReduceOp(opts.reduceOp);
      const void * sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      void *recvBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)output.data_ptr()));
      return scclAllReduce(
          sendBuff,
          recvBuff,
          input.numel(), toSCCLDtype(input.scalar_type()), reduceMethod,
          comm, handle);
    });
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::reduce(std::vector<at::Tensor> &inputs,
                           const ReduceOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::reduce: " + msg);
  };

  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto &device = inputs[0].device();
  const auto &layout = inputs[0].layout();
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(reduce of sparse tensors only works with ReduceOp.SUM)");
  }

  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  at::Tensor flatOutputTensor;
  if (getRank() != opts.rootRank) {
    flatOutputTensor = newLikeFlat(inputs);
  }

  return collective(
    *this, inputs[0],
    getRank() == opts.rootRank ? inputs[0] : flatOutputTensor,
    [&](at::Tensor &input, at::Tensor &output, scclComm_t comm,
        scclHandle_t handle) {
      scclReduceType_t reduceMethod = toSCCLReduceOp(opts.reduceOp);
      const void * sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      void *recvBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)output.data_ptr()));
      return scclReduce(
          sendBuff,
          recvBuff,
          input.numel(), toSCCLDtype(input.scalar_type()), reduceMethod,
          opts.rootRank, comm, handle);
    });
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::allgather(std::vector<std::vector<at::Tensor>> &outputs,
                              std::vector<at::Tensor> &inputs,
                              const AllgatherOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::allgather: " + msg);
  };

  if (inputs.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }

  if (inputs.size() != outputs.size()) {
    invalidArgument(
        "requires input/output tensor lists to have the same length");
  }

  for (const auto i : c10::irange(outputs.size())) {
    const auto expected = inputs.size() * getSize();
    const auto actual = outputs[i].size();
    if (actual != expected) {
      invalidArgument("invalid output tensor list at index " +
                      std::to_string(i) + " (expected length " +
                      std::to_string(expected) + ", got " +
                      std::to_string(actual) + ")");
    }
  }

  const auto &options = inputs[0].options();
  const auto &sizes = inputs[0].sizes();
  assertTypeAndSizesMatch(invalidArgument, inputs, options, sizes);
  for (const auto &output : outputs) {
    assertTypeAndSizesMatch(invalidArgument, output, options, sizes);
  }

  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  bool continous_addr = true;
  for (size_t i = 0; i < outputs[0].size() - 1; ++i) {
    auto addr = GetAddrByUnifiedAddr((uint64_t)outputs[0][i].data_ptr());
    auto next_addr = GetAddrByUnifiedAddr((uint64_t)outputs[0][i + 1].data_ptr());
    uint64_t bytes = outputs[0][i].numel() * outputs[0][i].element_size();
    if (addr + bytes != next_addr) {
      continous_addr = false;
      break;
    }
  }
  at::Tensor flatOutputTensor = continous_addr == false ? newLikeFlat(outputs[0]) : outputs[0][0];

  auto work = collective(
    *this, inputs[0], flatOutputTensor,
    [](at::Tensor &input, at::Tensor &output, scclComm_t comm,
        scclHandle_t handle) {
      const void * sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      void *recvBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)output.data_ptr()));
      return scclAllGather(
          sendBuff,
          recvBuff,
          input.numel(), toSCCLDtype(input.scalar_type()), comm, handle);
    });

  // Unflatten into output tensors.
  if (continous_addr == false) {
    for (auto &outputGroup : outputs) {
      for (const auto j : c10::irange(outputGroup.size())) {
        outputGroup[j].copy_(flatOutputTensor[j]);
      }
    }
  }

  return work;
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::_allgather_base(at::Tensor &outputTensor,
                                    at::Tensor &inputTensor,
                                    const AllgatherOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::_allgather_base: " + msg);
  };

  TORCH_CHECK(
      outputTensor.device() == inputTensor.device(),
      "output tensor and input tensor must be on the same type of device");

  const auto &device = outputTensor.device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  auto work = collective(
    *this, inputTensor, outputTensor,
    [](at::Tensor &input, at::Tensor &output, scclComm_t comm,
        scclHandle_t handle) {
      const void * sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      void *recvBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)output.data_ptr()));
      return scclAllGather(
          sendBuff,
          recvBuff,
          input.numel(), toSCCLDtype(input.scalar_type()), comm, handle);
    });

  return work;
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::gather(std::vector<std::vector<at::Tensor>> &outputs,
                           std::vector<at::Tensor> &inputs,
                           const GatherOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementInput(invalidArgument, inputs);
  // assertDense(invalidArgument, inputs);

  if (getRank() == opts.rootRank) {
    if (outputs.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputs[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputs[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto &options = inputs[0].options();
    const auto &sizes = inputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, outputs[0], options, sizes);
  } else {
    if (!outputs.empty()) {
      invalidArgument("requires empty output on non-root");
    }
  }

  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  at::Tensor flatOutputTensor;
  if (getRank() == opts.rootRank) {
    flatOutputTensor = newLikeFlat(outputs[0]);
  } else {
    flatOutputTensor = newLikeFlat(inputs);
  }

  auto work = collective(
      *this, inputs[0], flatOutputTensor,
      [&](at::Tensor &input, at::Tensor &output, scclComm_t comm,
          scclHandle_t handle) {
        const void * sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
        void *recvBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)output.data_ptr()));
        return scclGather(
            sendBuff,
            recvBuff,
            input.numel(), toSCCLDtype(input.scalar_type()), opts.rootRank, comm,
            handle);
      });

  // Unflatten into output tensors on root process.
  if (getRank() == opts.rootRank) {
    for (const auto i : c10::irange(outputs[0].size())) {
      outputs[0][i].copy_(flatOutputTensor[i]);
    }
  }
  return work;
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::scatter(std::vector<at::Tensor> &outputs,
    std::vector<std::vector<at::Tensor>> &inputs,
    const ScatterOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementOutput(invalidArgument, outputs);
  // assertDense(invalidArgument, outputs);

  if (getRank() == opts.rootRank) {
    if (inputs.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors";
      invalidArgument(ss.str());
    } else if (inputs[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputs[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }
    const auto &options = outputs[0].options();
    const auto &sizes = outputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, inputs[0], options, sizes);
  } else {
    if (inputs.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
  }

  const auto &device = outputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  at::Tensor flatInputTensor;
  if (getRank() == opts.rootRank) {
    flatInputTensor = newLikeFlat(inputs[0]);
    for (const auto i : c10::irange(inputs[0].size())) {
      flatInputTensor[i].copy_(inputs[0][i]);
    }
  } else {
    flatInputTensor = newLikeFlat(outputs);
  }

  return collective(
    *this, flatInputTensor, outputs[0],
    [&](at::Tensor &input, at::Tensor &output, scclComm_t comm,
        scclHandle_t handle) {
      const void * sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      void *recvBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)output.data_ptr()));
      return scclScatter(
          sendBuff,
          recvBuff,
          output.numel(), toSCCLDtype(output.scalar_type()), opts.rootRank, comm,
          handle);
    });
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::reduce_scatter(std::vector<at::Tensor> &outputs,
    std::vector<std::vector<at::Tensor>> &inputs,
    const ReduceScatterOptions &opts) {
  TORCH_CHECK(false, "ProcessGroupSCCL does not support reduce_scatter");
}

c10::intrusive_ptr<Work> ProcessGroupSCCL::alltoall_base(
    at::Tensor &outputTensor, at::Tensor &inputTensor,
    std::vector<int64_t> &outputCounts, std::vector<int64_t> &inputCounts,
    const AllToAllOptions & /* unused */) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSCCL::alltoall_base: " + msg);
  };

  TORCH_CHECK(
      outputTensor.device() == inputTensor.device(),
      "output tensor and input tensor must be on the same type of device");

  const auto &device = outputTensor.device();

  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  return collective(
    *this, inputTensor, outputTensor,
    [](at::Tensor &input, at::Tensor &output, scclComm_t comm,
        scclHandle_t handle) {
      const void * sendBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)input.data_ptr()));
      void *recvBuff = scclPhysToVirt(handle, GetAddrByUnifiedAddr((uint64_t)output.data_ptr()));
      return scclAllToAll(
          sendBuff,
          recvBuff,
          output.numel(), toSCCLDtype(output.scalar_type()), comm, handle);
    });
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::barrier(const BarrierOptions &opts) {
  std::vector<at::Device> devices;
  for (auto it = usedDeviceIdxs_.cbegin(); it != usedDeviceIdxs_.cend(); ++it){
    std::cout << "usedDeviceIdxs_:"<< *it << std::endl;
  }

  // Use user defined TPU device ids if provided
  if (!opts.device_ids.empty()) {
    for (auto device : opts.device_ids) {
      devices.emplace_back(at::kPrivateUse1, device);
    }
  } else if (usedDeviceIdxs_.empty()) {
    int device = tpu::TPUGetDeviceIndex();
    devices.emplace_back(at::kPrivateUse1, device);
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.emplace_back(at::kPrivateUse1, usedDeviceIdx);
    }
  }

  // Use one device only
  auto device = devices.back();
  if (device.type() != at::kPrivateUse1) {
    TORCH_CHECK(false, "unsupported device type", device.type(), "\n");
  }

  std::vector<at::Tensor> barrierTensor = {
      at::empty({1}, at::TensorOptions().device(device).dtype(at::ScalarType::Float)),
      at::empty({1}, at::TensorOptions().device(device).dtype(at::ScalarType::Float))
      };
  auto work = allreduce(barrierTensor);

  // Work will take over barrierTensors
  auto scclWork = dynamic_cast<ProcessGroupSCCL::WorkSCCL*>(work.get());
  scclWork->barrierTensor_ = std::move(barrierTensor);
  return work;
}

void ProcessGroupSCCL::setSequenceNumberForGroup() {
} // SCCL just starts sequence numbers at 0.

uint64_t ProcessGroupSCCL::getSequenceNumberForGroup() {
  // to do
  return 0;
}

c10::intrusive_ptr<c10d::ProcessGroup>
ProcessGroupSCCL::createProcessGroupSCCL(
    const c10d::DistributedBackendOptions &dis_opts, Options &options) {
  if (!options.chip_map.empty()) {
    TORCH_CHECK((int)options.chip_map.size() >= dis_opts.group_size,
                "chip map size must be same with the nranks\n");
  }

  options.timeout =
      std::chrono::duration_cast<std::chrono::milliseconds>(dis_opts.timeout);
  //  NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return c10::make_intrusive<c10d::ProcessGroupSCCL>(
      dis_opts.store, dis_opts.group_rank, dis_opts.group_size,
      c10::make_intrusive<Options>(options));
}

__attribute__((constructor)) void ProcessGroupSCCLConstructor() {
  py::object module = py::module::import("torch.distributed");
  py::object register_backend =
      module.attr("Backend").attr("register_backend");
  register_backend("SCCL", py::cpp_function(ProcessGroupSCCL::createProcessGroupSCCL), true, "tpu");
}

} // namespace c10d
