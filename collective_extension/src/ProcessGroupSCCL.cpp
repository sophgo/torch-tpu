#include <c10/util/Exception.h>

#include <chrono>
#include <exception>
#include <ratio>
#include <tuple>

#ifdef _WIN32
#include <sophon/common/win.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <ATen/SparseCsrTensorUtils.h>
#include <c10/util/StringUtil.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <sophon/barrier.h>
#include <sophon/config.h>
#include <sophon/rendezvous/context.h>
#include <sophon/rendezvous/prefix_store.h>
#include <sys/types.h>

#include <type_traits>

#include "ProcessGroupSCCL.hpp"
#include "SCCLDeviceFactory.hpp"
#include "TPUAddrHelper.h"
#include "sccl.h"
#include "tpuv7_rt.h"

#ifdef _WIN32
#define GENERATE_ALL_TYPES(type, func, ...)                                    \
  switch (type) {                                                              \
  case ::at::ScalarType::Float:                                                \
    func<float>(__VA_ARGS__);                                                  \
    break;                                                                     \
  case ::at::ScalarType::Double:                                               \
    func<double>(__VA_ARGS__);                                                 \
    break;                                                                     \
  case ::at::ScalarType::Half:                                                 \
    func<sophon::float16>(__VA_ARGS__);                                        \
    break;                                                                     \
  case ::at::ScalarType::Char:                                                 \
    func<int8_t>(__VA_ARGS__);                                                 \
    break;                                                                     \
  case ::at::ScalarType::Byte:                                                 \
    func<uint8_t>(__VA_ARGS__);                                                \
    break;                                                                     \
  case ::at::ScalarType::Int:                                                  \
    func<int32_t>(__VA_ARGS__);                                                \
    break;                                                                     \
  case ::at::ScalarType::Long:                                                 \
    func<int64_t>(__VA_ARGS__);                                                \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false, "Invalid scalar type");                                 \
  }

#define HOST_NAME_MAX 256
#else
#define GENERATE_ALL_TYPES(type, func, args...)                                \
  switch (type) {                                                              \
  case ::at::ScalarType::Float:                                                \
    func<float>(args);                                                         \
    break;                                                                     \
  case ::at::ScalarType::Double:                                               \
    func<double>(args);                                                        \
    break;                                                                     \
  case ::at::ScalarType::Half:                                                 \
    func<sophon::float16>(args);                                               \
    break;                                                                     \
  case ::at::ScalarType::Char:                                                 \
    func<int8_t>(args);                                                        \
    break;                                                                     \
  case ::at::ScalarType::Byte:                                                 \
    func<uint8_t>(args);                                                       \
    break;                                                                     \
  case ::at::ScalarType::Int:                                                  \
    func<int32_t>(args);                                                       \
    break;                                                                     \
  case ::at::ScalarType::Long:                                                 \
    func<int64_t>(args);                                                       \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false, "Invalid scalar type");                                 \
  }
#endif

namespace c10d {

namespace {

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

tpudnnDataType_t toTpudnnDtype(::at::ScalarType type) {
  tpudnnDataType_t dtype = TPUDNN_DTYPE_FP32;
  switch (type) {
  case ::at::ScalarType::Float:
    dtype = TPUDNN_DTYPE_FP32;
    break;
  case ::at::ScalarType::Half:
    dtype = TPUDNN_DTYPE_FP16;
    break;
  case ::at::ScalarType::Char:
    dtype = TPUDNN_DTYPE_INT8;
    break;
  case ::at::ScalarType::Byte:
    dtype = TPUDNN_DTYPE_UINT8;
    break;
  case ::at::ScalarType::Int:
    dtype = TPUDNN_DTYPE_INT32;
    break;
  default:
    TORCH_CHECK(false, "Invalid scalar type");
  }
  return dtype;
}

typedef void (*ReduceFunc)(void *, const void *, const void *, size_t);

template <typename F>
static c10::intrusive_ptr<ProcessGroupSCCL::WorkSCCL> collective(
    const std::shared_ptr<sophon::Context> &context,
    at::Tensor &input, at::Tensor &output, F func) {
  sophon::scclComm_t comm;
  TORCH_CHECK(sophon::scclCommInitRank(
                  &comm, context->size, context->scclID, context->rank,
                  context->chip_map.data()) == sophon::scclSuccess,
              "sccl comm init rank failed\n");
  auto ret = func(input, output, comm, context->handle);
  TORCH_CHECK(sophon::scclCommDestroy(comm) == sophon::scclSuccess,
              "sccl comm destroy rank failed\n");
  TORCH_CHECK(ret == sophon::scclSuccess);

  return c10::make_intrusive<ProcessGroupSCCL::WorkSCCL>(output);
}

template <typename T, typename O>
void setInputs(O &opts, std::vector<at::Tensor> &tensors) {
  opts.setInputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setInput(O &opts, at::Tensor &tensor) {
  opts.setInput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setInput(O &opts, at::Tensor &tensor, std::vector<size_t> &counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setInput(O &opts, at::Tensor &tensor, std::vector<int64_t> &counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutputs(O &opts, std::vector<at::Tensor> &tensors) {
  opts.setOutputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setOutput(O &opts, at::Tensor &tensor) {
  opts.setOutput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setOutput(O &opts, at::Tensor &tensor, std::vector<size_t> &counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutput(O &opts, at::Tensor &tensor, std::vector<int64_t> &counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

const auto kLoopbackAddress = "127.0.0.1";
} // namespace

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

void socketInitialize() {
#ifdef _WIN32
  ::sophon::init_winsock();
#endif
}

bool doesHostnameResolveToUsableAddress(const std::string &hostname) {
  socketInitialize();
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *result;
  auto rv = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
  if (rv < 0) {
    return false;
  }
  struct addrinfo *rp;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }
    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
    if (rv == -1) {
      continue;
    }
    break;
  }
  freeaddrinfo(result);
  return rp != nullptr;
}

std::shared_ptr<::sophon::transport::Device>
ProcessGroupSCCL::createDeviceForInterface(
    const std::string &interface_name) {
  return ::c10d::SCCLDeviceFactory::makeDeviceForInterface(interface_name);
}

std::shared_ptr<::sophon::transport::Device>
ProcessGroupSCCL::createDeviceForHostname(const std::string &hostname) {
  TORCH_CHECK(doesHostnameResolveToUsableAddress(hostname), "Cannot resolve ",
              hostname, " to a (local) address");
  return ::c10d::SCCLDeviceFactory::makeDeviceForHostname(hostname);
}

#if defined(__linux__) || defined(_WIN32)
std::shared_ptr<::sophon::transport::Device>
ProcessGroupSCCL::createDefaultDevice() {
  // Use the hostname to resolve the network address to
  // use. Note: if the hostname does not resolve to an address (e.g.
  // because of misconfigured /etc/hosts file), this will not work.
  socketInitialize();
  std::array<char, HOST_NAME_MAX> hostname{};
  auto rv = gethostname(hostname.data(), HOST_NAME_MAX);
  if (rv != 0) {
    // throw std::system_error(errno, std::system_category());
  }

  // Use this machine's hostname if it resolves to an address.
  if (doesHostnameResolveToUsableAddress(hostname.data())) {
    return ::c10d::SCCLDeviceFactory::makeDeviceForHostname(hostname.data());
  }

  // Otherwise, use the loopback address.
  TORCH_WARN_ONCE("Unable to resolve hostname to a (local) address. ",
                  "Using the loopback address as fallback. ",
                  "Manually set the network interface to bind to with "
                  "SCCL_SOCKET_IFNAME.");
  return createDeviceForHostname(kLoopbackAddress);
}
#endif

void ProcessGroupSCCL::broadcastUniqueSCCLID(sophon::scclUniqueId *scclID,
                                               int rank) {
  const std::string key = "ProcessGroupSCCL";
  memset(scclID, 0x0, sizeof(sophon::scclUniqueId));
  if (rank == 0) {
    TORCH_CHECK(tpuRtGetUniqueId(reinterpret_cast<char *>(scclID)) ==
                    tpuRtSuccess,
                "sccl get unique ID failed\n");
    auto vec = std::vector<char>(reinterpret_cast<char *>(scclID),
                                 reinterpret_cast<char *>(scclID) +
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
    : ProcessGroup(rank, size), store_(new SophonStore(store)),
      options_(options), stop_(false), collectiveCounter_(0) {
  sophon::scclUniqueId scclID;
  broadcastUniqueSCCLID(&scclID, rank);

  c10_tpu::TPUStream stream = c10_tpu::getCurrentTPUStream();
  auto &devices = options->devices;
  if (devices.empty()) {
    TORCH_CHECK(false, "No device(s) specified");
  }
  contexts_.reserve(options->devices.size());
  for (const auto i : c10::irange(options->devices.size())) {
    auto context =
        std::make_shared<::sophon::rendezvous::Context>(rank_, size_);
    auto store = ::sophon::rendezvous::PrefixStore(std::to_string(i), *store_);
    context->setTimeout(options->timeout);
    context->chip_map = options->chip_map;
    context->handle = stream;
    memcpy(&context->scclID, &scclID, sizeof(scclID));
    try {
      context->connectFullMesh(store, options->devices[i]);
    } catch (const std::runtime_error &e) {
      auto err = e.what();
      // TORCH_CHECK to print the cpp stacktrace.
      auto msg = c10::str("SCCL connectFullMesh failed with ", err);
      logAndThrow(msg, msg);
    }
    contexts_.push_back(std::move(context));
  }

  init();
}

ProcessGroupSCCL::~ProcessGroupSCCL() {
  tpudnnDestroy(dev_handle_);
}

uint32_t ProcessGroupSCCL::nextTag() { return collectiveCounter_++; }

std::shared_ptr<::sophon::Context>
ProcessGroupSCCL::getContext(uint32_t tag) {
  return contexts_[tag % contexts_.size()];
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

  auto tag = nextTag();
  auto context = getContext(tag);

  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  return collective(context, inputs[opts.rootTensor], inputs[opts.rootTensor],
    [&](at::Tensor &input, at::Tensor &output,
        sophon::scclComm_t comm, tpudnnHandle_t handle) {
      return sophon::scclBroadcast(
          (void *)GetAddrByUnifiedAddr((uint64_t)input.data_ptr()),
          input.numel(), toTpudnnDtype(input.scalar_type()), opts.rootRank,
          comm, handle);
    });
}

// allreduce

inline tpudnnReduceType_t reduceMethod(const ReduceOp reduceOp){
  tpudnnReduceType_t reduce_method = TPUDNN_REDUCE_SUM;

    switch (reduceOp) {
    case ReduceOp::SUM:
      reduce_method = TPUDNN_REDUCE_SUM;
      break;
    case ReduceOp::PRODUCT:
      reduce_method = TPUDNN_REDUCE_PROD;
      break;
    case ReduceOp::MIN:
      reduce_method = TPUDNN_REDUCE_MIN;
      break;
    case ReduceOp::MAX:
      reduce_method = TPUDNN_REDUCE_MAX;
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
  return reduce_method;
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

  auto tag = nextTag();
  auto context = getContext(tag);

  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  if (layout != c10::kStrided) {
    invalidArgument("unsupported layout");
  }

  tpudnnReduceType_t reduce_method = reduceMethod(opts.reduceOp);

  return collective(
    context, inputs[0], inputs[0],
    [&](at::Tensor &input, at::Tensor &output, sophon::scclComm_t comm,
        tpudnnHandle_t handle) {
      return sophon::scclAllReduce(
          (const void *)GetAddrByUnifiedAddr((uint64_t)input.data_ptr()),
          (void *)GetAddrByUnifiedAddr((uint64_t)output.data_ptr()),
          input.numel(), toTpudnnDtype(input.scalar_type()), reduce_method,
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
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  at::Tensor flatOutputTensor;
  if (context->rank != opts.rootRank) {
    flatOutputTensor = newLikeFlat(inputs);
  }

  tpudnnReduceType_t reduce_method = reduceMethod(opts.reduceOp);
  
  return collective(
    context, inputs[0],
    context->rank == opts.rootRank ? inputs[0] : flatOutputTensor,
    [&](at::Tensor &input, at::Tensor &output, sophon::scclComm_t comm,
        tpudnnHandle_t handle) {
      return sophon::scclReduce(
          (const void *)GetAddrByUnifiedAddr((uint64_t)input.data_ptr()),
          (void *)GetAddrByUnifiedAddr((uint64_t)output.data_ptr()),
          input.numel(), toTpudnnDtype(input.scalar_type()), reduce_method,
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

  auto tag = nextTag();
  auto context = getContext(tag);

  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);

  auto work = collective(
    context, inputs[0], flatOutputTensor,
    [](at::Tensor &input, at::Tensor &output, sophon::scclComm_t comm,
        tpudnnHandle_t handle) {
      return sophon::scclAllGather(
          (const void *)GetAddrByUnifiedAddr((uint64_t)input.data_ptr()),
          (void *)GetAddrByUnifiedAddr((uint64_t)output.data_ptr()),
          input.numel(), toTpudnnDtype(input.scalar_type()), comm, handle);
    });
  
  // Unflatten into output tensors.
  for (auto &outputgroup : outputs) {
    for (const auto j : c10::irange(outputgroup.size())) {
      outputgroup[j].copy_(flatOutputTensor[j]);
    }
  }

  return work;
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::_allgather_base(at::Tensor & /*unused */,
                                    at::Tensor & /*unused */,
                                    const AllgatherOptions & /*unused */) {
  TORCH_CHECK(false, "no support for _allgather_base in SCCL process group");
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

  auto tag = nextTag();
  auto context = getContext(tag);

  const auto &device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  at::Tensor flatOutputTensor;
  if (context->rank == opts.rootRank) {
    flatOutputTensor = newLikeFlat(outputs[0]);
  } else {
    flatOutputTensor = newLikeFlat(inputs);
  }

  auto work = collective(
      context, inputs[0], flatOutputTensor,
      [&](at::Tensor &input, at::Tensor &output, sophon::scclComm_t comm,
          tpudnnHandle_t handle) {
        return sophon::scclGather(
            (const void *)GetAddrByUnifiedAddr((uint64_t)input.data_ptr()),
            (void *)GetAddrByUnifiedAddr((uint64_t)output.data_ptr()),
            input.numel(), toTpudnnDtype(input.scalar_type()), opts.rootRank, comm,
            handle);
      });

  // Unflatten into output tensors on root process.
  if (context->rank == opts.rootRank) {
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

  auto tag = nextTag();
  auto context = getContext(tag);

  const auto &device = outputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  at::Tensor flatInputTensor;
  if (context->rank == opts.rootRank) {
    flatInputTensor = newLikeFlat(inputs[0]);
    for (const auto i : c10::irange(inputs[0].size())) {
      flatInputTensor[i].copy_(inputs[0][i]);
    }
  } else {
    flatInputTensor = newLikeFlat(outputs);
  }

  return collective(
    context, flatInputTensor, outputs[0],
    [&](at::Tensor &input, at::Tensor &output, sophon::scclComm_t comm,
        tpudnnHandle_t handle) {
      return sophon::scclScatter(
          (const void *)GetAddrByUnifiedAddr((uint64_t)input.data_ptr()),
          (void *)GetAddrByUnifiedAddr((uint64_t)output.data_ptr()),
          output.numel(), toTpudnnDtype(output.scalar_type()), opts.rootRank, comm,
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
  auto tag = nextTag();
  auto context = getContext(tag);

  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  // invalidArgument("Not support alltoall_base when !(outputCounts.empty() && inputCounts.empty())");

  return collective(
    context, inputTensor, outputTensor,
    [](at::Tensor &input, at::Tensor &output, sophon::scclComm_t comm,
        tpudnnHandle_t handle) {
      return sophon::scclAllToAll(
          (const void *)GetAddrByUnifiedAddr((uint64_t)input.data_ptr()),
          (void *)GetAddrByUnifiedAddr((uint64_t)output.data_ptr()),
          output.numel(), toTpudnnDtype(output.scalar_type()), comm, handle);
    });
}

at::Tensor &checkSingleTensor(std::vector<at::Tensor> &tensors) {
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

uint32_t checkTag(int32_t tag) {
  TORCH_CHECK(tag >= 0, "Tag must be nonnegative");
  return (uint32_t)tag;
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::send(std::vector<at::Tensor> &tensors, int dstRank,
                         int tag) {
  auto &tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.data_ptr();
  auto size = tensor.numel() * tensor.element_size();

  // Construct unbound buffer.
  auto context = getContext(tag);
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->send(dstRank, utag);

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the send.
  TORCH_CHECK(false, "send not implement");
  return c10::make_intrusive<Work>();
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::recv(std::vector<at::Tensor> &tensors, int srcRank,
                         int tag) {
  auto &tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.data_ptr();
  auto size = tensor.numel() * tensor.element_size();

  // Construct unbound buffer.
  auto context = getContext(tag);
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->recv(srcRank, utag);

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the recv.
  TORCH_CHECK(false, "recv not implement");
  return c10::make_intrusive<Work>();
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::recvAnysource(std::vector<at::Tensor> &tensors, int tag) {
  auto &tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto ptr = tensor.data_ptr();
  auto size = tensor.numel() * tensor.element_size();

  // Construct unbound buffer.
  auto context = getContext(tag);
  auto buf = context->createUnboundBuffer(ptr, size);

  // Build list of ranks that this operation can recv from. In these
  // bindings we don't differentiate between ranks and can receive
  // from any other process in the group.
  std::vector<int> srcRanks;
  srcRanks.resize(size_);
  for (const auto i : c10::irange(size_)) {
    srcRanks.push_back(i);
  }

  buf->recv(srcRanks, utag);

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the recv.
  TORCH_CHECK(false, "recvAnysource not implement");
  return c10::make_intrusive<Work>();
}

c10::intrusive_ptr<Work>
ProcessGroupSCCL::barrier(const BarrierOptions &opts) {
  std::vector<c10::weak_intrusive_ptr<WorkSCCL>> priorWork;
  TORCH_CHECK(false, "barrier not implement");
  return c10::make_intrusive<Work>();
}

void ProcessGroupSCCL::setSequenceNumberForGroup() {
  if (rank_ == 0) {
    // Create and broadcast sequence number
    auto seq = 1 + rand();
    sequenceNum_ = c10d::SequenceNum(seq);
    std::vector<char> values = c10d::toVec<char>(seq, kBytes);
    store_->set(kSeqNumStoreKey, values);
  } else {
    // Read rank 0's sequence number from store.
    sequenceNum_ = c10d::SequenceNum();
    store_->wait({kSeqNumStoreKey}, options_->timeout);
    std::vector<char> values = store_->get(kSeqNumStoreKey);
    uint64_t num = c10d::fromVec<char>(values);
    sequenceNum_->set(num);
  }
}

uint64_t ProcessGroupSCCL::getSequenceNumberForGroup() {
  if (sequenceNum_ == c10::nullopt) {
    return 0;
  }
  return sequenceNum_->get();
}

c10::intrusive_ptr<c10d::ProcessGroup>
ProcessGroupSCCL::createProcessGroupSCCL(
    const c10d::DistributedBackendOptions &dis_opts, Options &options) {
  if (!options.chip_map.empty()) {
    TORCH_CHECK((int)options.chip_map.size() >= dis_opts.group_size,
                "chip map size must be same with the nranks\n");
  }

  // Use interfaces listed in "SCCL_SOCKET_IFNAME", if set.
  char *ifnameEnv = getenv(SCCL_SOCKET_IFNAME_ENV.c_str());
  if (ifnameEnv && strlen(ifnameEnv) > 1) {
    for (const auto &iface : split0(',', ifnameEnv)) {
      options.devices.push_back(createDeviceForInterface(iface));
    }
  } else {
    // If no hostname is specified, this function looks up
    // the machine's hostname and returns a device instance
    // associated with the address that the hostname resolves to.
    options.devices.push_back(createDefaultDevice());
  }

  options.timeout =
      std::chrono::duration_cast<std::chrono::milliseconds>(dis_opts.timeout);
  //  NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return c10::make_intrusive<c10d::ProcessGroupSCCL>(
      dis_opts.store, dis_opts.group_rank, dis_opts.group_size,
      c10::make_intrusive<Options>(options));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupSCCL", &ProcessGroupSCCL::createProcessGroupSCCL);

  py::class_<ProcessGroupSCCL::Options>(m, "ProcessGroupSCCLOptions")
      .def(py::init<>())
      .def_readwrite("chip_map", &ProcessGroupSCCL::Options::chip_map);
}

} // namespace c10d