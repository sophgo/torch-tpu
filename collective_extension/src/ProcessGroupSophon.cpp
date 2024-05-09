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
#include <ATen/SparseTensorUtils.h>
#include <ATen/ThreadLocalState.h>
#include <TPUDeviceManager.h>
#include <c10/util/StringUtil.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <sophon/allgather.h>
#include <sophon/allgatherv.h>
#include <sophon/allreduce.h>
#include <sophon/alltoall.h>
#include <sophon/alltoallv.h>
#include <sophon/barrier.h>
#include <sophon/broadcast.h>
#include <sophon/config.h>
#include <sophon/gather.h>
#include <sophon/reduce.h>
#include <sophon/rendezvous/context.h>
#include <sophon/rendezvous/prefix_store.h>
#include <sophon/scatter.h>
#include <sys/types.h>
#include <thread>

#include <type_traits>

// #include "../include/ProcessGroupSophon.hpp"
#include "ProcessGroupSophon.hpp"
#include "SophonDeviceFactory.hpp"
#include "bmlib_runtime.h"
#include "common_def.h" // for sg_data_type_t defines

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

static const std::string SOPHON_SOCKET_IFNAME_ENV = "SOPHON_SOCKET_IFNAME";
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

// 要改？
typedef void (*ReduceFunc)(void *, const void *, const void *, size_t);

template <typename T,
          typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
ReduceFunc toFunction(const ReduceOp &r) {
  switch (r) {
  case ReduceOp::SUM:
    return ReduceFunc(&::sophon::sum<T>);
  case ReduceOp::PRODUCT:
    return ReduceFunc(&::sophon::product<T>);
  case ReduceOp::MIN:
    return ReduceFunc(&::sophon::min<T>);
  case ReduceOp::MAX:
    return ReduceFunc(&::sophon::max<T>);
  case ReduceOp::BAND:
    TORCH_CHECK(false, "Cannot use ReduceOp.BAND with non-integral dtype");
    break;
  case ReduceOp::BOR:
    TORCH_CHECK(false, "Cannot use ReduceOp.BOR with non-integral dtype");
    break;
  case ReduceOp::BXOR:
    TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with non-integral dtype");
    break;
  case ReduceOp::AVG:
    TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Sophon");
    break;
  case ReduceOp::PREMUL_SUM:
    TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Sophon");
    break;
  case ReduceOp::UNUSED:
    break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

// Bitwise AND with SFINAE guard for integral types.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void band(void *c, const void *a, const void *b, size_t n) {
  auto tc = static_cast<T *>(c);
  auto ta = static_cast<const T *>(a);
  auto tb = static_cast<const T *>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] & tb[i];
  }
}

// Bitwise OR with SFINAE guard for integral types.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void bor(void *c, const void *a, const void *b, size_t n) {
  auto tc = static_cast<T *>(c);
  auto ta = static_cast<const T *>(a);
  auto tb = static_cast<const T *>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] | tb[i];
  }
}

// Bitwise XOR with SFINAE guard for integral types.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void bxor(void *c, const void *a, const void *b, size_t n) {
  auto tc = static_cast<T *>(c);
  auto ta = static_cast<const T *>(a);
  auto tb = static_cast<const T *>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] ^ tb[i];
  }
}

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
ReduceFunc toFunction(const ReduceOp &r) {
  switch (r) {
  case ReduceOp::SUM:
    return ReduceFunc(&::sophon::sum<T>);
  case ReduceOp::PRODUCT:
    return ReduceFunc(&::sophon::product<T>);
  case ReduceOp::MIN:
    return ReduceFunc(&::sophon::min<T>);
  case ReduceOp::MAX:
    return ReduceFunc(&::sophon::max<T>);
  case ReduceOp::BAND:
    return ReduceFunc(&band<T>);
  case ReduceOp::BOR:
    return ReduceFunc(&bor<T>);
  case ReduceOp::BXOR:
    return ReduceFunc(&bxor<T>);
  case ReduceOp::AVG:
    TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Sophon");
    break;
  case ReduceOp::PREMUL_SUM:
    TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Sophon");
    break;
  case ReduceOp::UNUSED:
    break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
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

void ProcessGroupSophon::AsyncWork::execute(
    c10::intrusive_ptr<AsyncWork> work) {
  if (work->recordFunctionBeforeCallback_) {
    work->recordFunctionBeforeCallback_();
  }
  try {
    work->run();
  } catch (...) {
    work->finishWorkSophonError(std::current_exception());
    return;
  }

  work->synchronize();
  work->finishWorkSophon();
}

std::vector<at::Tensor> ProcessGroupSophon::AsyncWork::result() {
  TORCH_CHECK(isCompleted(),
              "Work needs to be completed before calling result(). "
              "Should call wait() before result().");
  TORCH_CHECK(outputTensors_.size() <= 1,
              "work result does not support list of lists, use .getFuture() "
              "and value()");
  return outputTensors_.size() == 0 ? std::vector<at::Tensor>()
                                    : outputTensors_.at(0);
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupSophon::AsyncWork::getFuture() {
  return future_;
}

namespace {
c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
    const std::vector<std::vector<at::Tensor>> &outputTensors) {
  if (outputTensors.size() > 1) {
    return c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  }
  return c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

void returnFutureWithOutput(
    c10::intrusive_ptr<c10::ivalue::Future> &future,
    const std::vector<std::vector<at::Tensor>> &outputTensors) {
  if (outputTensors.size() == 0) {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
    return;
  }
  if (outputTensors.size() > 1) {
    future->markCompleted(c10::IValue(outputTensors));
    return;
  }
  future->markCompleted(c10::IValue(outputTensors[0]));
}
} // namespace

inline void ProcessGroupSophon::AsyncWork::recordAsyncWorkProfilingInfo(
    const char *profilingTitle,
    const c10::optional<std::vector<at::Tensor>> &inputTensors) {
  auto recordingFunction =
      std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
  if (recordingFunction->isActive()) {
    std::function<void()> before_handler = [inputTensors, profilingTitle,
                                            recordingFunction]() {
      // The work will be started and completed by different threads.
      recordingFunction->_setAsync();
      std::vector<c10::IValue> inputs;
      if (inputTensors) {
        inputs.reserve(inputTensors->size());
        for (const auto &tensor : *inputTensors) {
          inputs.emplace_back(tensor);
        }
      }
      recordingFunction->before(
          profilingTitle,
          c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
    };
    recordFunctionBeforeCallback_ = at::wrapPropagateTLSState(before_handler);
    std::function<void()> end_handler = [recordingFunction]() {
      recordingFunction->end();
    };
    recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
  }
}

ProcessGroupSophon::AsyncWork::AsyncWork(
    std::vector<std::vector<at::Tensor>> outputTensors,
    const char *profilingTitle,
    const c10::optional<std::vector<at::Tensor>> &inputTensors)
    : Work(-1, OpType::UNKNOWN, nullptr, inputTensors),
      outputTensors_(std::move(outputTensors)),
      future_(createFutureAsOutput(outputTensors_)) {
  if (profilingTitle != nullptr) {
    recordAsyncWorkProfilingInfo(profilingTitle, inputTensors);
  }
}

void ProcessGroupSophon::AsyncWork::finishWorkSophonError(
    std::exception_ptr eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupSophon::AsyncWork::finishWorkSophon() {
  returnFutureWithOutput(future_, outputTensors_);
  finish();
}

ProcessGroupSophon::SendWork::SendWork(
    at::Tensor &tensor,
    std::unique_ptr<::sophon::transport::UnboundBuffer> buffer)
    : Work(-1, OpType::SEND, "sophon:send",
           c10::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor), buffer_(std::move(buffer)) {}

bool ProcessGroupSophon::SendWork::wait(std::chrono::milliseconds timeout) {
  bool sendCompleted = false;
  std::exception_ptr exception{nullptr};
  try {
    if (timeout == kNoTimeout) {
      sendCompleted = buffer_->waitSend();
    } else {
      sendCompleted = buffer_->waitSend(timeout);
    }
  } catch (...) {
    exception = std::current_exception();
  }

  // Completes the Work object and throws the exception.
  finishAndThrow(exception);
  return sendCompleted;
}

void ProcessGroupSophon::SendWork::abort() { buffer_->abortWaitSend(); }

ProcessGroupSophon::RecvWork::RecvWork(
    at::Tensor &tensor,
    std::unique_ptr<::sophon::transport::UnboundBuffer> buffer,
    const char *profilingTitle)
    : Work(-1, OpType::UNKNOWN, profilingTitle,
           c10::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor), buffer_(std::move(buffer)), srcRank_(-1) {}

int ProcessGroupSophon::RecvWork::sourceRank() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return srcRank_;
}

bool ProcessGroupSophon::RecvWork::wait(std::chrono::milliseconds timeout) {
  bool recvCompleted = false;
  std::exception_ptr exception{nullptr};
  try {
    if (timeout == kNoTimeout) {
      recvCompleted = buffer_->waitRecv(&srcRank_);
    } else {
      recvCompleted = buffer_->waitRecv(&srcRank_, timeout);
    }
  } catch (...) {
    exception = std::current_exception();
  }

  // Completes the Work object and throws the exception.
  finishAndThrow(exception);
  return recvCompleted;
}

void ProcessGroupSophon::RecvWork::abort() { buffer_->abortWaitRecv(); }

ProcessGroupSophon::Options::Options(std::chrono::milliseconds timeout)
    : ProcessGroup::Options(SOPHON_BACKEND_NAME, timeout), threads(2) {}

namespace {
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
} // namespace

std::shared_ptr<::sophon::transport::Device>
ProcessGroupSophon::createDeviceForInterface(
    const std::string &interface_name) {
  return ::c10d::SophonDeviceFactory::makeDeviceForInterface(interface_name);
}

std::shared_ptr<::sophon::transport::Device>
ProcessGroupSophon::createDeviceForHostname(const std::string &hostname) {
  TORCH_CHECK(doesHostnameResolveToUsableAddress(hostname), "Cannot resolve ",
              hostname, " to a (local) address");
  return ::c10d::SophonDeviceFactory::makeDeviceForHostname(hostname);
}

#if defined(__linux__) || defined(_WIN32)
std::shared_ptr<::sophon::transport::Device>
ProcessGroupSophon::createDefaultDevice() {
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
    return ::c10d::SophonDeviceFactory::makeDeviceForHostname(hostname.data());
  }

  // Otherwise, use the loopback address.
  TORCH_WARN_ONCE("Unable to resolve hostname to a (local) address. ",
                  "Using the loopback address as fallback. ",
                  "Manually set the network interface to bind to with "
                  "SOPHON_SOCKET_IFNAME.");
  return createDeviceForHostname(kLoopbackAddress);
}
#endif

ProcessGroupSophon::ProcessGroupSophon(const c10::intrusive_ptr<Store> &store,
                                       int rank, int size,
                                       c10::intrusive_ptr<Options> options)
    : ProcessGroup(rank, size), store_(new SophonStore(store)),
      options_(options), stop_(false), collectiveCounter_(0) {
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
    try {
      context->connectFullMesh(store, options->devices[i]);
    } catch (const std::runtime_error &e) {
      auto err = e.what();
      // TORCH_CHECK to print the cpp stacktrace.
      auto msg = c10::str("sophon connectFullMesh failed with ", err);
      logAndThrow(msg, msg);
    }
    contexts_.push_back(std::move(context));
  }
  workInProgress_.resize(options->threads);

  threads_.resize(options->threads);
  for (const auto i : c10::irange(threads_.size())) {
    threads_[i] = std::thread(&ProcessGroupSophon::runLoop, this, i);
  }

  init();
}

ProcessGroupSophon::~ProcessGroupSophon() {
  std::unique_lock<std::mutex> lock(workMutex_);
  workConsumeCV_.wait(lock, [&] { return workQueue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();

  workProduceCV_.notify_all();

  // Wait for worker threads to terminate
  for (auto &thread : threads_) {
    thread.join();
  }
}

uint32_t ProcessGroupSophon::nextTag() { return collectiveCounter_++; }

std::shared_ptr<::sophon::Context>
ProcessGroupSophon::getContext(uint32_t tag) {
  return contexts_[tag % contexts_.size()];
}

void ProcessGroupSophon::runLoop(int workerIndex) {
  std::unique_lock<std::mutex> lock(workMutex_);

  while (!stop_) {
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }

    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    workInProgress_[workerIndex] = work;
    lock.unlock();

    // Notify after releasing the lock so that the waiter
    // does not immediately block.
    workConsumeCV_.notify_one();

    AsyncWork::execute(std::move(work));
    lock.lock();
    workInProgress_[workerIndex].reset();
  }
}

void ProcessGroupSophon::enqueue(c10::intrusive_ptr<AsyncWork> work) {
  std::unique_lock<std::mutex> lock(workMutex_);
  // Bump collective counter
  if (sequenceNum_) {
    sequenceNum_->increment();
  }
  workQueue_.push_back(std::move(work));
  lock.unlock();

  // Notify after releasing the lock so that the waiter
  // does not immediately block.
  workProduceCV_.notify_one();
}

// broadcast
namespace {
class AsyncBroadcastWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncBroadcastWork(const std::shared_ptr<sophon::Context> &context,
                     std::vector<at::Tensor> &inputs, int rootRank,
                     int rootTensor, uint32_t tag)
      : ProcessGroupSophon::AsyncWork({inputs}, "sophon:broadcast", inputs),
        context(context), inputs(inputs), rootRank(rootRank),
        rootTensor(rootTensor), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;

  void broadcast(at::Tensor &tensor) {
    const auto &scalarType = tensor.scalar_type();
    sophon::BroadcastOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    sophon::broadcast(opts);
  }

  void run() override {
    broadcast(inputs[rootTensor]);

    // Copy to non-root tensors
    for (const auto i : c10::irange(inputs.size())) {
      if (i == static_cast<size_t>(rootTensor)) {
        continue;
      }
      inputs[i].copy_(inputs[rootTensor]);
    }
  }
  void finishWorkSophon() override {
    // TODO
  }
};

class AsyncBroadcastTPUWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncBroadcastTPUWork(const std::shared_ptr<sophon::Context> &context,
                        std::vector<at::Tensor> &inputs, int rootRank,
                        int rootTensor, uint32_t tag)
      : ProcessGroupSophon::AsyncWork({inputs}, "sophon:broadcast", inputs),
        context(context), inputs(inputs), rootRank(rootRank),
        rootTensor(rootTensor), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;

  void broadcast(at::Tensor &tensor) {
    const auto &scalarType = tensor.scalar_type();
    sophon::BroadcastOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);

    bm_handle_t handle_ = tpu::TPUGetDeviceResource();
    size_t bytes_ = tensor.nbytes();
    bm_device_mem_t buff_dev =
        bm_mem_from_device((unsigned long long)tensor.data_ptr(), bytes_);

    // for dev mem
    switch (scalarType) {
    case ::at::ScalarType::Float:
      opts.setOutputSophon(SG_DTYPE_FP32, handle_, bytes_, buff_dev);
      break;
    case ::at::ScalarType::Half:
      opts.setOutputSophon(SG_DTYPE_FP16, handle_, bytes_, buff_dev);
      break;
    case ::at::ScalarType::Char:
      opts.setOutputSophon(SG_DTYPE_INT8, handle_, bytes_, buff_dev);
      break;
    case ::at::ScalarType::Byte:
      opts.setOutputSophon(SG_DTYPE_UINT8, handle_, bytes_, buff_dev);
      break;
    case ::at::ScalarType::Int:
      opts.setOutputSophon(SG_DTYPE_INT32, handle_, bytes_, buff_dev);
      break;
    default:
      TORCH_CHECK(false, "Invalid scalar type");
    }

    sophon::broadcast2260(opts); // 2260通信接口
  }

  void run() override {
    broadcast(inputs[rootTensor]);

    // Copy to non-root tensors
    for (const auto i : c10::irange(inputs.size())) {
      if (i == static_cast<size_t>(rootTensor)) {
        continue;
      }
      inputs[i].copy_(inputs[rootTensor]);
    }
  }
  void finishWorkSophon() override { finish(); }
};
} // namespace

c10::intrusive_ptr<Work>
ProcessGroupSophon::broadcast(std::vector<at::Tensor> &inputs,
                              const BroadcastOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSophon::broadcast: " + msg);
  };
  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(invalidArgument, opts.rootTensor, inputs.size());
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto &device = inputs[0].device();
  switch (device.type()) {
  case at::kCPU:
    break;
  case at::kPrivateUse1:
    break;
  default:
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncBroadcastWork>(
        std::move(context), inputs, opts.rootRank, opts.rootTensor, tag);
  } else if (device.type() == at::kPrivateUse1) {
    work = c10::make_intrusive<AsyncBroadcastTPUWork>(
        std::move(context), inputs, opts.rootRank, opts.rootTensor, tag);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }

  enqueue(work);
  return work;
}

// allreduce
namespace {

class AsyncAllreduceWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncAllreduceWork(const std::shared_ptr<sophon::Context> &context,
                     std::vector<at::Tensor> &tensors, ReduceOp reduceOp,
                     uint32_t tag)
      : ProcessGroupSophon::AsyncWork({inputs}, "sophon:all_reduce", inputs),
        context(context), inputs(inputs), reduceOp(reduceOp), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<at::Tensor> inputs;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void allreduce(std::vector<at::Tensor> &tensors) {
    const auto &scalarType = tensors[0].scalar_type();
    sophon::AllreduceOptions opts(context);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutputs, opts, tensors);
    sophon::allreduce(opts);
  }

  void run() override { allreduce(inputs); }

  void finishWorkSophon() override {
    // TODO
  }

  template <typename T>
  void getFunction(sophon::AllreduceOptions::Func &fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  sophon::AllreduceOptions::Func getFunction(const at::ScalarType &dtype,
                                             const ReduceOp op) {
    sophon::AllreduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

class AsyncAllreduceTPUWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncAllreduceTPUWork(const std::shared_ptr<sophon::Context> &context,
                        std::vector<at::Tensor> &inputs, ReduceOp reduceOp,
                        uint32_t tag)
      : ProcessGroupSophon::AsyncWork({inputs}, "sophon:all_reduce", inputs),
        context(context), inputs(inputs), reduceOp(reduceOp), tag(tag) {}
  std::shared_ptr<sophon::Context> context;
  std::vector<at::Tensor> inputs;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void allreduce(std::vector<at::Tensor> &tensors) {
    const auto &scalarType = tensors[0].scalar_type();
    sophon::AllreduceOptions opts(context);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutputs, opts, tensors);

    bm_handle_t handle_ = tpu::TPUGetDeviceResource();
    size_t bytes_ = tensors[0].nbytes();
    bm_device_mem_t buff =
        bm_mem_from_device((unsigned long long)tensors[0].data_ptr(), bytes_);
    sg_reduce_method_t reduce_method = SG_REDUCE_SUM;

    switch (reduceOp) {
    case ReduceOp::SUM:
      reduce_method = SG_REDUCE_SUM;
      break;
    case ReduceOp::PRODUCT:
      reduce_method = SG_REDUCE_PROD;
      break;
    case ReduceOp::MIN:
      reduce_method = SG_REDUCE_MIN;
      break;
    case ReduceOp::MAX:
      reduce_method = SG_REDUCE_MAX;
      break;
    case ReduceOp::BAND:
      TORCH_CHECK(false, "Cannot use ReduceOp.BAND with Sophon");
      break;
    case ReduceOp::BOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BOR with Sophon");
      break;
    case ReduceOp::BXOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with Sophon");
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Sophon");
      break;
    case ReduceOp::UNUSED:
      break;
    }

    // for dev mem
    switch (scalarType) {
    case ::at::ScalarType::Float:
      opts.setOutputSophon(SG_DTYPE_FP32, handle_, bytes_, buff, reduce_method);
      break;
    case ::at::ScalarType::Half:
      opts.setOutputSophon(SG_DTYPE_FP16, handle_, bytes_, buff, reduce_method);
      break;
    case ::at::ScalarType::Char:
      opts.setOutputSophon(SG_DTYPE_INT8, handle_, bytes_, buff, reduce_method);
      break;
    case ::at::ScalarType::Byte:
      opts.setOutputSophon(SG_DTYPE_UINT8, handle_, bytes_, buff,
                           reduce_method);
      break;
    case ::at::ScalarType::Int:
      opts.setOutputSophon(SG_DTYPE_INT32, handle_, bytes_, buff,
                           reduce_method);
      break;
    default:
      TORCH_CHECK(false, "Invalid scalar type");
    }

    sophon::allreduce2260(opts);
  }

  void run() override { allreduce(inputs); }

  void finishWorkSophon() override { finish(); }

  template <typename T>
  void getFunction(sophon::AllreduceOptions::Func &fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  sophon::AllreduceOptions::Func getFunction(const at::ScalarType &dtype,
                                             const ReduceOp op) {
    sophon::AllreduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

} // namespace

c10::intrusive_ptr<Work>
ProcessGroupSophon::allreduce(std::vector<at::Tensor> &inputs,
                              const AllreduceOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSophon::allreduce: " + msg);
  };
  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto &device = inputs[0].device();
  switch (device.type()) {
  case at::kCPU:
    break;
  case at::kPrivateUse1:
    break;
  default:
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  const auto &layout = inputs[0].layout();

  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  if (device.type() == at::kCPU) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncAllreduceWork>(std::move(context), inputs,
                                                     opts.reduceOp, tag);
    } else { // 这里没写稀疏的
      invalidArgument("unsupported layout");
    }
  } else if (device.type() == at::kPrivateUse1) {
    if (layout == c10::kStrided) {
      // 这里可能需要不同的参数，主要要看怎么改算子接入
      work = c10::make_intrusive<AsyncAllreduceTPUWork>(
          std::move(context), inputs, opts.reduceOp, tag);
    } else { // 这里没写稀疏的
      invalidArgument("unsupported layout");
    }
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }

  enqueue(work);
  return work;
}

// reduce
namespace {

class AsyncReduceWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncReduceWork(const std::shared_ptr<sophon::Context> &context,
                  std::vector<at::Tensor> &inputs, int rootRank, int rootTensor,
                  ReduceOp reduceOp, uint32_t tag)
      : ProcessGroupSophon::AsyncWork({inputs}, "sophon:reduce", inputs),
        context(context), inputs(inputs), rootRank(rootRank),
        rootTensor(rootTensor), reduceOp(reduceOp), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void reduce(std::vector<at::Tensor> &tensors) {
    const auto &scalarType = tensors[0].scalar_type();
    sophon::ReduceOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensors[0]);
    sophon::reduce(opts);
  }

  void run() override { reduce(inputs); }
  void finishWorkSophon() override {
    // TODO
  }

protected:
  template <typename T>
  void getFunction(sophon::ReduceOptions::Func &fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  sophon::ReduceOptions::Func getFunction(const at::ScalarType &dtype,
                                          const ReduceOp op) {
    sophon::ReduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

class AsyncReduceTPUWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncReduceTPUWork(const std::shared_ptr<sophon::Context> &context,
                     std::vector<at::Tensor> &inputs, int rootRank,
                     int rootTensor, ReduceOp reduceOp, uint32_t tag)
      : ProcessGroupSophon::AsyncWork({inputs}, "sophon:reduce", inputs),
        context(context), inputs(inputs), rootRank(rootRank),
        rootTensor(rootTensor), reduceOp(reduceOp), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void reduce(std::vector<at::Tensor> &tensors) {
    const auto &scalarType = tensors[0].scalar_type();
    sophon::ReduceOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensors[0]);

    bm_handle_t handle_ = tpu::TPUGetDeviceResource();
    size_t bytes_ = tensors[0].nbytes();
    bm_device_mem_t send_buff =
        bm_mem_from_device((unsigned long long)tensors[0].data_ptr(), bytes_);
    bm_device_mem_t rec_buff =
        bm_mem_from_device((unsigned long long)tensors[0].data_ptr(), bytes_);
    sg_reduce_method_t reduce_method = SG_REDUCE_SUM;

    switch (reduceOp) {
    case ReduceOp::SUM:
      reduce_method = SG_REDUCE_SUM;
      break;
    case ReduceOp::PRODUCT:
      reduce_method = SG_REDUCE_PROD;
      break;
    case ReduceOp::MIN:
      reduce_method = SG_REDUCE_MIN;
      break;
    case ReduceOp::MAX:
      reduce_method = SG_REDUCE_MAX;
      break;
    case ReduceOp::BAND:
      TORCH_CHECK(false, "Cannot use ReduceOp.BAND with Sophon");
      break;
    case ReduceOp::BOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BOR with Sophon");
      break;
    case ReduceOp::BXOR:
      TORCH_CHECK(false, "Cannot use ReduceOp.BXOR with Sophon");
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Sophon");
      break;
    case ReduceOp::UNUSED:
      break;
    }

    // for dev mem
    switch (scalarType) {
    case ::at::ScalarType::Float:
      opts.setOutputSophon(SG_DTYPE_FP32, handle_, bytes_, send_buff, rec_buff,
                           reduce_method);
      break;
    case ::at::ScalarType::Half:
      opts.setOutputSophon(SG_DTYPE_FP16, handle_, bytes_, send_buff, rec_buff,
                           reduce_method);
      break;
    case ::at::ScalarType::Char:
      opts.setOutputSophon(SG_DTYPE_INT8, handle_, bytes_, send_buff, rec_buff,
                           reduce_method);
      break;
    case ::at::ScalarType::Byte:
      opts.setOutputSophon(SG_DTYPE_UINT8, handle_, bytes_, send_buff, rec_buff,
                           reduce_method);
      break;
    case ::at::ScalarType::Int:
      opts.setOutputSophon(SG_DTYPE_INT32, handle_, bytes_, send_buff, rec_buff,
                           reduce_method);
      break;
    default:
      TORCH_CHECK(false, "Invalid scalar type");
    }

    sophon::reduce2260(opts); // 2260 interface
  }

  void run() override { reduce(inputs); }
  void finishWorkSophon() override { finish(); }

protected:
  template <typename T>
  void getFunction(sophon::ReduceOptions::Func &fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  sophon::ReduceOptions::Func getFunction(const at::ScalarType &dtype,
                                          const ReduceOp op) {
    sophon::ReduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

} // namespace

c10::intrusive_ptr<Work>
ProcessGroupSophon::reduce(std::vector<at::Tensor> &inputs,
                           const ReduceOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSophon::reduce: " + msg);
  };

  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto &device = inputs[0].device();
  switch (device.type()) {
  case at::kCPU:
    break;
  case at::kPrivateUse1:
    break;
  default:
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }
  const auto &layout = inputs[0].layout();
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }
  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  if (device.type() == at::kCPU) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncReduceWork>(
          std::move(context), inputs, opts.rootRank, opts.rootTensor,
          opts.reduceOp, tag);
    } else {
      invalidArgument("unsupported layout");
    }
  } else if (device.type() == at::kPrivateUse1) {
    work = c10::make_intrusive<AsyncReduceTPUWork>(
        std::move(context), inputs, opts.rootRank, opts.rootTensor,
        opts.reduceOp, tag);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }

  enqueue(work);
  return work;
}

// allgather
namespace {

class AsyncAllgatherWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncAllgatherWork(const std::shared_ptr<sophon::Context> &context,
                     std::vector<std::vector<at::Tensor>> &outputs,
                     std::vector<at::Tensor> &inputs, uint32_t tag)
      : ProcessGroupSophon::AsyncWork(outputs, "sophon:all_gather", inputs),
        context(context), outputs(outputs), inputs(inputs), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const uint32_t tag;

  void allgather(std::vector<std::vector<at::Tensor>> &outputs,
                 std::vector<at::Tensor> &inputs) {
    const auto &scalarType = inputs[0].scalar_type();
    sophon::AllgatherOptions opts(context);
    opts.setTag(tag);

    // Use single flattened input tensor.
    at::Tensor flatInputTensor = flattenDenseTensors(inputs);
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    // Use single flat output tensor.
    // The first dimension corresponds to the index into outputs[N],
    // so copying into the actual output later is easy.
    at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    sophon::allgather(opts);

    // Unflatten into output tensors.
    for (auto &outputgroup : outputs) {
      for (const auto j : c10::irange(outputgroup.size())) {
        outputgroup[j].copy_(flatOutputTensor[j]);
      }
    }
  }

  void run() override { allgather(outputs, inputs); }
  void finishWorkSophon() override {
    // TODO
  }
};

class AsyncAllgatherTPUWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncAllgatherTPUWork(const std::shared_ptr<sophon::Context> &context,
                        std::vector<std::vector<at::Tensor>> &outputs,
                        std::vector<at::Tensor> &inputs, uint32_t tag)
      : ProcessGroupSophon::AsyncWork(outputs, "sophon:all_gather", inputs),
        context(context), outputs(outputs), inputs(inputs), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const uint32_t tag;

  void allgather(std::vector<std::vector<at::Tensor>> &outputs,
                 std::vector<at::Tensor> &inputs) {
    const auto &scalarType = inputs[0].scalar_type();
    sophon::AllgatherOptions opts(context);
    opts.setTag(tag);

    // Use single flattened input tensor.
    at::Tensor flatInputTensor = flattenDenseTensors(inputs);
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    // Use single flat output tensor.
    // The first dimension corresponds to the index into outputs[N],
    // so copying into the actual output later is easy.
    at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    sophon::allgather(opts);

    // Unflatten into output tensors.
    for (auto &outputgroup : outputs) {
      for (const auto j : c10::irange(outputgroup.size())) {
        outputgroup[j].copy_(flatOutputTensor[j]);
      }
    }
  }

  void run() override { allgather(outputs, inputs); }
  void finishWorkSophon() override {
    // TODO
  }
};

} // namespace

c10::intrusive_ptr<Work>
ProcessGroupSophon::allgather(std::vector<std::vector<at::Tensor>> &outputs,
                              std::vector<at::Tensor> &inputs,
                              const AllgatherOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSophon::allgather: " + msg);
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

  // assertDense(invalidArgument, inputs);

  const auto &options = inputs[0].options();
  const auto &sizes = inputs[0].sizes();
  assertTypeAndSizesMatch(invalidArgument, inputs, options, sizes);
  for (const auto &output : outputs) {
    assertTypeAndSizesMatch(invalidArgument, output, options, sizes);
  }

  const auto &device = inputs[0].device();
  switch (device.type()) {
  case at::kCPU:
    break;
  case at::kPrivateUse1:
    break;
  default:
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncAllgatherWork>(std::move(context), outputs,
                                                   inputs, tag);
  } else if (device.type() == at::kPrivateUse1) {
    work = c10::make_intrusive<AsyncAllgatherTPUWork>(std::move(context),
                                                      outputs, inputs, tag);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work>
ProcessGroupSophon::_allgather_base(at::Tensor & /*unused */,
                                    at::Tensor & /*unused */,
                                    const AllgatherOptions & /*unused */) {
  TORCH_CHECK(false, "no support for _allgather_base in Sophon process group");
}

// gather
namespace {

class AsyncGatherWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncGatherWork(const std::shared_ptr<sophon::Context> &context,
                  std::vector<std::vector<at::Tensor>> &outputs,
                  std::vector<at::Tensor> &inputs, int root, uint32_t tag)
      : ProcessGroupSophon::AsyncWork(outputs, "sophon:gather", inputs),
        context(context), outputs(outputs), inputs(inputs), root(root),
        tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const int root;
  const uint32_t tag;

  void gather(std::vector<std::vector<at::Tensor>> &outputs,
              std::vector<at::Tensor> &inputs) {
    const auto scalarType = inputs[0].scalar_type();
    sophon::GatherOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set single temporary tensor on root process.
    // This is later scattered to the separate output tensors.
    at::Tensor flatOutputTensor;
    if (context->rank == root) {
      flatOutputTensor = newLikeFlat(outputs[0]);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    }

    // Set single input tensor on all processes.
    GENERATE_ALL_TYPES(scalarType, setInput, opts, inputs[0]);
    sophon::gather(opts);

    // Unflatten into output tensors on root process.
    if (context->rank == root) {
      for (const auto i : c10::irange(outputs[0].size())) {
        outputs[0][i].copy_(flatOutputTensor[i]);
      }
    }
  }

  void run() override { gather(outputs, inputs); }
  void finishWorkSophon() override {
    // TODO
  }
};

class AsyncGatherTPUWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncGatherTPUWork(const std::shared_ptr<sophon::Context> &context,
                     std::vector<std::vector<at::Tensor>> &outputs,
                     std::vector<at::Tensor> &inputs, int root, uint32_t tag)
      : ProcessGroupSophon::AsyncWork(outputs, "sophon:gather", inputs),
        context(context), outputs(outputs), inputs(inputs), root(root),
        tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const int root;
  const uint32_t tag;

  void gather(std::vector<std::vector<at::Tensor>> &outputs,
              std::vector<at::Tensor> &inputs) {
    const auto scalarType = inputs[0].scalar_type();
    sophon::GatherOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set single temporary tensor on root process.
    // This is later scattered to the separate output tensors.
    at::Tensor flatOutputTensor;
    if (context->rank == root) {
      flatOutputTensor = newLikeFlat(outputs[0]);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    }

    // Set single input tensor on all processes.
    GENERATE_ALL_TYPES(scalarType, setInput, opts, inputs[0]);
    sophon::gather(opts);

    // Unflatten into output tensors on root process.
    if (context->rank == root) {
      for (const auto i : c10::irange(outputs[0].size())) {
        outputs[0][i].copy_(flatOutputTensor[i]);
      }
    }
  }

  void run() override { gather(outputs, inputs); }
  void finishWorkSophon() override {
    // TODO
  }
};
} // namespace

c10::intrusive_ptr<Work>
ProcessGroupSophon::gather(std::vector<std::vector<at::Tensor>> &outputs,
                           std::vector<at::Tensor> &inputs,
                           const GatherOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSophon::gather: " + msg);
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
  switch (device.type()) {
  case at::kCPU:
    break;
  case at::kPrivateUse1:
    break;
  default:
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncGatherWork>(std::move(context), outputs,
                                                inputs, opts.rootRank, tag);
  } else if (device.type() == at::kPrivateUse1) {
    work = c10::make_intrusive<AsyncGatherTPUWork>(std::move(context), outputs,
                                                   inputs, opts.rootRank, tag);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

// scatter
namespace {

class AsyncScatterWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncScatterWork(const std::shared_ptr<sophon::Context> &context,
                   std::vector<at::Tensor> &outputs,
                   std::vector<std::vector<at::Tensor>> &inputs, int root,
                   uint32_t tag)
      : ProcessGroupSophon::AsyncWork(
            {outputs}, "sophon:scatter",
            inputs.size() > 0
                ? c10::optional<std::vector<at::Tensor>>(inputs[0])
                : c10::nullopt),
        context(context), outputs(outputs), inputs(inputs), root(root),
        tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<at::Tensor> outputs;
  std::vector<std::vector<at::Tensor>> inputs;
  const int root;
  const uint32_t tag;

  void scatter(std::vector<at::Tensor> &outputs,
               std::vector<std::vector<at::Tensor>> &inputs) {
    const auto scalarType = outputs[0].scalar_type();
    sophon::ScatterOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set list of input tensors on root process
    if (context->rank == root) {
      GENERATE_ALL_TYPES(scalarType, setInputs, opts, inputs[0]);
    }

    // Set single output tensor on all processes
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputs[0]);
    sophon::scatter(opts);
  }

  void run() override { scatter(outputs, inputs); }
  void finishWorkSophon() override {
    // TODO
  }
};

class AsyncScatterTPUWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncScatterTPUWork(const std::shared_ptr<sophon::Context> &context,
                      std::vector<at::Tensor> &outputs,
                      std::vector<std::vector<at::Tensor>> &inputs, int root,
                      uint32_t tag)
      : ProcessGroupSophon::AsyncWork(
            {outputs}, "sophon:scatter",
            inputs.size() > 0
                ? c10::optional<std::vector<at::Tensor>>(inputs[0])
                : c10::nullopt),
        context(context), outputs(outputs), inputs(inputs), root(root),
        tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<at::Tensor> outputs;
  std::vector<std::vector<at::Tensor>> inputs;
  const int root;
  const uint32_t tag;

  void scatter(std::vector<at::Tensor> &outputs,
               std::vector<std::vector<at::Tensor>> &inputs) {
    const auto scalarType = outputs[0].scalar_type();
    sophon::ScatterOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set list of input tensors on root process
    if (context->rank == root) {
      GENERATE_ALL_TYPES(scalarType, setInputs, opts, inputs[0]);
    }

    // Set single output tensor on all processes
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputs[0]);
    sophon::scatter(opts);
  }

  void run() override { scatter(outputs, inputs); }
  void finishWorkSophon() override {
    // TODO
  }
};

} // namespace

c10::intrusive_ptr<Work>
ProcessGroupSophon::scatter(std::vector<at::Tensor> &outputs,
                            std::vector<std::vector<at::Tensor>> &inputs,
                            const ScatterOptions &opts) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSophon::scatter: " + msg);
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
  switch (device.type()) {
  case at::kCPU:
    break;
  case at::kPrivateUse1:
    break;
  default:
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncScatterWork>(std::move(context), outputs,
                                                 inputs, opts.rootRank, tag);
  } else if (device.type() == at::kPrivateUse1) {
    work = c10::make_intrusive<AsyncScatterTPUWork>(std::move(context), outputs,
                                                    inputs, opts.rootRank, tag);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work>
ProcessGroupSophon::reduce_scatter(std::vector<at::Tensor> &outputs,
                                   std::vector<std::vector<at::Tensor>> &inputs,
                                   const ReduceScatterOptions &opts) {
  TORCH_CHECK(false, "ProcessGroupSophon does not support reduce_scatter");
}

// alltoall
namespace {

class AsyncAlltoallWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncAlltoallWork(const std::shared_ptr<sophon::Context> &context,
                    at::Tensor &outputTensor, at::Tensor &inputTensor,
                    std::vector<int64_t> &outputCounts,
                    std::vector<int64_t> &inputCounts, uint32_t tag)
      : ProcessGroupSophon::AsyncWork(
            {{outputTensor}}, "sophon:all_to_all",
            c10::optional<std::vector<at::Tensor>>({inputTensor})),
        context(context), outputTensor(outputTensor), inputTensor(inputTensor),
        outputCounts(std::move(outputCounts)),
        inputCounts(std::move(inputCounts)), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  at::Tensor outputTensor;
  at::Tensor inputTensor;
  std::vector<int64_t> outputCounts;
  std::vector<int64_t> inputCounts;
  const uint32_t tag;

  void alltoall(at::Tensor &outputTensor, at::Tensor &inputTensor) {
    const auto scalarType = outputTensor.scalar_type();
    if (outputCounts.size() == 0 && inputCounts.size() == 0) {
      // Sophon alltoall
      sophon::AlltoallOptions opts(context);
      opts.setTag(tag);
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor);
      sophon::alltoall(opts);
    } else {
      // sophon alltoallv
      c10d::checkSplitSizes(inputCounts, inputTensor, context->size);
      c10d::checkSplitSizes(outputCounts, outputTensor, context->size);
      std::vector<int64_t> sendCounts(context->size);
      std::vector<int64_t> recvCounts(context->size);
      std::vector<int64_t> sendOffsets(context->size);
      std::vector<int64_t> recvOffsets(context->size);
      c10d::computeLengthsAndOffsets(inputCounts, inputTensor, &sendCounts,
                                     &sendOffsets);
      c10d::computeLengthsAndOffsets(outputCounts, outputTensor, &recvCounts,
                                     &recvOffsets);
      sophon::AlltoallvOptions opts(context);
      opts.setTag(tag);
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor, sendCounts);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor, recvCounts);
      sophon::alltoallv(opts);
    }
  }

  void run() override { alltoall(outputTensor, inputTensor); }
  void finishWorkSophon() override {
    // TODO
  }
};

class AsyncAlltoallTPUWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncAlltoallTPUWork(const std::shared_ptr<sophon::Context> &context,
                       at::Tensor &outputTensor, at::Tensor &inputTensor,
                       std::vector<int64_t> &outputCounts,
                       std::vector<int64_t> &inputCounts, uint32_t tag)
      : ProcessGroupSophon::AsyncWork(
            {{outputTensor}}, "sophon:all_to_all",
            c10::optional<std::vector<at::Tensor>>({inputTensor})),
        context(context), outputTensor(outputTensor), inputTensor(inputTensor),
        outputCounts(std::move(outputCounts)),
        inputCounts(std::move(inputCounts)), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  at::Tensor outputTensor;
  at::Tensor inputTensor;
  std::vector<int64_t> outputCounts;
  std::vector<int64_t> inputCounts;
  const uint32_t tag;

  void alltoall(at::Tensor &outputTensor, at::Tensor &inputTensor) {
    const auto scalarType = outputTensor.scalar_type();
    if (outputCounts.size() == 0 && inputCounts.size() == 0) {
      // Sophon alltoall
      sophon::AlltoallOptions opts(context);
      opts.setTag(tag);
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor);
      sophon::alltoall(opts);
    } else {
      // sophon alltoallv
      c10d::checkSplitSizes(inputCounts, inputTensor, context->size);
      c10d::checkSplitSizes(outputCounts, outputTensor, context->size);
      std::vector<int64_t> sendCounts(context->size);
      std::vector<int64_t> recvCounts(context->size);
      std::vector<int64_t> sendOffsets(context->size);
      std::vector<int64_t> recvOffsets(context->size);
      c10d::computeLengthsAndOffsets(inputCounts, inputTensor, &sendCounts,
                                     &sendOffsets);
      c10d::computeLengthsAndOffsets(outputCounts, outputTensor, &recvCounts,
                                     &recvOffsets);
      sophon::AlltoallvOptions opts(context);
      opts.setTag(tag);
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor, sendCounts);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor, recvCounts);
      sophon::alltoallv(opts);
    }
  }

  void run() override { alltoall(outputTensor, inputTensor); }
  void finishWorkSophon() override {
    // TODO
  }
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupSophon::alltoall_base(
    at::Tensor &outputTensor, at::Tensor &inputTensor,
    std::vector<int64_t> &outputCounts, std::vector<int64_t> &inputCounts,
    const AllToAllOptions & /* unused */) {
  static auto invalidArgument = [](const std::string &msg) {
    TORCH_CHECK(false, "ProcessGroupSophon::alltoall_base: " + msg);
  };

  TORCH_CHECK(
      outputTensor.device() == inputTensor.device(),
      "output tensor and input tensor must be on the same type of device");
  // assertDense(invalidArgument, {outputTensor});
  // assertDense(invalidArgument, {inputTensor});

  const auto &device = outputTensor.device();
  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);

  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncAlltoallWork>(
        std::move(context), outputTensor, inputTensor, outputCounts,
        inputCounts, tag);
  } else if (device.type() == at::kPrivateUse1) {
    work = c10::make_intrusive<AsyncAlltoallTPUWork>(
        std::move(context), outputTensor, inputTensor, outputCounts,
        inputCounts, tag);
  } else {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }
  enqueue(work);
  return work;
}

at::Tensor &checkSingleTensor(std::vector<at::Tensor> &tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(false, "ProcessGroupSophon::send takes a single tensor");
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
ProcessGroupSophon::send(std::vector<at::Tensor> &tensors, int dstRank,
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
  return c10::make_intrusive<SendWork>(tensor, std::move(buf));
}

c10::intrusive_ptr<Work>
ProcessGroupSophon::recv(std::vector<at::Tensor> &tensors, int srcRank,
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
  return c10::make_intrusive<RecvWork>(tensor, std::move(buf), "sophon:recv");
}

c10::intrusive_ptr<Work>
ProcessGroupSophon::recvAnysource(std::vector<at::Tensor> &tensors, int tag) {
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
  return c10::make_intrusive<RecvWork>(tensor, std::move(buf),
                                       "sophon:recvAnySource");
}

// barrier
namespace {

class AsyncBarrierWork : public ProcessGroupSophon::AsyncWork {
public:
  AsyncBarrierWork(const std::shared_ptr<sophon::Context> &context,
                   std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork,
                   uint32_t tag)
      : ProcessGroupSophon::AsyncWork({}, "sophon:barrier", c10::nullopt),
        context(context), priorWork(std::move(priorWork)), tag(tag) {}

  std::shared_ptr<sophon::Context> context;
  std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork;
  const uint32_t tag;

  void run() override {
    // Wait on prior work to complete
    for (auto &weakWork : priorWork) {
      auto work = weakWork.lock();
      if (work) {
        work->wait();
      }
    }

    sophon::BarrierOptions opts(context);
    opts.setTag(tag);
    sophon::barrier(opts);
  }
  void finishWorkSophon() override {
    // TODO
  }
};

} // namespace

c10::intrusive_ptr<Work>
ProcessGroupSophon::barrier(const BarrierOptions &opts) {
  std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork;

  // Snapshot all in progress and pending work as weak_ptr.
  // When executing a barrier, we need to ensure that all prior work
  // has completed before completing itself.
  {
    std::unique_lock<std::mutex> lock(workMutex_);
    priorWork.insert(priorWork.end(), workInProgress_.begin(),
                     workInProgress_.end());
    priorWork.insert(priorWork.end(), workQueue_.begin(), workQueue_.end());
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  auto work = c10::make_intrusive<AsyncBarrierWork>(std::move(context),
                                                    std::move(priorWork), tag);
  enqueue(work);
  return work;
}

void ProcessGroupSophon::monitoredBarrier(const BarrierOptions &opts,
                                          bool waitAllRanks) {
  C10_LOG_API_USAGE_ONCE("torch.distributed.monitored_barrier");
  // Use default timeout if no timeout was specified.
  auto monitoredBarrierTimeout =
      (opts.timeout == kUnsetTimeout) ? this->options_->timeout : opts.timeout;
  auto rank = this->getRank();
  auto t1 = nextTag();
  auto t2 = nextTag();
  std::vector<at::Tensor> commTensor = {at::tensor({rank})};
  // only enforce timeout on rank 0. This is so that other ranks aren't timed
  // out first, bringing down the job without reporting which rank timed out.
  if (rank != 0) {
    auto sendWork = send(commTensor, 0, t1);
    auto recvWork = recv(commTensor, 0, t2);
    try {
      sendWork->wait();
      recvWork->wait();
    } catch (const std::exception &e) {
      const std::string error =
          c10::str("Rank ", rank,
                   " successfully reached monitoredBarrier, but received "
                   "errors while waiting",
                   " for send/recv from rank 0. Please check rank 0 logs for "
                   "faulty rank.");
      logAndThrow(error,
                  c10::str(error, "\n Original exception: \n", e.what()));
    }
    return;
  }
  auto startTime = std::chrono::steady_clock::now();
  auto worldSize = this->getSize();
  // Mappings of rank to recvWork/sendWork respectively.
  std::map<int, c10::intrusive_ptr<Work>> recvWorkMap;
  std::map<int, c10::intrusive_ptr<Work>> sendWorkMap;
  // Kick off recvWork and wait to unblock sendWork->wait() from non-zero ranks.
  // Failed/hanging ranks will not ack this call, letting rank 0 know about the
  // failure.
  for (const auto dstRank : c10::irange(1, worldSize)) {
    recvWorkMap.insert({dstRank, recv(commTensor, dstRank, t1)});
  }

  auto waitLoop = [&](const std::map<int, c10::intrusive_ptr<Work>> &works) {
    std::vector<int> processedRanks;
    for (auto &work : works) {
      bool rankResponded = false;
      try {
        // Note: if waitAllRanks=false, we recompute the time remaining in
        // barrier and use this recomputed time in wait(). However, if
        // waitAllRanks=true, we use the original timeout, since if we use
        // up the entire timeout waiting for response from rank n, then we
        // won't have any timeout left to query ranks beginning with n + 1.
        auto remainingTime =
            getRemainingTime(startTime, monitoredBarrierTimeout, waitAllRanks);
        if (!waitAllRanks) {
          checkRemainingTime(monitoredBarrierTimeout, remainingTime,
                             processedRanks, rank);
        }
        work.second->wait(remainingTime);
        rankResponded = true;
      } catch (const std::exception &e) {
        const std::string error =
            c10::str("[Rank 0]: Rank ", work.first,
                     " failed to pass monitoredBarrier in ",
                     monitoredBarrierTimeout.count(), " ms");
        if (waitAllRanks) {
          LOG(ERROR) << error;
        } else {
          logAndThrow(error,
                      c10::str(error, "\n Original exception: \n", e.what()));
        }
      }
      if (rankResponded) {
        processedRanks.push_back(work.first);
      }
    }
    // If we are collecting all failed ranks, check if we need to throw if
    // some ranks have not responded.
    // Ensure all ranks from 1, ... WORLD_SIZE -1 have been successfully
    // processed.
    auto rankFailure = (processedRanks.size() != size_ - 1);
    if (waitAllRanks && rankFailure) {
      std::vector<int> failedRanks;
      for (const auto i : c10::irange(1, size_)) {
        if (std::find(processedRanks.begin(), processedRanks.end(), i) ==
            processedRanks.end()) {
          failedRanks.push_back(i);
        }
      }

      TORCH_INTERNAL_ASSERT(!failedRanks.empty());
      const std::string ranksStr = c10::Join(", ", failedRanks);
      const std::string error = c10::str(
          "[Rank 0]: Ranks ", ranksStr, " failed to pass monitoredBarrier in ",
          monitoredBarrierTimeout.count(), " ms");
      logAndThrow(error, error);
    }
  };

  waitLoop(recvWorkMap);
  // If we've reached here successfully, this means all ranks have acked in
  // monitoredBarrier. Unblock all ranks now by responding to their recv(). This
  // ensures that this is a true barrier in that all ranks  exit it successfully
  // or none of them do.
  for (const auto dstRank : c10::irange(1, worldSize)) {
    sendWorkMap.insert({dstRank, send(commTensor, dstRank, t2)});
  }

  waitLoop(sendWorkMap);
}

void ProcessGroupSophon::setSequenceNumberForGroup() {
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

uint64_t ProcessGroupSophon::getSequenceNumberForGroup() {
  if (sequenceNum_ == c10::nullopt) {
    return 0;
  }
  return sequenceNum_->get();
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupSophon::createProcessGroupSophon(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> &timeout) {
  auto options = Options::create();
  // Use interfaces listed in "GLOO_SOCKET_IFNAME", if set.
  char *ifnameEnv = getenv(SOPHON_SOCKET_IFNAME_ENV.c_str());
  if (ifnameEnv && strlen(ifnameEnv) > 1) {
    for (const auto &iface : split0(',', ifnameEnv)) {
      options->devices.push_back(
          ::c10d::ProcessGroupSophon::createDeviceForInterface(iface));
    }
  } else {
    // If no hostname is specified, this function looks up
    // the machine's hostname and returns a device instance
    // associated with the address that the hostname resolves to.
    options->devices.push_back(createDefaultDevice());
  }

  // options->timeout = timeout.float();
  //  NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  options->threads = options->devices.size() * 2;
  return c10::make_intrusive<ProcessGroupSophon>(store, rank, size, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupSophon",
        &ProcessGroupSophon::createProcessGroupSophon);
}

} // namespace c10d