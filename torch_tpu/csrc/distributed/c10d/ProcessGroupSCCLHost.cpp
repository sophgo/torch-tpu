#include <c10/util/Exception.h>

#include <chrono>
#include <exception>
#include <ratio>
#include <tuple>

#ifdef _WIN32
#include <gloo/common/win.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <sys/types.h>

#include <type_traits>

#include <gloo/allgather.h>
#include <gloo/allgatherv.h>
#include <gloo/allreduce.h>
#include <gloo/alltoall.h>
#include <gloo/alltoallv.h>
#include <gloo/barrier.h>
#include <gloo/broadcast.h>
#include <gloo/gather.h>
#include <gloo/reduce.h>
#include <gloo/scatter.h>

#include <ATen/ThreadLocalState.h>

#include <c10/util/StringUtil.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <gloo/config.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/prefix_store.h>
#include "ProcessGroupSCCLHost.hpp"
#include "SCCLHostDeviceFactory.hpp"

#ifdef _WIN32
#define GENERATE_ALL_TYPES(type, func, ...)      \
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(__VA_ARGS__);                  \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(__VA_ARGS__);                 \
      break;                                     \
    case ::at::ScalarType::Half:                 \
      func<gloo::float16>(__VA_ARGS__);          \
      break;                                     \
    case ::at::ScalarType::Char:                 \
      func<int8_t>(__VA_ARGS__);                 \
      break;                                     \
    case ::at::ScalarType::Byte:                 \
      func<uint8_t>(__VA_ARGS__);                \
      break;                                     \
    case ::at::ScalarType::Int:                  \
      func<int32_t>(__VA_ARGS__);                \
      break;                                     \
    case ::at::ScalarType::Long:                 \
      func<int64_t>(__VA_ARGS__);                \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }

#define HOST_NAME_MAX 256
#else
#define GENERATE_ALL_TYPES(type, func, args...)  \
  switch (type) {                                \
    case ::at::ScalarType::Float:                \
      func<float>(args);                         \
      break;                                     \
    case ::at::ScalarType::Double:               \
      func<double>(args);                        \
      break;                                     \
    case ::at::ScalarType::Half:                 \
      func<gloo::float16>(args);                 \
      break;                                     \
    case ::at::ScalarType::Char:                 \
      func<int8_t>(args);                        \
      break;                                     \
    case ::at::ScalarType::Byte:                 \
      func<uint8_t>(args);                       \
      break;                                     \
    case ::at::ScalarType::Int:                  \
      func<int32_t>(args);                       \
      break;                                     \
    case ::at::ScalarType::Long:                 \
      func<int64_t>(args);                       \
      break;                                     \
    default:                                     \
      TORCH_CHECK(false, "Invalid scalar type"); \
  }
#endif

namespace tpu {
  void TPUCopyHostToDevice ( void * Dst, const void * Src, size_t Size, bool non_blocking = false );
}

namespace c10d {

namespace {
static const std::string SCCL_HOST_SOCKET_IFNAME_ENV = "SCCL_HOST_SOCKET_IFNAME";
constexpr int kBytes = 8;
std::vector<std::string> split0(char separator, const std::string& string) {
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

std::chrono::milliseconds getRemainingTime(
    steady_clock_time_point startTime,
    const std::chrono::milliseconds& timeout,
    bool waitAllRanks) {
  if (waitAllRanks) {
    // See Note in monitoredBarrier
    return timeout;
  }
  auto elapsedTime = std::chrono::steady_clock::now() - startTime;
  auto remainingMillis = timeout -
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime);

  // If no more remaining time, return -1 to indicate to caller.
  if (remainingMillis.count() <= 0) {
    return std::chrono::milliseconds(-1);
  }

  return remainingMillis;
}

// Emit a LOG(ERROR) and throws using TORCH_CHECK with the given messages.
void logAndThrow(
    const std::string& logMessage,
    const std::string& errorMessage) {
  LOG(ERROR) << logMessage;
  TORCH_CHECK(false, errorMessage);
}

// For monitoredBarrier, checks remaining time left to finish processing ranks
// and throws error if timeout.
void checkRemainingTime(
    const std::chrono::milliseconds& monitoredBarrierTimeout,
    const std::chrono::milliseconds& remainingTime,
    const std::vector<int>& processedRanks,
    int currentRank) {
  const std::string kNoRemainingTimeError = c10::str(
      "Rank ",
      currentRank,
      " timed out in monitoredBarrier after ",
      monitoredBarrierTimeout.count(),
      " ms.");
  if (remainingTime.count() < 0) {
    std::string rankInfo;
    if (processedRanks.size() > 0) {
      rankInfo = c10::str(
          "Successfully processed ranks: ", c10::Join(", ", processedRanks));
    } else {
      rankInfo = "No ranks successfully processed in monitoredBarrier.";
    }
    auto error = c10::str(kNoRemainingTimeError, "\n", rankInfo);
    logAndThrow(error, error);
  }
}

typedef void (*ReduceFunc)(void*, const void*, const void*, size_t);

template <
    typename T,
    typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);
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
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Gloo");
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Gloo");
      break;
    case ReduceOp::UNUSED:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

// Bitwise AND with SFINAE guard for integral types.
template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void band(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] & tb[i];
  }
}

// Bitwise OR with SFINAE guard for integral types.
template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void bor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] | tb[i];
  }
}

// Bitwise XOR with SFINAE guard for integral types.
template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void bxor(void* c, const void* a, const void* b, size_t n) {
  auto tc = static_cast<T*>(c);
  auto ta = static_cast<const T*>(a);
  auto tb = static_cast<const T*>(b);
  for (const auto i : c10::irange(n)) {
    tc[i] = ta[i] ^ tb[i];
  }
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
ReduceFunc toFunction(const ReduceOp& r) {
  switch (r) {
    case ReduceOp::SUM:
      return ReduceFunc(&::gloo::sum<T>);
    case ReduceOp::PRODUCT:
      return ReduceFunc(&::gloo::product<T>);
    case ReduceOp::MIN:
      return ReduceFunc(&::gloo::min<T>);
    case ReduceOp::MAX:
      return ReduceFunc(&::gloo::max<T>);
    case ReduceOp::BAND:
      return ReduceFunc(&band<T>);
    case ReduceOp::BOR:
      return ReduceFunc(&bor<T>);
    case ReduceOp::BXOR:
      return ReduceFunc(&bxor<T>);
    case ReduceOp::AVG:
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Gloo");
      break;
    case ReduceOp::PREMUL_SUM:
      TORCH_CHECK(false, "Cannot use ReduceOp.PREMUL_SUM with Gloo");
      break;
    case ReduceOp::UNUSED:
      break;
  }

  TORCH_CHECK(false, "Unhandled ReduceOp");
}

template <typename T, typename O>
void setInputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setInputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor) {
  opts.setInput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setInput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setInput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutputs(O& opts, std::vector<at::Tensor>& tensors) {
  opts.setOutputs(getDataPointers<T>(tensors), tensors[0].numel());
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor) {
  opts.setOutput(getDataPointer<T>(tensor), tensor.numel());
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<size_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

template <typename T, typename O>
void setOutput(O& opts, at::Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setOutput(getDataPointer<T>(tensor), counts);
}

const auto kLoopbackAddress = "127.0.0.1";

} // namespace

// static
void ProcessGroupSCCLHost::AsyncWork::execute(c10::intrusive_ptr<AsyncWork> work) {
  if (work->recordFunctionBeforeCallback_) {
    work->recordFunctionBeforeCallback_();
  }
  try {
    work->run();
  } catch (...) {
    work->finishWorkGlooError(std::current_exception());
    return;
  }

  work->synchronize();
  work->finishWorkSCCLHost();
  work->finishWorkGloo();
}

std::vector<at::Tensor> ProcessGroupSCCLHost::AsyncWork::result() {
  TORCH_CHECK(
      isCompleted(),
      "Work needs to be completed before calling result(). "
      "Should call wait() before result().");
  TORCH_CHECK(
      outputTensors_.size() <= 1,
      "work result does not support list of lists, use .getFuture() and value()");
  return outputTensors_.size() == 0 ? std::vector<at::Tensor>()
                                    : outputTensors_.at(0);
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupSCCLHost::AsyncWork::
    getFuture() {
  return future_;
}

namespace {
c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.size() > 1) {
    return c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  }
  return c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

void returnFutureWithOutput(
    c10::intrusive_ptr<c10::ivalue::Future>& future,
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
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

inline void ProcessGroupSCCLHost::AsyncWork::recordAsyncWorkProfilingInfo(
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputTensors) {
  auto recordingFunction =
      std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
  if (recordingFunction->isActive()) {
    std::function<void()> before_handler =
        [inputTensors, profilingTitle, recordingFunction]() {
          // The work will be started and completed by different threads.
          recordingFunction->_setAsync();
          std::vector<c10::IValue> inputs;
          if (inputTensors) {
            inputs.reserve(inputTensors->size());
            for (const auto& tensor : *inputTensors) {
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

ProcessGroupSCCLHost::AsyncWork::AsyncWork(
    std::vector<std::vector<at::Tensor>> outputTensors,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputTensors)
    // Profiler: Pass nullptr as profilingTitle to parent constructor to
    // replace default profiler implementation with async version that reports
    // correct timestamps for work that is asynchronously executed.
    : Work(-1, OpType::UNKNOWN, nullptr, inputTensors),
      outputTensors_(std::move(outputTensors)),
      future_(createFutureAsOutput(outputTensors)) {
  if (profilingTitle != nullptr) {
    recordAsyncWorkProfilingInfo(profilingTitle, inputTensors);
  }
}

void ProcessGroupSCCLHost::AsyncWork::finishWorkGlooError(std::exception_ptr eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupSCCLHost::AsyncWork::finishWorkGloo() {
  returnFutureWithOutput(future_, outputTensors_);
  finish();
}


ProcessGroupSCCLHost::SendWork::SendWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer)
    : Work(
          -1,
          OpType::SEND,
          "gloo:send",
          c10::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor),
      buffer_(std::move(buffer)) {}

bool ProcessGroupSCCLHost::SendWork::wait(std::chrono::milliseconds timeout) {
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

void ProcessGroupSCCLHost::SendWork::abort() {
  buffer_->abortWaitSend();
}

ProcessGroupSCCLHost::RecvWork::RecvWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    const char* profilingTitle)
    : Work(
          -1,
          OpType::UNKNOWN,
          profilingTitle,
          c10::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor),
      buffer_(std::move(buffer)),
      srcRank_(-1) {}

int ProcessGroupSCCLHost::RecvWork::sourceRank() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return srcRank_;
}

bool ProcessGroupSCCLHost::RecvWork::wait(std::chrono::milliseconds timeout) {
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

void ProcessGroupSCCLHost::RecvWork::abort() {
  buffer_->abortWaitRecv();
}

ProcessGroupSCCLHost::RecvTPUWork::RecvTPUWork(
    at::Tensor& tensor,
    at::Tensor tensor_cpu,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    const char* profilingTitle)
    : Work(
          -1,
          OpType::UNKNOWN,
          profilingTitle,
          c10::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor),
      tensor_cpu(tensor_cpu),
      buffer_(std::move(buffer)),
      srcRank_(-1) {}

int ProcessGroupSCCLHost::RecvTPUWork::sourceRank() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return srcRank_;
}

bool ProcessGroupSCCLHost::RecvTPUWork::wait(std::chrono::milliseconds timeout) {
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
  tpu::TPUCopyHostToDevice ( tensor_.data_ptr(), tensor_cpu.data_ptr(), tensor_cpu.nbytes() );
  // Completes the Work object and throws the exception.
  finishAndThrow(exception);
  return recvCompleted;
}

void ProcessGroupSCCLHost::RecvTPUWork::abort() {
  buffer_->abortWaitRecv();
}

ProcessGroupSCCLHost::Options::Options(std::chrono::milliseconds timeout)
    : ProcessGroup::Options(SCCL_HOST_BACKEND_NAME, timeout), threads(2) {}

namespace {

void socketInitialize() {
#ifdef _WIN32
  ::gloo::init_winsock();
#endif
}

// Gloo assumes that this machine's hostname can always be resolved
// to an address. If it doesn't it throws a runtime error saying
// that it can't be resolved. Instead of catching it, we choose
// to proactively check if an address can be resolved, so we can
// gracefully fall back to an alternative if it doesn't.
bool doesHostnameResolveToUsableAddress(const std::string& hostname) {
  socketInitialize();
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* result;
  auto rv = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
  if (rv < 0) {
    return false;
  }
  struct addrinfo* rp;
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

std::shared_ptr<::gloo::transport::Device> ProcessGroupSCCLHost::
    createDeviceForInterface(const std::string& interface_name) {
  return ::c10d::SCCLHostDeviceFactory::makeDeviceForInterface(interface_name);
}

std::shared_ptr<::gloo::transport::Device> ProcessGroupSCCLHost::
    createDeviceForHostname(const std::string& hostname) {
  TORCH_CHECK(
      doesHostnameResolveToUsableAddress(hostname),
      "Cannot resolve ",
      hostname,
      " to a (local) address");
  return ::c10d::SCCLHostDeviceFactory::makeDeviceForHostname(hostname);
}

#if defined(__linux__) || defined(_WIN32)
std::shared_ptr<::gloo::transport::Device> ProcessGroupSCCLHost::
    createDefaultDevice() {
  // Use the hostname to resolve the network address to
  // use. Note: if the hostname does not resolve to an address (e.g.
  // because of misconfigured /etc/hosts file), this will not work.
  socketInitialize();
 std::array<char, HOST_NAME_MAX> hostname{};
  auto rv = gethostname(hostname.data(), HOST_NAME_MAX);
 if (rv != 0) {
    //throw std::system_error(errno, std::system_category());
  }

  // Use this machine's hostname if it resolves to an address.
  if (doesHostnameResolveToUsableAddress(hostname.data())) {
   return ::c10d::SCCLHostDeviceFactory::makeDeviceForHostname(hostname.data());
  }

  // Otherwise, use the loopback address.
  TORCH_WARN_ONCE(
      "Unable to resolve hostname to a (local) address. ",
      "Using the loopback address as fallback. ",
      "Manually set the network interface to bind to with SCCL_HOST_SOCKET_IFNAME.");
 return createDeviceForHostname(kLoopbackAddress);
}
#endif


ProcessGroupSCCLHost::ProcessGroupSCCLHost(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : ProcessGroup(rank, size),
      store_(new GlooStore(store)),
      options_(options),
      stop_(false),
      collectiveCounter_(0) {
  auto& devices = options->devices;
  if (devices.empty()) {
    TORCH_CHECK(false, "No device(s) specified");
  }

  // Create and connect a context for every device.
  //
  // Note that the same device can be specified multiple times, either
  // the same object, or the same logical device as different objects.
  // Either mode is fine and only has performance implications.
  //
  // Using the same object multiple times means all contexts share a
  // single I/O thread. If you use different objects for the same
  // logical device they will have independent I/O threads. The latter
  // option is needed if you have a fast NIC that cannot be saturated
  // by a single I/O thread.
  //
  contexts_.reserve(options->devices.size());
  for (const auto i : c10::irange(options->devices.size())) {
    auto context = std::make_shared<::gloo::rendezvous::Context>(rank_, size_);
    auto store = ::gloo::rendezvous::PrefixStore(std::to_string(i), *store_);
    context->setTimeout(options->timeout);
    context->connectFullMesh(store, options->devices[i]);
    contexts_.push_back(std::move(context));
  }

  // Every worker thread stores the AsyncWork object it's currently
  // working on in the workInProgress_ vector. It must have size equal
  // to the number of workers such that they can simply index into it
  // using the worker index they are started with.
  workInProgress_.resize(options->threads);

  threads_.resize(options->threads);
  for (const auto i : c10::irange(threads_.size())) {
    threads_[i] = std::thread(&ProcessGroupSCCLHost::runLoop, this, i);
  }

  init();
}

ProcessGroupSCCLHost::~ProcessGroupSCCLHost() {
  std::unique_lock<std::mutex> lock(workMutex_);
  workConsumeCV_.wait(lock, [&] { return workQueue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();

  workProduceCV_.notify_all();

  // Wait for worker threads to terminate
  for (auto& thread : threads_) {
    thread.join();
  }
}

uint32_t ProcessGroupSCCLHost::nextTag() {
  return collectiveCounter_++;
}

std::shared_ptr<::gloo::Context> ProcessGroupSCCLHost::getContext(uint32_t tag) {
  return contexts_[tag % contexts_.size()];
}

void ProcessGroupSCCLHost::runLoop(int workerIndex) {
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

void ProcessGroupSCCLHost::enqueue(c10::intrusive_ptr<AsyncWork> work) {
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

class AsyncBroadcastTPUWork : public ProcessGroupSCCLHost::AsyncWork {
 public:
  AsyncBroadcastTPUWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor> inputs_cpu,
      int rootRank,
      int rootTensor,
      uint32_t tag)
      : ProcessGroupSCCLHost::AsyncWork({inputs_cpu}, "gloo:broadcast", inputs_cpu),
        context(context),
        inputs(inputs),
        inputs_cpu(inputs_cpu),
        rootRank(rootRank),
        rootTensor(rootTensor),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> inputs_cpu;
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;

  void broadcast(at::Tensor& tensor) {
    const auto& scalarType = tensor.scalar_type();
    gloo::BroadcastOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    gloo::broadcast(opts);
  }

  void run() override {
    broadcast(inputs_cpu[rootTensor]);

    // Copy to non-root tensors
    for (const auto i : c10::irange(inputs_cpu.size())) {
      if (i == static_cast<size_t>(rootTensor)) {
        continue;
      }
      inputs_cpu[i].copy_(inputs_cpu[rootTensor]);
    }
  }
  void finishWorkSCCLHost() override{
    for (size_t i = 0; i < inputs.size(); i++){
      tpu::TPUCopyHostToDevice ( inputs[i].data_ptr(), inputs_cpu[i].data_ptr(), inputs_cpu[i].nbytes() );
    }
  }
};

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::broadcast(
    std::vector<at::Tensor>& inputs,
    const BroadcastOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupSCCLHost::broadcast: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(invalidArgument, opts.rootTensor, inputs.size());
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  
  std::vector<at::Tensor> inputs_cpu;
  for (size_t i = 0; i < inputs.size(); i++){
    inputs_cpu.push_back(inputs[i].cpu());
  }
  work = c10::make_intrusive<AsyncBroadcastTPUWork>(
      std::move(context), inputs, inputs_cpu, opts.rootRank, opts.rootTensor, tag);

  enqueue(work);
  return work;
}

class AsyncAllreduceTPUWork : public ProcessGroupSCCLHost::AsyncWork {
 public:
  AsyncAllreduceTPUWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor> inputs_cpu,
      ReduceOp reduceOp,
      uint32_t tag)
      : ProcessGroupSCCLHost::AsyncWork({inputs_cpu}, "gloo:all_reduce", inputs_cpu),
        context(context),
        inputs(inputs),
        inputs_cpu(inputs_cpu),
        reduceOp(reduceOp),
        tag(tag) {
        }

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> inputs_cpu;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void allreduce(std::vector<at::Tensor>& tensors) {
    const auto& scalarType = tensors[0].scalar_type();
    gloo::AllreduceOptions opts(context);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    opts.setTag(tag);
    GENERATE_ALL_TYPES(scalarType, setOutputs, opts, tensors);
    gloo::allreduce(opts);
  }

  void run() override {
    allreduce(inputs_cpu);
  }

  void finishWorkSCCLHost() override{
    for (size_t i = 0; i < inputs.size(); i++){
      tpu::TPUCopyHostToDevice ( inputs[i].data_ptr(), inputs_cpu[i].data_ptr(), inputs_cpu[i].nbytes() );
    }
  }

  template <typename T>
  void getFunction(gloo::AllreduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  gloo::AllreduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp op) {
    gloo::AllreduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::allreduce(
    std::vector<at::Tensor>& inputs,
    const AllreduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupSCCLHost::allreduce: " + msg);
  };

  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  const auto& layout = inputs[0].layout();
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  if (layout == c10::kStrided) {
    std::vector<at::Tensor> inputs_cpu;
    for (size_t i = 0; i < inputs.size(); i++){
      inputs_cpu.push_back(inputs[i].cpu());
    }
    work = c10::make_intrusive<AsyncAllreduceTPUWork>(
      std::move(context), inputs, inputs_cpu, opts.reduceOp, tag);
  } else {
    invalidArgument("unsupport layout");
  }

  enqueue(work);
  return work;
}

class AsyncReduceTPUWork : public ProcessGroupSCCLHost::AsyncWork {
 public:
  AsyncReduceTPUWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor> inputs_cpu,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag)
      : ProcessGroupSCCLHost::AsyncWork({inputs_cpu}, "gloo:reduce", inputs_cpu),
        context(context),
        inputs(inputs),
        inputs_cpu(inputs_cpu),
        rootRank(rootRank),
        rootTensor(rootTensor),
        reduceOp(reduceOp),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> inputs_cpu;
  const int rootRank;
  const int rootTensor;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void reduce(std::vector<at::Tensor>& tensors) {
    const auto& scalarType = tensors[0].scalar_type();
    gloo::ReduceOptions opts(context);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensors[0]);
    gloo::reduce(opts);
  }

  void run() override {
    reduce(inputs_cpu);
  }

  void finishWorkSCCLHost() override{
    for (size_t i = 0; i < inputs.size(); i++){
      tpu::TPUCopyHostToDevice ( inputs[i].data_ptr(), inputs_cpu[i].data_ptr(), inputs_cpu[i].nbytes() );
    }
  }

 protected:
  template <typename T>
  void getFunction(gloo::ReduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  gloo::ReduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp op) {
    gloo::ReduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::reduce(
    std::vector<at::Tensor>& inputs,
    const ReduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupSCCLHost::reduce: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(invalidArgument, opts.rootTensor, inputs.size());
  assertSingleElement(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);

  std::vector<at::Tensor> inputs_cpu;
  for (size_t i = 0; i < inputs.size(); i++){
    inputs_cpu.push_back(inputs[i].cpu());
  }
  work = c10::make_intrusive<AsyncReduceTPUWork>(
      std::move(context),
      inputs,
      inputs_cpu,
      opts.rootRank,
      opts.rootTensor,
      opts.reduceOp,
      tag);

  enqueue(work);
  return work;
}


class AsyncAllgatherTPUWork : public ProcessGroupSCCLHost::AsyncWork {
 public:
  AsyncAllgatherTPUWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<std::vector<at::Tensor>> outputs_cpu,
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor> inputs_cpu,
      uint32_t tag)
      : ProcessGroupSCCLHost::AsyncWork(outputs_cpu, "gloo:all_gather", inputs_cpu),
        context(context),
        outputs(outputs),
        outputs_cpu(outputs_cpu),
        inputs(inputs),
        inputs_cpu(inputs_cpu),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<std::vector<at::Tensor>> outputs_cpu;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> inputs_cpu;
  const uint32_t tag;

  void allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs) {
    const auto& scalarType = inputs[0].scalar_type();
    gloo::AllgatherOptions opts(context);
    opts.setTag(tag);

    // Use single flattened input tensor.
    at::Tensor flatInputTensor = flattenDenseTensors(inputs);
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    // Use single flat output tensor.
    // The first dimension corresponds to the index into outputs[N],
    // so copying into the actual output later is easy.
    at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    gloo::allgather(opts);

    // Unflatten into output tensors.
    for (auto& outputgroup : outputs) {
      for (const auto j : c10::irange(outputgroup.size())) {
        outputgroup[j].copy_(flatOutputTensor[j]);
      }
    }
  }

  void run() override {
    allgather(outputs_cpu, inputs_cpu);
  }
  void finishWorkSCCLHost() override{
    for (size_t i = 0; i < inputs.size(); i++){
      tpu::TPUCopyHostToDevice ( inputs[i].data_ptr(), inputs_cpu[i].data_ptr(), inputs_cpu[i].nbytes() );
    }
    for (size_t i = 0; i < outputs.size(); i++){
      for (size_t j = 0; j < outputs[i].size(); j++)
      tpu::TPUCopyHostToDevice ( outputs[i][j].data_ptr(), outputs_cpu[i][j].data_ptr(), outputs_cpu[i][j].nbytes() );
    }
  }
};

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupSCCLHost::allgather: " + msg);
  };

  if (inputs.size() == 0) {
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
      invalidArgument(
          "invalid output tensor list at index " + std::to_string(i) +
          " (expected length " + std::to_string(expected) + ", got " +
          std::to_string(actual) + ")");
    }
  }

  assertDense(invalidArgument, inputs);

  // Expect all input/output tensors to have the same type and sizes
  const auto& options = inputs[0].options();
  const auto& sizes = inputs[0].sizes();
  assertTypeAndSizesMatch(invalidArgument, inputs, options, sizes);
  for (const auto& output : outputs) {
    assertTypeAndSizesMatch(invalidArgument, output, options, sizes);
  }

  const auto& device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);

  std::vector<std::vector<at::Tensor>> outputs_cpu;
  for (size_t i = 0; i < outputs.size(); i++){
    std::vector<at::Tensor> outputs_cpu_;
    for (size_t j = 0; j < outputs[i].size(); j++){
      outputs_cpu_.push_back(outputs[i][j].cpu());
    }
    outputs_cpu.push_back(outputs_cpu_);
  }
  std::vector<at::Tensor> inputs_cpu;
  for (size_t i = 0; i < inputs.size(); i++){
    inputs_cpu.push_back(inputs[i].cpu());
  }
  work = c10::make_intrusive<AsyncAllgatherTPUWork>(
    std::move(context), outputs, outputs_cpu, inputs, inputs_cpu, tag);

  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::_allgather_base(
    at::Tensor& /*unused */,
    at::Tensor& /*unused */,
    const AllgatherOptions& /*unused */) {
  TORCH_CHECK(false, "no support for _allgather_base in scclHost process group");
}

class AsyncGatherTPUWork : public ProcessGroupSCCLHost::AsyncWork {
 public:
  AsyncGatherTPUWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<std::vector<at::Tensor>> outputs_cpu,
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor> inputs_cpu,
      int root,
      uint32_t tag)
      : ProcessGroupSCCLHost::AsyncWork(outputs_cpu, "gloo:gather", inputs_cpu),
        context(context),
        outputs(outputs),
        outputs_cpu(outputs_cpu),
        inputs(inputs),
        inputs_cpu(inputs_cpu),
        root(root),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<std::vector<at::Tensor>> outputs_cpu;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> inputs_cpu;
  const int root;
  const uint32_t tag;

  void gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs) {
    const auto scalarType = inputs[0].scalar_type();
    gloo::GatherOptions opts(context);
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
    gloo::gather(opts);

    // Unflatten into output tensors on root process.
    if (context->rank == root) {
      for (const auto i : c10::irange(outputs[0].size())) {
        outputs[0][i].copy_(flatOutputTensor[i]);
      }
    }
  }

  void run() override {
    gather(outputs_cpu, inputs_cpu);
  }
  void finishWorkSCCLHost() override{
    for (size_t i = 0; i < inputs.size(); i++){
      tpu::TPUCopyHostToDevice ( inputs[i].data_ptr(), inputs_cpu[i].data_ptr(), inputs_cpu[i].nbytes() );
    }
    for (size_t i = 0; i < outputs.size(); i++){
      for (size_t j = 0; j < outputs[i].size(); j++)
      tpu::TPUCopyHostToDevice ( outputs[i][j].data_ptr(), outputs_cpu[i][j].data_ptr(), outputs_cpu[i][j].nbytes() );
    }
  }
};

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupSCCLHost::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementInput(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);

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

    const auto& options = inputs[0].options();
    const auto& sizes = inputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, outputs[0], options, sizes);
  } else {
    if (outputs.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
  }

  const auto& device = inputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);

  std::vector<std::vector<at::Tensor>> outputs_cpu;
  for (size_t i = 0; i < outputs.size(); i++){
    std::vector<at::Tensor> outputs_cpu_;
    for (size_t j = 0; j < outputs[i].size(); j++){
      outputs_cpu_.push_back(outputs[i][j].cpu());
    }
    outputs_cpu.push_back(outputs_cpu_);
  }
  std::vector<at::Tensor> inputs_cpu;
  for (size_t i = 0; i < inputs.size(); i++){
    inputs_cpu.push_back(inputs[i].cpu());
  }
  work = c10::make_intrusive<AsyncGatherTPUWork>(
    std::move(context), outputs, outputs_cpu, inputs, inputs_cpu, opts.rootRank, tag);

  enqueue(work);
  return work;
}

class AsyncScatterTPUWork : public ProcessGroupSCCLHost::AsyncWork {
 public:
  AsyncScatterTPUWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor> outputs_cpu,
      std::vector<std::vector<at::Tensor>>& inputs,
      std::vector<std::vector<at::Tensor>> inputs_cpu,
      int root,
      uint32_t tag)
      : ProcessGroupSCCLHost::AsyncWork(
            {outputs_cpu},
            "gloo:scatter",
            inputs_cpu.size() > 0
                ? c10::optional<std::vector<at::Tensor>>(inputs_cpu[0])
                : c10::nullopt),
        context(context),
        outputs(outputs),
        outputs_cpu(outputs_cpu),
        inputs(inputs),
        inputs_cpu(inputs_cpu),
        root(root),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<at::Tensor> outputs;
  std::vector<at::Tensor> outputs_cpu;
  std::vector<std::vector<at::Tensor>> inputs;
  std::vector<std::vector<at::Tensor>> inputs_cpu;
  const int root;
  const uint32_t tag;

  void scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs) {
    const auto scalarType = outputs[0].scalar_type();
    gloo::ScatterOptions opts(context);
    opts.setRoot(root);
    opts.setTag(tag);

    // Set list of input tensors on root process
    if (context->rank == root) {
      GENERATE_ALL_TYPES(scalarType, setInputs, opts, inputs[0]);
    }

    // Set single output tensor on all processes
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputs[0]);
    gloo::scatter(opts);
  }

  void run() override {
    scatter(outputs_cpu, inputs_cpu);
  }
  void finishWorkSCCLHost() override{
    for (size_t i = 0; i < outputs.size(); i++){
      tpu::TPUCopyHostToDevice ( outputs[i].data_ptr(), outputs_cpu[i].data_ptr(), outputs_cpu[i].nbytes() );
    }
    for (size_t i = 0; i < inputs.size(); i++){
      for (size_t j = 0; j < inputs[i].size(); j++)
      tpu::TPUCopyHostToDevice ( inputs[i][j].data_ptr(), inputs_cpu[i][j].data_ptr(), inputs_cpu[i][j].nbytes() );
    }
  }
};

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupSCCLHost::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertSingleElementOutput(invalidArgument, outputs);
  assertDense(invalidArgument, outputs);

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
    const auto& options = outputs[0].options();
    const auto& sizes = outputs[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, inputs[0], options, sizes);
  } else {
    if (inputs.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
  }

  const auto& device = outputs[0].device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);

  std::vector<std::vector<at::Tensor>> inputs_cpu;
  for (size_t i = 0; i < inputs.size(); i++){
    std::vector<at::Tensor> inputs_cpu_;
    for (size_t j = 0; j < inputs[i].size(); j++){
      inputs_cpu_.push_back(inputs[i][j].cpu());
    }
    inputs_cpu.push_back(inputs_cpu_);
  }
  std::vector<at::Tensor> outputs_cpu;
  for (size_t i = 0; i < outputs.size(); i++){
    outputs_cpu.push_back(outputs[i].cpu());
  }
  work = c10::make_intrusive<AsyncScatterTPUWork>(
    std::move(context), outputs, outputs_cpu, inputs, inputs_cpu, opts.rootRank, tag);

  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupSCCLHost does not support reduce_scatter");
}

class AsyncAlltoallTPUWork : public ProcessGroupSCCLHost::AsyncWork {
 public:
  AsyncAlltoallTPUWork(
      const std::shared_ptr<gloo::Context>& context,
      at::Tensor& outputTensor,
      at::Tensor outputTensor_cpu,
      at::Tensor& inputTensor,
      at::Tensor inputTensor_cpu,
      std::vector<int64_t>& outputCounts,
      std::vector<int64_t>& inputCounts,
      uint32_t tag)
      : ProcessGroupSCCLHost::AsyncWork(
            {{outputTensor_cpu}},
            "gloo:all_to_all",
            c10::optional<std::vector<at::Tensor>>({inputTensor_cpu})),
        context(context),
        outputTensor(outputTensor),
        outputTensor_cpu(outputTensor_cpu),
        inputTensor(inputTensor),
        inputTensor_cpu(inputTensor_cpu),
        outputCounts(std::move(outputCounts)),
        inputCounts(std::move(inputCounts)),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  at::Tensor outputTensor;
  at::Tensor outputTensor_cpu;
  at::Tensor inputTensor;
  at::Tensor inputTensor_cpu;
  std::vector<int64_t> outputCounts;
  std::vector<int64_t> inputCounts;
  const uint32_t tag;

  void alltoall(at::Tensor& outputTensor, at::Tensor& inputTensor) {
    const auto scalarType = outputTensor.scalar_type();
    if (outputCounts.size() == 0 && inputCounts.size() == 0) {
      // Gloo alltoall
      gloo::AlltoallOptions opts(context);
      opts.setTag(tag);
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor);
      gloo::alltoall(opts);
    } else {
      // Gloo alltoallv
      c10d::checkSplitSizes(inputCounts, inputTensor, context->size);
      c10d::checkSplitSizes(outputCounts, outputTensor, context->size);
      std::vector<int64_t> sendCounts(context->size);
      std::vector<int64_t> recvCounts(context->size);
      std::vector<int64_t> sendOffsets(context->size);
      std::vector<int64_t> recvOffsets(context->size);
      c10d::computeLengthsAndOffsets(
          inputCounts, inputTensor, &sendCounts, &sendOffsets);
      c10d::computeLengthsAndOffsets(
          outputCounts, outputTensor, &recvCounts, &recvOffsets);
      gloo::AlltoallvOptions opts(context);
      opts.setTag(tag);
      GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor, sendCounts);
      GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor, recvCounts);
      gloo::alltoallv(opts);
    }
  }

  void run() override {
    alltoall(outputTensor_cpu, inputTensor_cpu);
  }
  void finishWorkSCCLHost() override{
    tpu::TPUCopyHostToDevice ( outputTensor.data_ptr(), outputTensor_cpu.data_ptr(), outputTensor_cpu.nbytes() );
    tpu::TPUCopyHostToDevice ( inputTensor.data_ptr(), inputTensor_cpu.data_ptr(), inputTensor_cpu.nbytes() );
    
  }
};

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputCounts,
    std::vector<int64_t>& inputCounts,
    const AllToAllOptions& /* unused */) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupSCCLHost::alltoall_base: " + msg);
  };

  TORCH_CHECK(
      outputTensor.device() == inputTensor.device(),
      "output tensor and input tensor must be on the same type of device");
  assertDense(invalidArgument, {outputTensor});
  assertDense(invalidArgument, {inputTensor});

  const auto& device = outputTensor.device();
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type", device.type()));
  }
  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);

  work = c10::make_intrusive<AsyncAlltoallTPUWork>(
    std::move(context),
    outputTensor,
    outputTensor.cpu(),
    inputTensor,
    inputTensor.cpu(),
    outputCounts,
    inputCounts,
    tag);

  enqueue(work);
  return work;
}

static inline at::Tensor& checkSingleTensor(std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(false, "ProcessGroupSCCLHost::send takes a single tensor");
  }
  auto& tensor = tensors[0];
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }
  return tensor;
}

static inline uint32_t checkTag(int32_t tag) {
  TORCH_CHECK(tag >= 0, "Tag must be nonnegative");
  return (uint32_t)tag;
}

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto tensor_cpu = tensor.cpu();
  auto ptr = tensor_cpu.data_ptr();
  auto size = tensor_cpu.numel() * tensor_cpu.element_size();

  // Construct unbound buffer.
  auto context = getContext(tag);
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->send(dstRank, utag);

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the send.
  return c10::make_intrusive<SendWork>(tensor_cpu, std::move(buf));
}

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto tensor_cpu = tensor.cpu();
  auto ptr = tensor_cpu.data_ptr();
  auto size = tensor_cpu.numel() * tensor_cpu.element_size();

  // Construct unbound buffer.
  auto context = getContext(tag);
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->recv(srcRank, utag);

  // The work captures the tensor to prevent it being deallocated and
  // the unbound buffer to synchronize on completion of the recv.
  const auto& device = tensor.device();
  if (device.type() == at::kPrivateUse1){
    return c10::make_intrusive<RecvTPUWork>(tensor, tensor_cpu, std::move(buf), "gloo:recv");
  }

  return c10::make_intrusive<RecvWork>(tensor_cpu, std::move(buf), "gloo:recv");
}

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  auto& tensor = checkSingleTensor(tensors);
  auto utag = checkTag(tag);
  auto tensor_cpu = tensor.cpu();
  auto ptr = tensor_cpu.data_ptr();
  auto size = tensor_cpu.numel() * tensor_cpu.element_size();

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
  const auto& device = tensor.device();
  if (device.type() == at::kPrivateUse1){
    return c10::make_intrusive<RecvTPUWork>(tensor, tensor_cpu, std::move(buf), "gloo:recvAnySource");
  }

  return c10::make_intrusive<RecvWork>(
      tensor_cpu, std::move(buf), "gloo:recvAnySource");
}

namespace {

class AsyncBarrierWork : public ProcessGroupSCCLHost::AsyncWork {
 public:
  AsyncBarrierWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork,
      uint32_t tag)
      : ProcessGroupSCCLHost::AsyncWork({}, "gloo:barrier", c10::nullopt),
        context(context),
        priorWork(std::move(priorWork)),
        tag(tag) {}

  std::shared_ptr<gloo::Context> context;
  std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork;
  const uint32_t tag;

  void run() override {
    // Wait on prior work to complete
    for (auto& weakWork : priorWork) {
      auto work = weakWork.lock();
      if (work) {
        work->wait();
      }
    }

    gloo::BarrierOptions opts(context);
    opts.setTag(tag);
    gloo::barrier(opts);
  }
  void finishWorkSCCLHost() override{
    // TODO
  }
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupSCCLHost::barrier(const BarrierOptions& opts) {
  std::vector<c10::weak_intrusive_ptr<AsyncWork>> priorWork;

  // Snapshot all in progress and pending work as weak_ptr.
  // When executing a barrier, we need to ensure that all prior work
  // has completed before completing itself.
  {
    std::unique_lock<std::mutex> lock(workMutex_);
    priorWork.insert(
        priorWork.end(), workInProgress_.begin(), workInProgress_.end());
    priorWork.insert(priorWork.end(), workQueue_.begin(), workQueue_.end());
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  auto work = c10::make_intrusive<AsyncBarrierWork>(
      std::move(context), std::move(priorWork), tag);
  enqueue(work);
  return work;
}

void ProcessGroupSCCLHost::monitoredBarrier(
    const BarrierOptions& opts,
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
    } catch (const std::exception& e) {
      const std::string error = c10::str(
          "Rank ",
          rank,
          " successfully reached monitoredBarrier, but received errors while waiting",
          " for send/recv from rank 0. Please check rank 0 logs for faulty rank.");
      logAndThrow(
          error, c10::str(error, "\n Original exception: \n", e.what()));
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

  auto waitLoop = [&](const std::map<int, c10::intrusive_ptr<Work>>& works) {
    std::vector<int> processedRanks;
    for (auto& work : works) {
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
          checkRemainingTime(
              monitoredBarrierTimeout, remainingTime, processedRanks, rank);
        }
        work.second->wait(remainingTime);
        rankResponded = true;
      } catch (const std::exception& e) {
        const std::string error = c10::str(
            "[Rank 0]: Rank ",
            work.first,
            " failed to pass monitoredBarrier in ",
            monitoredBarrierTimeout.count(),
            " ms");
        if (waitAllRanks) {
          LOG(ERROR) << error;
        } else {
          logAndThrow(
              error, c10::str(error, "\n Original exception: \n", e.what()));
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
    auto rankFailure = (processedRanks.size() != (size_t)size_ - 1);
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
          "[Rank 0]: Ranks ",
          ranksStr,
          " failed to pass monitoredBarrier in ",
          monitoredBarrierTimeout.count(),
          " ms");
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

void ProcessGroupSCCLHost::setSequenceNumberForGroup() {
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

uint64_t ProcessGroupSCCLHost::getSequenceNumberForGroup() {
  if (sequenceNum_ == c10::nullopt) {
    return 0;
  }
  return sequenceNum_->get();
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupSCCLHost::createProcessGroupSCCLHost(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  auto options = Options::create();
  // Use interfaces listed in "SCCL_HOST_SOCKET_IFNAME", if set.
  char* ifnameEnv = getenv(SCCL_HOST_SOCKET_IFNAME_ENV.c_str());
  if (ifnameEnv && strlen(ifnameEnv) > 1) {
    for (const auto& iface : split0(',', ifnameEnv)) {
      options->devices.push_back(
          ::c10d::ProcessGroupSCCLHost::createDeviceForInterface(iface));
    }
  } else {
    // If no hostname is specified, this function looks up
    // the machine's hostname and returns a device instance
    // associated with the address that the hostname resolves to.
    options->devices.push_back( createDefaultDevice() );
  }

  //options->timeout = timeout.float();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  options->threads = options->devices.size() * 2;
  return c10::make_intrusive<ProcessGroupSCCLHost>(
      store, rank, size, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupSCCLHost", &ProcessGroupSCCLHost::createProcessGroupSCCLHost);
}

__attribute__((constructor)) void ProcessGroupSCCLHostConstructor() {
  py::object module = py::module::import("torch.distributed");
  py::object register_backend =
      module.attr("Backend").attr("register_backend");
  register_backend("SCCLHOST", py::cpp_function(ProcessGroupSCCLHost::createProcessGroupSCCLHost));
}

} // namespace c10d
