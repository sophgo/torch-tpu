#include "TPUEvent.h"

namespace at_tpu {

class TPUEvent::Impl{
public:
  tpuEvent_t event;
};

void TPUEvent::createEvent(c10::DeviceIndex device_index) {
  device_index_ = device_index;
  TPUGuard guard(device_index_);
  C10_TPU_CHECK(tpuEventCreate(&(event_->event)));
  LOG( INFO ) << "Event: SgrtCreateEvent is successfully executed.";
  is_created_ = true;
}

TPUEvent::~TPUEvent() {
  try {
    if (is_created_){
        auto stream = getCurrentTPUStream().stream();
        TPUGuard guard(device_index_);
        tpuEventDestroy(event_->event, stream);
    }
  }
  catch (...) {
  }
}

bool TPUEvent::query() const {
  if (!is_created_) {
    return true;
  }

  return tpuEventQuery((event_->event)) == tpuSuccess;
}

void TPUEvent::record(const TPUStream& stream) {
  if (!is_created_) {
    createEvent(stream.device_index());
  }

  TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
      " does not match recording stream's device ", stream.device_index(), ".");
  TPUGuard guard(device_index_);
  tpuEventRecord((event_->event), stream.stream());
  was_recorded_ = true;
}

void TPUEvent::record() { record(getCurrentTPUStream()); }

void TPUEvent::block(const TPUStream& stream) {
  if (is_created_) {
    TPUGuard guard(stream.device_index());
    tpuStreamWaitEvent(stream.stream(), (event_->event));
  }
}

float TPUEvent::elapsed_time(const TPUEvent& other) const {
  TORCH_CHECK(is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
  float time_ms = 0;
  C10_TPU_CHECK(tpuEventElapsedTime(&time_ms, (event_->event), other.event_->event));
  return time_ms;
}

void TPUEvent::synchronize() const {
  if (is_created_) {
    C10_TPU_CHECK(tpuEventSynchronize((event_->event)));
    LOG( INFO ) << "Event: sgrtSynchronizeEvent is successfully executed.";
  }
}

} // namespace at_tpu