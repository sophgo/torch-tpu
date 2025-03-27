#include "TPUEvent.h"

#ifdef BACKEND_SG2260
#include "torch_tpu/csrc/core/Interface/sgrtInterface.h"
#endif

namespace at_tpu {

class TPUEvent::Impl{
public:
#ifdef BACKEND_SG2260
  sgrt::sgrtEvent_t event;
#endif
};

void TPUEvent::createEvent(c10::DeviceIndex device_index) {
#ifdef BACKEND_SG2260
  device_index_ = device_index;
  TPUGuard guard(device_index_);
  C10_TPU_CHECK(sgrt::SgrtCreateEventWithFlag(&(event_->event), flags_));
  SOPHON_LOGI("Event: SgrtCreateEvent is successfully executed.");
  is_created_ = true;
#endif
}

TPUEvent::~TPUEvent() {
#ifdef BACKEND_SG2260
  try {
    if (is_created_){
        TPUGuard guard(device_index_);
        sgrt::SgrtEventDestroy((event_->event));
    }
  }
  catch (...) {
  }
#endif
}

bool TPUEvent::query() const {
#ifdef BACKEND_SG2260
  if (!is_created_) {
    return true;
  }

  sgrt::sgrtEventRecordedStatus currStatus =
      sgrt::SG_EVENT_RECORDED_STATUS_NOT_READY;
  C10_TPU_CHECK(sgrt::SgrtQueryEventStatus((event_->event), &currStatus));

  if (currStatus == sgrt::SG_EVENT_RECORDED_STATUS_COMPLETE) {
    return true;
  }
  return false;
#else
  return false;
#endif
}

void TPUEvent::record(const TPUStream& stream) {
#ifdef BACKEND_SG2260
  if (!is_created_) {
    createEvent(stream.device_index());
  }

  TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
      " does not match recording stream's device ", stream.device_index(), ".");
  TPUGuard guard(device_index_);
  sgrt::SgrtEventRecord((event_->event), stream.stream());
  was_recorded_ = true;
#endif
}

void TPUEvent::record() { record(getCurrentTPUStream()); }

void TPUEvent::block(const TPUStream& stream) {
#ifdef BACKEND_SG2260
  if (is_created_) {
    TPUGuard guard(stream.device_index());
    sgrt::SgrtStreamWaitEvent(stream.stream(), (event_->event));
  }
#endif
}

float TPUEvent::elapsed_time(const TPUEvent& other) const {
#ifdef BACKEND_SG2260
  TORCH_CHECK(is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
  float time_ms = 0;
  C10_TPU_CHECK(sgrt::SgrtEventElapsedTime(&time_ms, (event_->event), other.event_->event));
  return time_ms;
#else
  return 0;
#endif
}

void TPUEvent::synchronize() const {
#ifdef BACKEND_SG2260
  if (is_created_) {
    C10_TPU_CHECK(sgrt::SgrtSynchronizeEvent((event_->event)));
    SOPHON_LOGI("Event: sgrtSynchronizeEvent is successfully executed.");
  }
#endif
}

} // namespace at_tpu