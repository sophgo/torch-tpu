#include "TPUTorchUtils.h"

extern "C" {

  void tpu_op_timer_reset() {
    tpu::OpTimer::Instance().Clear();
  }

  void tpu_op_timer_dump() {
    tpu::OpTimer::Instance().Dump();
  }

  void tpu_op_timer_pause() {
    tpu::OpTimer::Instance().Pause();
  }

  void tpu_op_timer_start() {
    tpu::OpTimer::Instance().Start();
  }
} // extern "C"
