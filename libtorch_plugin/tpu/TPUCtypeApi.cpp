#include "TPUTorchUtils.h"

extern "C" {

  void tpu_op_timer_reset() {
    tpu::OpTimer::Instance().Clear();
  }

  void tpu_op_timer_dump() {
    tpu::OpTimer::Instance().Dump();
  }

  void tpu_timer_reset() {
    tpu::GlobalTimer::Instance().Reset();
  }

  void tpu_op_timer_dump() {
    tpu::GlobalTimer::Instance().Dump();
  }
} // extern "C"
