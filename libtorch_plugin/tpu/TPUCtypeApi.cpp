#include "TPUTorchUtils.h"

extern "C"{

void tpu_op_timer_reset(){
    tpu::OpTimer::Instance().Clear();
}

void tpu_op_timer_dump(){
    tpu::OpTimer::Instance().Dump();
}
} // extern "C"