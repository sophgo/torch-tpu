#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void tpu_op_timer_reset();
void tpu_op_timer_dump();
void tpu_op_timer_start();
void tpu_op_timer_pause();
void tpu_timer_reset();
void tpu_timer_dump();
#ifdef __cplusplus
}
#endif
