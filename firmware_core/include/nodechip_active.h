#ifndef NODECHIP_ACTIVE_H_
#define NODECHIP_ACTIVE_H_

#include "tpu_utils.h"
#include "tpu_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_active(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    const int* shape,
    int shape_dim,
    data_type_t dtype,
    sg_active_type_t active_type,
    float* coef);

#ifdef __cplusplus
}
#endif

#endif
