#ifndef NODECHIP_ACTIVE_LOCAL_H_
#define NODECHIP_ACTIVE_LOCAL_H_

#include "tpu_kernel.h"
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_active_local(
    local_addr_t in_addr,
    local_addr_t out_addr,
    local_addr_t buffer_addr,
    const int* shape,
    data_type_t dtype,
    sg_active_type_t active_type,
    int if_local_layer,
    float* coef);

#ifdef __cplusplus
}
#endif

#endif
