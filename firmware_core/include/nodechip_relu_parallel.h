#ifndef NODECHIP_RELU_PARALLEL_H
#define NODECHIP_RELU_PARALLEL_H

#include "sg_api_struct.h"
#include "tpu_defs.h"
#include "tpu_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_relu_parallel(
    global_addr_t   global_bottom_addr,
    global_addr_t   global_top_addr,
    float           upper_limit,
    int             bottom_n,
    int             bottom_c,
    int             bottom_h,
    int             bottom_w,
    data_type_t     dtype
);

#ifdef __cplusplus
}
#endif

#endif
