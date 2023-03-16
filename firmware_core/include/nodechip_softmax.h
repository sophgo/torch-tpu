#ifndef NODECHIP_SOFTMAX_H_
#define NODECHIP_SOFTMAX_H_

#include "tpu_kernel.h"
#include "common.h"
#include "tpu_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_softmax_local_1x1(
    global_addr_t   bottom_global_offset,
    global_addr_t   top_global_offset,
    int             nstart,
    int             n,
    int             hslice,
    int             Tensor_C,
    int             hstride,
    int             widx,
    float           scale_val,
    bool            log,
    data_type_t     dtype);

void nodechip_softmax(
    global_addr_t   bottom_global_offset,
    global_addr_t   top_global_offset,
    const int*      shape,
    int             dims,
    int             beg_axis,
    int             end_axis,
    int             log,
    float           scale_val,
    data_type_t     dtype);

#ifdef __cplusplus
}
#endif

#endif
