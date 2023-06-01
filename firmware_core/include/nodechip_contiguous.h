#ifndef NODECHIP_STRIDED_COPY_H_
#define NODECHIP_STRIDED_COPY_H_

#include "tpu_utils.h"
#include "tpu_kernel.h"
#include "sg_api_struct.h"

#ifdef __cplusplus
extern "C" {
#endif



void nodechip_strided_copy(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    int           dim,
    const int*    shape,
    const int*    in_stride,
    const int*    out_stride,
    data_type_t   dtype);

void nodechip_contiguous(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    int           dim,
    const int*    shape,
    const int*    in_stride,
    data_type_t   dtype);

#ifdef __cplusplus
}
#endif

#endif