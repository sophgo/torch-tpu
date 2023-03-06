#ifndef NODECHIP_CAST_H_
#define NODECHIP_CAST_H_

#include "tpu_utils.h"
#include "tpu_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_cast(
    global_addr_t   in_global_addr,
    global_addr_t   out_global_addr,
    const int*      shape,
    int             shape_dim,
    data_type_t     src_dtype,
    data_type_t     dst_dtype,
    rounding_mode_t round_mode);

#ifdef __cplusplus
}
#endif

#endif
