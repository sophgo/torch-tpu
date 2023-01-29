#ifndef NODECHIP_CONV_FLOAT_PARALLEL_H
#define NODECHIP_CONV_FLOAT_PARALLEL_H

#include "tpu_utils.h"
#include "tpu_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_conv_float_parallel(
        global_addr_t       input_global_addr,
        global_addr_t       weight_global_addr,
        global_addr_t       bias_global_addr,
        global_addr_t       output_global_addr,
        const dim4         *ishape,
        int                 groups,
        int                 output_c,
        const dim2         *kernel,
        const dim2         *stride,
        const dim2         *dilation,
        const padding_t    *pad,
        bool                has_bias,
        bool                if_relu,
        float               upper_limit,
        bool                result_add,
        data_type_t         idtype,
        data_type_t         odtype,
        bool                reshaped_bias);

#ifdef __cplusplus
}
#endif

#endif
