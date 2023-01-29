#ifndef NODECHIP_POOLING_PARALLEL_H
#define NODECHIP_POOLING_PARALLEL_H

#include "tpu_utils.h"
#include "tpu_defs.h"
#ifdef __cplusplus
extern "C" {
#endif

void nodechip_pooling_parallel_with_data_split(
    global_addr_t   ifmap_offset_global,
    global_addr_t   ofmap_offset_global,
    int             input_n,
    int             input_c,
    int             input_h,
    int             input_w,
    int             output_h,
    int             output_w,
    int             kh,
    int             kw,
    int             pad_h,
    int             pad_w,
    int             pad_h_after,
    int             pad_w_after,
    int             stride_h,
    int             stride_w,
    int             dilation_h,
    int             dilation_w,
    int             is_avg_pooling,
    int             avg_pooling_mode,
    int             if_relu,
    float           relu_upper_limit,
    data_type_t     dtype
);


/*
 * avg_pooling_mode:
 * 0 -- constant average: all position is divided by a constant
 * 1 -- average compensation: Pad value is elimated in average computation.
 */
void nodechip_pooling_parallel(
    const global_addr_t   ifmap_offset_global,
    const global_addr_t   ofmap_offset_global,
    const int             input_n,
    const int             input_c,
    const int             input_h,
    const int             input_w,
    const int             output_h,
    const int             output_w,
    const int             kh,
    const int             kw,
    const int             top_pad_h,
    const int             left_pad_w,
    int                   bottom_pad_h,
    int                   right_pad_w,
    int                   stride_h,
    int                   stride_w,
    const int             dilation_h,
    const int             dilation_w,
    const int             is_avg_pooling,
    const int             avg_pooling_mode,
    const int             if_relu,
    const float           relu_upper_limit,
    data_type_t           dtype,
    const int             c_step,
    const int             h_step,
    float                 Ratio
);

void nodechip_max_pooling_with_mask_forward(
    global_addr_t   bottom_global_offset,
    global_addr_t   top_global_offset,
    global_addr_t   top_mask_global_offset,
    int             input_n,
    int             input_c,
    int             input_h,
    int             input_w,
    int             output_h,
    int             output_w,
    int             kh,
    int             kw,
    int             pad_h,
    int             pad_w,
    int             pad_h_after,
    int             pad_w_after,
    int             stride_h,
    int             stride_w,
    int             dilation_h,
    int             dilation_w,
    int             is_avg_pooling,
    int             avg_pooling_mode,
    int             if_relu,
    float           relu_upper_limit,
    data_type_t     dtype
);

#ifdef __cplusplus
}
#endif

#endif /* NODECHIP_POOLING_H */
