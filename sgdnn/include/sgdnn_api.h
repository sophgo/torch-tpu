#ifndef SGDNN_API_H
#define SGDNN_API_H

#include "bmlib_runtime.h"
#include "common.h"
#include "sg_api_struct.h"

#if defined(__cplusplus)
extern "C" {
#endif

bm_status_t sgdnn_conv_backward(
    bm_handle_t        handle,
    bm_device_mem_t    grad_output,
    bm_device_mem_t    input,
    bm_device_mem_t    weight,
    bm_device_mem_t    grad_input,
    bm_device_mem_t    grad_weight,
    bm_device_mem_t    grad_bias,
    int                n,
    int                ic,
    int                ih,
    int                iw,
    int                oc,
    int                oh,
    int                ow,
    int                groups,
    int                kh,
    int                kw,
    int                stride_h,
    int                stride_w,
    int                dh,
    int                dw,
    int                pad_ht,
    int                pad_hb,
    int                pad_wl,
    int                pad_wr,
    bool               if_relu,
    bool               input_need_grad,
    bool               weight_need_grad,
    bool               bias_need_grad,
    sg_data_type_t     dtype);

bm_status_t sgdnn_batchnorm_backward(
    bm_handle_t        handle,
    bm_device_mem_t    grad_output,
    bm_device_mem_t    input,
    bm_device_mem_t    weight,
    bm_device_mem_t    mean,
    bm_device_mem_t    invstd,
    bm_device_mem_t    grad_input,
    bm_device_mem_t    grad_weight,
    bm_device_mem_t    grad_bias,
    int                n,  
    int                c,  
    int                h,  
    int                w,
    bool               input_need_grad,
    bool               weight_need_grad,
    bool               bias_need_grad,
    sg_data_type_t     dtype);
    
#if defined(__cplusplus)
}
#endif

#endif /* SGDNN_API_H */

