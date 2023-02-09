#ifndef SGDNN_API_H
#define SGDNN_API_H

#include "bmlib_runtime.h"
#include "common_def.h"
#include "sg_api_struct.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct{
    int         dtype;
    int         ndims;
    int         shape[FW_MAX_SHAPE_DIMS];
    int         stride[FW_MAX_SHAPE_DIMS];
    int         format;
} TensorDescriptor_t;

typedef struct{
    int         op_code;
} OpTensorDescriptor_t;

typedef struct{
    int     oc;
    int     ic;
    int     kh;
    int     kw;
    sg_data_type_t  dtype;
} FilterDescriptor_t;

typedef struct{
    int     pad_h;
    int     pad_w;
    int     stride_h;//u
    int     stride_w;//v
    int     dilation_h;
    int     dilation_w;
    int     groups;
    sg_data_type_t  computeType;
} ConvolutionDescriptor_t;

bm_status_t sgdnn_conv_forward(
    bm_handle_t        handle,
    bm_device_mem_t    input,
    bm_device_mem_t    weight,
    bm_device_mem_t    bias,
    bm_device_mem_t    output,
    int                n,
    int                ic,
    int                ih,
    int                iw,
    int                oc,
    int                groups,
    int                kh,
    int                kw,
    int                stride_h,
    int                stride_w,
    int                dh,
    int                dw,
    int                pht,
    int                phb,
    int                pwl,
    int                pwr,
    bool               has_bias,
    bool               if_relu,
    float              upper_limit,
    bool               result_add,
    sg_data_type_t     idtype,
    sg_data_type_t     odtype);

bm_status_t sgdnn_conv_forward_cudnn(
    bm_handle_t                     handle,
    const void                     *alpha,
    const TensorDescriptor_t        xDesc,
    const void                     *x,
    const FilterDescriptor_t        wDesc,
    const void                     *w,
    const TensorDescriptor_t        bDesc,
    const void                     *b,
    const ConvolutionDescriptor_t   convDesc,
    //ConvlutionFwdAlgo_t             algo,
    //void                           *workspace,
    //size_t                          workSpaceSizeInBytes,
    const void                     *beta,
    const TensorDescriptor_t        yDesc,
    void                           *y);

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

bm_status_t sgdnn_conv_backward_cudnn(
    bm_handle_t                     handle,
    const void                     *alpha,
    const void                     *beta,
    const TensorDescriptor_t        xDesc,
    const void                     *x,
    void                           *dx,
    const FilterDescriptor_t        wDesc,
    const void                     *w,
    void                           *dw,
    const TensorDescriptor_t        dbDesc,
    void                           *db,
    const TensorDescriptor_t        dyDesc,
    const void                     *dy,
    const ConvolutionDescriptor_t   convDesc,
    void                           *buffer,
    bool                            dx_enable,
    bool                            dw_enable,
    bool                            db_enable);

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

bm_status_t sgdnn_pooling_forward(
    bm_handle_t         handle,
    bm_device_mem_t     input,
    bm_device_mem_t     output,
    bm_device_mem_t     max_mask,
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 output_h,
    int                 output_w,
    int                 kh,
    int                 kw,
    int                 pad_h,
    int                 pad_w,
    int                 pad_h_after,
    int                 pad_w_after,
    int                 stride_h,
    int                 stride_w,
    int                 dilation_h,
    int                 dilation_w,
    int                 is_avg_pooling,
    int                 avg_pooling_mode,
    int                 if_mask_max,
    int                 if_relu,
    float               relu_upper_limit,
    sg_data_type_t      dtype);

bm_status_t sgdnn_avgpool_backward(
    bm_handle_t        handle,
    bm_device_mem_t    grad_output,
    bm_device_mem_t    grad_input,
    int                n,
    int                c,
    int                ih,
    int                iw,
    int                oh,
    int                ow,
    int                kh,
    int                kw,
    int                stride_h,
    int                stride_w,
    int                pad_h,
    int                pad_w,
    bool               ceil_mode,
    bool               count_include_pad,
    bool               divisor_override,
    sg_data_type_t     dtype);

bm_status_t sgdnn_maxpool_backward(
    bm_handle_t        handle,
    bm_device_mem_t    forward_input,
    bm_device_mem_t    forward_output,
    bm_device_mem_t    grad_output,
    bm_device_mem_t    grad_input,
    int                n,
    int                c,
    int                ih,
    int                iw,
    int                oh,
    int                ow,
    int                kh,
    int                kw,
    int                stride_h,
    int                stride_w,
    int                pad_h,
    int                pad_w,
    int                dilation_h,
    int                dilation_w,
    bool               ceil_mode,
    sg_data_type_t     dtype);

bm_status_t sgdnn_eltwise_forward(
    bm_handle_t                handle,
    const void*                alpha1,
    const TensorDescriptor_t   aDesc,
    const void*                A,
    const void*                alpha2,
    const TensorDescriptor_t   bDesc,
    const void*                B,
    const void*                beta,
    const TensorDescriptor_t   cDesc,
    void*                      C,
    const OpTensorDescriptor_t opTensorDesc);

bm_status_t sgdnn_eltwise_backward(
    bm_handle_t                handle,
    const void*                alpha1,
    const TensorDescriptor_t   aDesc,
    const void*                input_a,
    const void*                alpha2,
    const TensorDescriptor_t   bDesc,
    const void*                input_b,
    const void*                beta,
    const TensorDescriptor_t   cDesc,
    const void*                grad_output,
    void*                      grad_input_a,
    void*                      grad_input_b,
    bool                       grad_input_a_enable,
    bool                       grad_input_b_enable,
    const OpTensorDescriptor_t opTensorDesc);

bm_status_t sgdnn_linear_backward(
    bm_handle_t        handle,
    bm_device_mem_t    input,
    bm_device_mem_t    weight,
    bm_device_mem_t    grad_output,
    bm_device_mem_t    grad_input,
    bm_device_mem_t    grad_weight,
    bm_device_mem_t    grad_bias,
    int                batch,  
    int                in_features,  
    int                out_features,  
    bool               input_need_grad,
    bool               weight_need_grad,
    bool               bias_need_grad,
    sg_data_type_t     dtype);

bm_status_t sgdnn_relu_backward(
    bm_handle_t        handle,
    bm_device_mem_t    input,
    bm_device_mem_t    grad_output,
    bm_device_mem_t    grad_input,
    int                n,
    int                c,
    int                h,
    int                w,
    bool               input_need_grad,
    sg_data_type_t     dtype);

#if defined(__cplusplus)
}
#endif

#endif /* SGDNN_API_H */

