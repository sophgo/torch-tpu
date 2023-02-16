#include "sgdnn_api.h"
#include "sgdnn_helpers.h"
#include <assert.h>
#include <memory>
#include <string.h>
#include "tpu_fp16.h"

#define ASSERT_SAME_DIMS(A, B)            \
  assert(A.ndims == B.ndims);             \

#define ASSERT_SAME_SHAPE(A, B)           \
  assert(A.ndims == B.ndims);             \
  for (int dim = 0; dim < A.ndims; dim++) \
    assert(A.shape[dim] == B.shape[dim]); \

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
    sg_data_type_t     odtype) {

    assert(handle);
    assert(idtype == SG_DTYPE_FP32 || idtype == SG_DTYPE_FP16 || idtype == SG_DTYPE_BFP16);
    assert((idtype == SG_DTYPE_FP32 && odtype == SG_DTYPE_FP32) ||
           (idtype == SG_DTYPE_BFP16 && (odtype == SG_DTYPE_BFP16 || odtype == SG_DTYPE_FP32)) ||
           (idtype == SG_DTYPE_FP16 && (odtype == SG_DTYPE_FP16 || odtype == SG_DTYPE_FP32)));
    assert(!result_add || (result_add && odtype == SG_DTYPE_FP32));
    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;
    int ih_ext = ih + pht + phb;
    int iw_ext = iw + pwr + pwl;
    int oh = (ih_ext - kh_ext) / stride_h + 1;
    int ow = (iw_ext - kw_ext) / stride_w + 1;
    u64 isz = (u64)n * ic * ih * iw * (idtype == SG_DTYPE_FP32 ? 4 : 2);
    u64 wsz = (u64)oc * kh * kw * (idtype == SG_DTYPE_FP32 ? 4 * ic / groups : 2 * ALIGN(ic / groups, 32));
    u64 osz = (u64)n * oc * oh * ow * (odtype == SG_DTYPE_FP32 ? 4 : 2);
    bm_device_mem_t imem, wmem, bmem, omem;
    DEVICE_MEM_NEW_INPUT(handle, input, isz, imem);
    DEVICE_MEM_NEW_INPUT(handle, weight, wsz, wmem);
    if (has_bias)
        DEVICE_MEM_NEW_INPUT(handle, bias, oc * sizeof(float), bmem);
    if (result_add)
        DEVICE_MEM_NEW_INPUT(handle, output, osz, omem);
    else
        DEVICE_MEM_NEW_OUTPUT(handle, output, osz, omem);

    sg_api_conv_forward_t api = {
        bm_mem_get_device_addr(imem),
        bm_mem_get_device_addr(wmem),
        bm_mem_get_device_addr(bmem),
        bm_mem_get_device_addr(omem),
        {n, ic, ih, iw},
        groups,
        oc,
        {kh, kw},
        {stride_h, stride_w},
        {dh, dw},
        {pht, phb, pwl, pwr},
        has_bias,
        if_relu,
        upper_limit,
        result_add,
        idtype,
        odtype
    };

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_conv_forward", &api, sizeof(api));

    DEVICE_MEM_DEL_OUTPUT(handle, output, omem);
    DEVICE_MEM_DEL_INPUT(handle, input, imem);
    DEVICE_MEM_DEL_INPUT(handle, weight, wmem);
    if (has_bias)
        DEVICE_MEM_DEL_INPUT(handle, bias, bmem);
    return BM_SUCCESS;
}

//dstValue = alpha[0]*result + beta[0]*priorDstValue
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
    void                           *y) {

    assert(xDesc.ndims == 4 && yDesc.ndims == 4);
    int n = xDesc.shape[0];
    int ic = xDesc.shape[1];
    int ih = xDesc.shape[2];
    int iw = xDesc.shape[3];

    int oc = wDesc.oc;
    assert(xDesc.shape[1] == wDesc.ic);
    int kh = wDesc.kh;
    int kw = wDesc.kw;

    int pad_h = convDesc.pad_h;
    int pad_w = convDesc.pad_w;
    int stride_h = convDesc.stride_h;
    int stride_w = convDesc.stride_w;
    int dh = convDesc.dilation_h;
    int dw = convDesc.dilation_w;

    int groups = convDesc.groups;

    float alpha_ = ((float*)alpha)[0];
    assert(alpha_ == 1.0f);
    float beta_ = ((float*)beta)[0];
    assert(beta_ == 0.0f || beta_ == 1.0f);
    bool result_add = beta_ == 1.0f;

    sg_data_type_t idtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t odtype = (sg_data_type_t)(yDesc.dtype);
    sg_api_conv_forward_t api = {
        (unsigned long long)x,
        (unsigned long long)w,
        (unsigned long long)b,
        (unsigned long long)y,
        {n, ic, ih, iw},
        groups,
        oc,
        {kh, kw},
        {stride_h, stride_w},
        {dh, dw},
        {pad_h, pad_h, pad_w, pad_w},//pad
        b != NULL ? 1 : 0,//has_bias?
        0,//if_relu
        0,//upper_limit
        result_add ? 1 : 0,
        idtype,
        odtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_conv_forward", &api, sizeof(api));

    return BM_SUCCESS;
}

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
    sg_data_type_t     dtype
  ) {

    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;
    };

    bm_device_mem_t grad_output_mem;
    bm_device_mem_t input_mem, weight_mem, buffer_mem;
    bm_device_mem_t grad_input_mem, grad_weight_mem, grad_bias_mem;
    u64 grad_output_size = (u64)n * oc * oh * ow * dtype_size(dtype);
    u64 input_size = (u64)n * ic * ih * iw * dtype_size(dtype);
    u64 weight_size = (u64)oc * (dtype == SG_DTYPE_FP32 ? ic : ALIGN(ic, 32)) * kh * kw * dtype_size(dtype);
    u64 grad_input_size = input_size;
    //grad_weight arrange as [ic, oc, kh, kw]
    u64 grad_weight_size = (u64)ic * oc * kh * kw * dtype_size(dtype);
    u64 grad_bias_size = (u64)oc * dtype_size(dtype);

    // cal buffer size
    u64 weight_reorder_size = (dtype == SG_DTYPE_FP32 ? oc : ALIGN(oc, 32)) * kh * kw * ic * dtype_size(dtype);
    u64 grad_out_reorder_size = (dtype == SG_DTYPE_FP32 ? n : ALIGN(n, 32)) * oh * ow * oc * dtype_size(dtype);
    u64 buffer_size = dtype == SG_DTYPE_FP32 ? 0 : sg_max(weight_reorder_size, grad_out_reorder_size);//use for weight reorder

    DEVICE_MEM_NEW_INPUT(handle, grad_output, grad_output_size, grad_output_mem);
    DEVICE_MEM_NEW_INPUT(handle, input, input_size, input_mem);
    DEVICE_MEM_NEW_INPUT(handle, weight, weight_size, weight_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input, grad_input_size, grad_input_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_weight, grad_weight_size, grad_weight_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_bias, grad_bias_size, grad_bias_mem);
    DEVICE_MEM_NEW_BUFFER(handle, buffer_mem, buffer_size);

    sg_api_conv_backward_t api = {
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(weight_mem),
        bm_mem_get_device_addr(grad_output_mem),
        bm_mem_get_device_addr(grad_input_mem),
        bm_mem_get_device_addr(grad_weight_mem),
        bm_mem_get_device_addr(grad_bias_mem),
        bm_mem_get_device_addr(buffer_mem),
        {n, ic, ih, iw},
        {n, oc, oh, ow},
        groups,
        {kh, kw},
        {stride_h, stride_w},
        {dh, dw},
        {pad_ht, pad_hb, pad_wl, pad_wr},
        1, 1, 1,
        dtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_conv_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_BUFFER(handle, buffer_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_weight, grad_weight_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_bias, grad_bias_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, weight, weight_mem);

    return BM_SUCCESS;
}

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
    bool                            dx_enable,
    bool                            dw_enable,
    bool                            db_enable
 ) {

    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;
    };

    assert(xDesc.ndims == 4 && dyDesc.ndims == 4);
    int n = xDesc.shape[0];
    int ic = xDesc.shape[1];
    int ih = xDesc.shape[2];
    int iw = xDesc.shape[3];

    int oh = dyDesc.shape[2];
    int ow = dyDesc.shape[3];

    int oc = wDesc.oc;
    assert(xDesc.shape[1] == wDesc.ic);
    int kh = wDesc.kh;
    int kw = wDesc.kw;

    int pad_h = convDesc.pad_h;
    int pad_w = convDesc.pad_w;
    int stride_h = convDesc.stride_h;
    int stride_w = convDesc.stride_w;
    int dilation_h = convDesc.dilation_h;
    int dilation_w = convDesc.dilation_w;

    int groups = convDesc.groups;
    assert(groups == 1);

    sg_data_type_t idtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t wdtype = (sg_data_type_t)(wDesc.dtype);
    sg_data_type_t odtype = (sg_data_type_t)(dyDesc.dtype);
    assert(idtype == wdtype && wdtype == odtype);

    // cal buffer size
    u64 weight_reorder_size = (wdtype == SG_DTYPE_FP32 ? oc : ALIGN(oc, 32)) * kh * kw * ic * dtype_size(wdtype);
    u64 grad_out_reorder_size = (odtype == SG_DTYPE_FP32 ? n : ALIGN(n, 32)) * oh * ow * oc * dtype_size(odtype);
    u64 buffer_size = idtype == SG_DTYPE_FP32 ? 0 : sg_max(weight_reorder_size, grad_out_reorder_size);//use for weight reorder

    bm_device_mem_t buffer_mem;
    if (buffer_size > 0) {
        bm_status_t status = bm_malloc_device_byte(handle, &buffer_mem, buffer_size);
        assert(status == BM_SUCCESS);
    }

    sg_api_conv_backward_t api = {
        (unsigned long long)x,
        (unsigned long long)w,
        (unsigned long long)dy,
        (unsigned long long)dx,
        (unsigned long long)dw,
        (unsigned long long)db,
        bm_mem_get_device_addr(buffer_mem),
        {n, ic, ih, iw},//ishape
        {n, oc, oh, ow},//oshape
        groups,
        {kh, kw},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {pad_h, pad_h, pad_w, pad_w},
        dx_enable,
        dw_enable,
        db_enable,
        idtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_conv_backward", &api, sizeof(api));

    if (buffer_size > 0) {
        bm_free_device(handle, buffer_mem);
    }
    return BM_SUCCESS;
}

bm_status_t sgdnn_batchnorm_forward(
    bm_handle_t        handle,
    bm_device_mem_t    input,
    bm_device_mem_t    running_mean,
    bm_device_mem_t    running_var,
    bm_device_mem_t    weight,
    bm_device_mem_t    bias,
    bm_device_mem_t    updated_mean,
    bm_device_mem_t    updated_var,
    bm_device_mem_t    batch_mean,
    bm_device_mem_t    batch_invstd,
    bm_device_mem_t    output,
    int                n,
    int                c,
    int                h,
    int                w,
    float              momentum,
    float              eps,
    sg_data_type_t     dtype)
{
    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;};

    bm_device_mem_t input_mem, running_mean_mem, running_var_mem, weight_mem, bias_mem;
    bm_device_mem_t updated_mean_mem, updated_var_mem, batch_mean_mem, batch_invstd_mem, output_mem;
    u64 param_size = (u64)n * c * h * w * dtype_size(dtype);
    u64 c_param_size = (u64)c * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, input, param_size, input_mem);
    DEVICE_MEM_NEW_INPUT(handle, running_mean, c_param_size, running_mean_mem);
    DEVICE_MEM_NEW_INPUT(handle, running_var, c_param_size, running_var_mem);
    DEVICE_MEM_NEW_INPUT(handle, weight, c_param_size, weight_mem);
    DEVICE_MEM_NEW_INPUT(handle, bias, c_param_size, bias_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, updated_mean, c_param_size, updated_mean_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, updated_var, c_param_size, updated_var_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, batch_mean, c_param_size, batch_mean_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, batch_invstd, c_param_size, batch_invstd_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, output, param_size, output_mem);

    sg_api_batchnorm_forward_t api = {
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(running_mean_mem),
        bm_mem_get_device_addr(running_var_mem),
        bm_mem_get_device_addr(weight_mem),
        bm_mem_get_device_addr(bias_mem),
        bm_mem_get_device_addr(updated_mean_mem),
        bm_mem_get_device_addr(updated_var_mem),
        bm_mem_get_device_addr(batch_mean_mem),
        bm_mem_get_device_addr(batch_invstd_mem),
        bm_mem_get_device_addr(output_mem),
        {n, c, h, w},
        momentum,
        eps,
        dtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_batchnorm_forward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, running_mean, running_mean_mem);
    DEVICE_MEM_DEL_INPUT(handle, running_var, running_var_mem);
    DEVICE_MEM_DEL_INPUT(handle, weight, weight_mem);
    DEVICE_MEM_DEL_INPUT(handle, bias, bias_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, updated_mean, updated_mean_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, updated_var, updated_var_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, batch_mean, batch_mean_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, batch_invstd, batch_invstd_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, output, output_mem); 
    return BM_SUCCESS;
}

bm_status_t sgdnn_batchnorm_forward_cudnn(
    bm_handle_t                      handle,
    BatchNormMode                    mode,
    const void                      *alpha,
    const void                      *beta,
    const TensorDescriptor_t         xDesc,
    const void                      *x,
    const TensorDescriptor_t         yDesc,
    void                            *y,
    const TensorDescriptor_t         bnScaleBiasMeanVarDesc,
    const void                      *bnScale,
    const void                      *bnBias,
    double                           exponentialAverageFactor,
    void                            *resultRunningMean,
    void                            *resultRunningVariance,
    double                           epsilon,
    void                            *resultSaveMean,
    void                            *resultSaveInvVariance)
{
    unsigned long long input        = (unsigned long long)x;
    unsigned long long weight       = (unsigned long long)bnScale;
    unsigned long long bias         = (unsigned long long)bnBias;
    unsigned long long running_mean = (unsigned long long)resultRunningMean;
    unsigned long long running_var  = (unsigned long long)resultRunningVariance;
    unsigned long long batch_mean   = (unsigned long long)resultSaveMean;
    unsigned long long batch_invstd = (unsigned long long)resultSaveInvVariance;
    unsigned long long output       = (unsigned long long)y;

    assert(mode == BatchNorm_Spatial);
    float alpha_ = ((float*)alpha)[0];
    assert(alpha_ == 1.0f);
    float beta_ = ((float*)beta)[0];
    assert(beta_ == 0.0f || beta_ == 1.0f);

    assert(xDesc.ndims == 4 && yDesc.ndims == 4);
    assert(bnScaleBiasMeanVarDesc.ndims ==1 );
    int n = xDesc.shape[0];
    int c = xDesc.shape[1];
    int h = xDesc.shape[2];
    int w = xDesc.shape[3];
    
    float momentum = exponentialAverageFactor;
    float eps = epsilon;
    
    sg_data_type_t idtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t odtype = (sg_data_type_t)(yDesc.dtype);
    sg_data_type_t wdtype = (sg_data_type_t)(bnScaleBiasMeanVarDesc.dtype);
    assert(idtype == wdtype && wdtype == odtype);

    sg_api_batchnorm_forward_t api = {
        input,
        running_mean,
        running_var,
        weight,
        bias,
        running_mean,
        running_var,
        batch_mean,
        batch_invstd,
        output,
        {n, c, h, w},
        momentum,
        eps,
        idtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_batchnorm_forward", &api, sizeof(api));
    return BM_SUCCESS;
}

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
    sg_data_type_t     dtype)
{
    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;};

    bm_device_mem_t grad_output_mem, input_mem, weight_mem, mean_mem, invstd_mem;
    bm_device_mem_t grad_input_mem, grad_weight_mem, grad_bias_mem;
    u64 param_size = (u64)n * c * h * w * dtype_size(dtype);
    u64 c_param_size = (u64)c * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, grad_output, param_size, grad_output_mem);
    DEVICE_MEM_NEW_INPUT(handle, input, param_size, input_mem);
    DEVICE_MEM_NEW_INPUT(handle, weight, c_param_size, weight_mem);
    DEVICE_MEM_NEW_INPUT(handle, mean, c_param_size, mean_mem);
    DEVICE_MEM_NEW_INPUT(handle, invstd, c_param_size, invstd_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input, param_size, grad_input_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_weight, c_param_size, grad_weight_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_bias, c_param_size, grad_bias_mem);

    sg_api_batchnorm_backward_t api = {
        bm_mem_get_device_addr(grad_output_mem),
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(weight_mem),
        bm_mem_get_device_addr(mean_mem),
        bm_mem_get_device_addr(invstd_mem),
        bm_mem_get_device_addr(grad_input_mem),
        bm_mem_get_device_addr(grad_weight_mem),
        bm_mem_get_device_addr(grad_bias_mem),
        {n, c, h, w},
        1, 1, 1
    };

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_batchnorm_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, weight, weight_mem);
    DEVICE_MEM_DEL_INPUT(handle, mean, mean_mem);
    DEVICE_MEM_DEL_INPUT(handle, invstd, invstd_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_weight, grad_weight_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_bias, grad_bias_mem);

    return BM_SUCCESS;
}

bm_status_t sgdnn_batchnorm_backward_cudnn(
    bm_handle_t                      handle,
    BatchNormMode                    mode,
    const void                      *alphaDataDiff,
    const void                      *betaDataDiff,
    const void                      *alphaParamDiff,
    const void                      *betaParamDiff,
    const TensorDescriptor_t         xDesc,
    const void                      *x,
    const TensorDescriptor_t         dyDesc,
    const void                      *dy,
    const TensorDescriptor_t         dxDesc,
    void                            *dx,
    const TensorDescriptor_t         bnScaleBiasDiffDesc,
    const void                      *bnScale,
    void                            *resultBnScaleDiff,
    void                            *resultBnBiasDiff,
    double                           epsilon,
    const void                      *savedMean,
    const void                      *savedInvVariance,
    bool                             dx_enable,
    bool                             dw_enable,
    bool                             db_enable)
{
    unsigned long long grad_output   = (unsigned long long)dy;
    unsigned long long input         = (unsigned long long)x;
    unsigned long long weight        = (unsigned long long)bnScale;
    unsigned long long saved_mean    = (unsigned long long)savedMean;
    unsigned long long saved_invstd  = (unsigned long long)savedInvVariance;
    unsigned long long grad_input    = (unsigned long long)dx;
    unsigned long long grad_weight   = (unsigned long long)resultBnScaleDiff;
    unsigned long long grad_bias     = (unsigned long long)resultBnBiasDiff;

    assert(mode == BatchNorm_Spatial);
    
    float alpha_data = ((float*)alphaDataDiff)[0];
    assert(alpha_data == 1.0f);
    float alpha_param = ((float*)alphaParamDiff)[0];
    assert(alpha_param == 1.0f);
    float beta_data = ((float*)betaDataDiff)[0];
    assert(beta_data == 0.0f);
    float beta_param = ((float*)betaParamDiff)[0];
    assert(beta_param == 0.0f);
    
    assert(dyDesc.ndims == 4);
    assert( xDesc.ndims == 4);
    assert(dxDesc.ndims == 4);

    int n = xDesc.shape[0];
    int c = xDesc.shape[1];
    int h = xDesc.shape[2];
    int w = xDesc.shape[3];

    assert(bnScaleBiasDiffDesc.ndims == 1);
    
    sg_data_type_t dydtype = (sg_data_type_t)(dyDesc.dtype);
    sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t dxdtype = (sg_data_type_t)(dxDesc.dtype);
    sg_data_type_t wdtype = (sg_data_type_t)(bnScaleBiasDiffDesc.dtype);
    
    assert(dydtype == 0);
    assert(xdtype == 0);
    assert(dxdtype == 0);
    assert(wdtype == 0);

    sg_api_batchnorm_backward_t api = {
        grad_output,
        input,
        weight,
        saved_mean,
        saved_invstd,
        grad_input,
        grad_weight,
        grad_bias,
        {n, c, h, w},
        dx_enable,
        dw_enable,
        db_enable};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_batchnorm_backward", &api, sizeof(api));
    return BM_SUCCESS;
}

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
    sg_data_type_t      dtype) {

    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;
    };

    if (output_h == 0 || output_w == 0) {
        output_h = (input_h + pad_h + pad_h_after - kh) / stride_h + 1;
        output_w = (input_w + pad_w + pad_w_after - kw) / stride_w + 1;
        if ((input_h + pad_h + pad_h_after - kh) % stride_h != 0 &&
                output_h * stride_h < input_h + pad_h)
            output_h++;

        if ((input_w + pad_w + pad_w_after - kw) % stride_w != 0 &&
                output_w * stride_w < input_w + pad_w)
            output_w++;

        if (input_h + 2 * pad_h - kh < 0)  output_h = 1;
        if (input_w + 2 * pad_w - kw < 0)  output_w = 1;
    }

    bm_device_mem_t input_mem, output_mem, max_mask_mem;
    u64 input_size = (u64)input_n * input_c * input_h * input_w * dtype_size(dtype);
    u64 output_size = (u64)input_n * input_c * output_h * output_w * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, input, input_size, input_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, output, output_size, output_mem);
    if (if_mask_max == 1) {
        DEVICE_MEM_NEW_OUTPUT(handle, max_mask, output_size, max_mask_mem);
    }

    sg_api_pooling_forward_t api = {
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(output_mem),
        bm_mem_get_device_addr(max_mask_mem),
        input_n,
        input_c,
        input_h,
        input_w,
        output_h,
        output_w,
        kh,
        kw,
        pad_h,
        pad_w,
        pad_h_after,
        pad_w_after,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        is_avg_pooling,
        avg_pooling_mode,
        if_mask_max,
        if_relu,
        relu_upper_limit,
        dtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_pooling_forward", &api, sizeof(api));

    DEVICE_MEM_DEL_OUTPUT(handle, output, output_mem);
    if (if_mask_max == 1) {
        DEVICE_MEM_DEL_OUTPUT(handle, max_mask, max_mask_mem);
    }
    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    return BM_SUCCESS;
}

bm_status_t sgdnn_pooling_forward_cudnn(
    bm_handle_t                 handle,
    const PoolingDescriptor_t   poolingDesc,
    const void                 *alpha,
    const TensorDescriptor_t    xDesc,
    const void                 *x,
    const void                 *beta,
    const TensorDescriptor_t    yDesc,
    void                       *y
 ) {

    assert(xDesc.ndims == 4 && yDesc.ndims == 4);
    int n = xDesc.shape[0];
    int c = xDesc.shape[1];
    int ih = xDesc.shape[2];
    int iw = xDesc.shape[3];
    int oh = yDesc.shape[2];
    int ow = yDesc.shape[3];

    int pooling_mode = (PoolingMode_t)poolingDesc.mode;
    int kh = poolingDesc.kh;
    int kw = poolingDesc.kw;
    int pad_h = poolingDesc.pad_h;
    int pad_w = poolingDesc.pad_w;
    int stride_h = poolingDesc.stride_h;
    int stride_w = poolingDesc.stride_w;

    sg_data_type_t idtype = (sg_data_type_t)xDesc.dtype;
    sg_data_type_t odtype = (sg_data_type_t)yDesc.dtype;
    assert(idtype == odtype);

    sg_api_pooling_forward_t api = {
        (unsigned long long)x,
        (unsigned long long)y,
        0,//max_mask
        n, c, ih, iw,
        oh, ow,
        kh, kw,
        pad_h, pad_w, pad_h, pad_w,
        stride_h, stride_w,
        1, 1,//dilation_h && dilation_w
        pooling_mode == Pooling_AVERAGE,
        0,//avgpool_mode
        0,//if_mask_max
        0,//if_relu
        0,//relu_upper_limit
        idtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_pooling_forward", &api, sizeof(api));

    return BM_SUCCESS;
}

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
    int                divisor_override,
    sg_data_type_t     dtype
  ) {

    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;};

    bm_device_mem_t grad_output_mem, grad_input_mem;
    u64 grad_output_size = (u64)n * c * oh * ow * dtype_size(dtype);
    u64 grad_input_size = (u64)n * c * ih * iw * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, grad_output, grad_output_size, grad_output_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input, grad_input_size, grad_input_mem);

    sg_api_avgpool_backward_t api = {
        bm_mem_get_device_addr(grad_output_mem),
        bm_mem_get_device_addr(grad_input_mem),
        {n, c, ih, iw},
        {n, c, oh, ow},
        {kh, kw},
        {stride_h, stride_w},
        {pad_h, pad_w},
        ceil_mode == true ? 1 : 0,
        count_include_pad == true ? 1 : 0,
        divisor_override,
        dtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_avgpool_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    return BM_SUCCESS;
}

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
    sg_data_type_t     dtype
  ) {

    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;
    };

    bm_device_mem_t forward_input_mem, forward_output_mem;
    bm_device_mem_t grad_input_mem, grad_output_mem;
    u64 grad_output_size = (u64)n * c * oh * ow * dtype_size(dtype);
    u64 grad_input_size = (u64)n * c * ih * iw * dtype_size(dtype);
    u64 forward_input_size = grad_input_size;
    u64 forward_output_size = grad_output_size;

    DEVICE_MEM_NEW_INPUT(handle, forward_input, forward_input_size, forward_input_mem);
    DEVICE_MEM_NEW_INPUT(handle, forward_output, forward_output_size, forward_output_mem);
    DEVICE_MEM_NEW_INPUT(handle, grad_output, grad_output_size, grad_output_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input, grad_input_size, grad_input_mem);

    sg_api_maxpool_backward_t api = {
        bm_mem_get_device_addr(forward_input_mem),
        bm_mem_get_device_addr(forward_output_mem),
        bm_mem_get_device_addr(grad_output_mem),
        bm_mem_get_device_addr(grad_input_mem),
        {n, c, ih, iw},
        {n, c, oh, ow},
        {kh, kw},
        {stride_h, stride_w},
        {pad_h, pad_w},
        {dilation_h, dilation_w},
        ceil_mode == true ? 1 : 0,
        dtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_maxpool_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);
    DEVICE_MEM_DEL_INPUT(handle, forward_input, forward_input_mem);
    DEVICE_MEM_DEL_INPUT(handle, forward_output, forward_output_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    return BM_SUCCESS;
}

bm_status_t sgdnn_pooling_backward_cudnn(
    bm_handle_t                 handle,
    const PoolingDescriptor_t   poolingDesc,
    const void                 *alpha,
    const void                 *beta,
    const TensorDescriptor_t    yDesc,
    const void                 *y,
    const TensorDescriptor_t    dyDesc,
    const void                 *dy,
    const TensorDescriptor_t    xDesc,
    const void                 *x,
    const TensorDescriptor_t    dxDesc,
    void                       *dx
 ) {

    ASSERT_SAME_SHAPE(xDesc, dxDesc);
    ASSERT_SAME_SHAPE(yDesc, dyDesc);
    assert(xDesc.ndims == 4 && yDesc.ndims == 4);

    int n = xDesc.shape[0];
    int c = xDesc.shape[1];
    int ih = xDesc.shape[2];
    int iw = xDesc.shape[3];
    int oh = yDesc.shape[2];
    int ow = yDesc.shape[3];

    int pooling_mode = (PoolingMode_t)poolingDesc.mode;
    int kh = poolingDesc.kh;
    int kw = poolingDesc.kw;
    int pad_h = poolingDesc.pad_h;
    int pad_w = poolingDesc.pad_w;
    int stride_h = poolingDesc.stride_h;
    int stride_w = poolingDesc.stride_w;

    assert(xDesc.dtype == yDesc.dtype);
    assert(xDesc.dtype == dxDesc.dtype);
    assert(yDesc.dtype == dyDesc.dtype);
    sg_data_type_t dtype = (sg_data_type_t)(xDesc.dtype);

    if (pooling_mode == Pooling_MAX) {
        sg_api_maxpool_backward_t api = {
            (unsigned long long)x,
            (unsigned long long)y,
            (unsigned long long)dy,
            (unsigned long long)dx,
            {n, c, ih, iw},
            {n, c, oh, ow},
            {kh, kw},
            {stride_h, stride_w},
            {pad_h, pad_w},
            {1, 1},//{dilation_h, dilation_w},
            0,//ceil_mode
            dtype};

        tpu_kernel_launch_sync(handle, "tpu_kernel_api_maxpool_backward", &api, sizeof(api));
    } else if (pooling_mode == Pooling_AVERAGE) {
        sg_api_avgpool_backward_t api = {
            (unsigned long long)dy,
            (unsigned long long)dx,
            {n, c, ih, iw},
            {n, c, oh, ow},
            {kh, kw},
            {stride_h, stride_w},
            {pad_h, pad_w},
            0,//ceil_mode
            1,//count_include_pad
            kh * kw,//divisor_override
            dtype};

        tpu_kernel_launch_sync(handle, "tpu_kernel_api_avgpool_backward", &api, sizeof(api));
    }

    return BM_SUCCESS;
}

// C = op(alpha1[0] * A, alpha2[0] * B) + beta[0] * C
bm_status_t sgdnn_eltwise_forward(
    bm_handle_t                 handle,
    const void*                 alpha1,
    const TensorDescriptor_t    aDesc,
    const void*                 A,
    const void*                 alpha2,
    const TensorDescriptor_t    bDesc,
    const void*                 B,
    const void*                 beta,
    const TensorDescriptor_t    cDesc,
    void*                       C,
    const OpTensorDescriptor_t  opTensorDesc
) {

    int op_code = opTensorDesc.op_code;

    assert(aDesc.ndims == 4 && bDesc.ndims == 4 && cDesc.ndims == 4);
    int A_n = aDesc.shape[0];
    int A_c = aDesc.shape[1];
    int A_h = aDesc.shape[2];
    int A_w = aDesc.shape[3];

    int B_n = bDesc.shape[0];
    int B_c = bDesc.shape[1];
    int B_h = bDesc.shape[2];
    int B_w = bDesc.shape[3];

    int C_n = cDesc.shape[0];
    int C_c = cDesc.shape[1];
    int C_h = cDesc.shape[2];
    int C_w = cDesc.shape[3];

    assert(A_n == B_n && B_n == C_n);
    assert(A_c == B_c && B_c == C_c);
    assert(A_h == B_h && B_h == C_h);
    assert(A_w == B_w && B_w == C_w);

    DataUnion alpha_A, alpha_B;
    alpha_A.f32val = ((float*)alpha1)[0];
    alpha_B.f32val = ((float*)alpha2)[0];
    assert(((float*)beta)[0] == 0.0f);

    sg_data_type_t dtype_A = (sg_data_type_t)(aDesc.dtype);
    sg_data_type_t dtype_B = (sg_data_type_t)(bDesc.dtype);
    assert(dtype_A == dtype_B);

    sg_data_type_t dtype_C = (sg_data_type_t)(cDesc.dtype);

    sg_api_eltwise_forward_t api = {
        (unsigned long long)A,
        (unsigned long long)B,
        (unsigned long long)C,
        0,// mask_global_addr
        2,// input number
        A_n, A_c, A_h, A_w,
        op_code,
        alpha_A.i32val,
        alpha_B.i32val,
        0, 0, 0,
        0,
        dtype_A,
        dtype_C};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_eltwise_forward", &api, sizeof(api));

    return BM_SUCCESS;
}

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
    const OpTensorDescriptor_t opTensorDesc
 ) {

    int op_code = opTensorDesc.op_code;

    assert(aDesc.ndims == 4 && bDesc.ndims == 4 && cDesc.ndims == 4);
    int A_n = aDesc.shape[0];
    int A_c = aDesc.shape[1];
    int A_h = aDesc.shape[2];
    int A_w = aDesc.shape[3];

    int B_n = bDesc.shape[0];
    int B_c = bDesc.shape[1];
    int B_h = bDesc.shape[2];
    int B_w = bDesc.shape[3];

    int C_n = cDesc.shape[0];
    int C_c = cDesc.shape[1];
    int C_h = cDesc.shape[2];
    int C_w = cDesc.shape[3];

    assert(A_n == B_n && B_n == C_n);
    assert(A_c == B_c && B_c == C_c);
    assert(A_h == B_h && B_h == C_h);
    assert(A_w == B_w && B_w == C_w);

    int alpha_A = ((int*)alpha1)[0];
    int alpha_B = ((int*)alpha2)[0];
    int beta_C = ((int*)beta)[0];
    assert(beta_C == 0);

    sg_data_type_t dtype_A = (sg_data_type_t)(aDesc.dtype);
    sg_data_type_t dtype_B = (sg_data_type_t)(bDesc.dtype);
    assert(dtype_A == dtype_B);

    sg_data_type_t dtype_C = (sg_data_type_t)(cDesc.dtype);

    sg_api_eltwise_backward_t api = {
        (unsigned long long)input_a,
        (unsigned long long)input_b,
        (unsigned long long)grad_output,
        (unsigned long long)grad_input_a,
        (unsigned long long)grad_input_b,
        {A_n, A_c, A_h, A_w},
        op_code,
        alpha_A,
        alpha_B,
        grad_input_a_enable,
        grad_input_b_enable,
        dtype_A,
        dtype_C};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_eltwise_backward", &api, sizeof(api));

    return BM_SUCCESS;
}

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
    sg_data_type_t     dtype)
{
    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;};

    bm_device_mem_t grad_output_mem, input_mem, weight_mem;
    bm_device_mem_t grad_input_mem, grad_weight_mem, grad_bias_mem;
    u64 input_size = (u64)batch * in_features * dtype_size(dtype);
    u64 weight_size = (u64)in_features * out_features * dtype_size(dtype);
    u64 grad_output_size = (u64)batch * out_features * dtype_size(dtype);
    u64 bias_size = (u64)out_features * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, input, input_size, input_mem);
    DEVICE_MEM_NEW_INPUT(handle, weight, weight_size, weight_mem);
    DEVICE_MEM_NEW_INPUT(handle, grad_output, grad_output_size, grad_output_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input, input_size, grad_input_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_weight, weight_size, grad_weight_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_bias, bias_size, grad_bias_mem);

    sg_api_linear_backward_t api = {
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(weight_mem),
        bm_mem_get_device_addr(grad_output_mem),
        bm_mem_get_device_addr(grad_input_mem),
        bm_mem_get_device_addr(grad_weight_mem),
        bm_mem_get_device_addr(grad_bias_mem),
        batch,
        {in_features, out_features},
        1, 1, 1};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_linear_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, weight, weight_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_weight, grad_weight_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_bias, grad_bias_mem);

    return BM_SUCCESS;
}

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
    sg_data_type_t     dtype)
{
    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;};

    bm_device_mem_t grad_output_mem, input_mem;
    bm_device_mem_t grad_input_mem;
    u64 size = (u64)n * c * h * w * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, input, size, input_mem);
    DEVICE_MEM_NEW_INPUT(handle, grad_output, size, grad_output_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input, size, grad_input_mem);

    sg_api_relu_backward_t api = {
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(grad_output_mem),
        bm_mem_get_device_addr(grad_input_mem),
        {n, c, h, w},
        1};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_relu_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);

    return BM_SUCCESS;
}

bm_status_t sgdnn_cross_entropy_backward(
    bm_handle_t        handle,
    bm_device_mem_t    input,
    bm_device_mem_t    target,
    bm_device_mem_t    grad_input,
    int                batch,
    int                cls_num,
    int                reduction,
    sg_data_type_t     dtype)
{
    auto dtype_size = [](int dtype) {
        int size = 1;
        if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
        else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                 dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
            size = 2;
        else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
            size = 4;
        return size;};

    bm_device_mem_t input_mem, target_mem;
    bm_device_mem_t grad_input_mem;
    u64 size = (u64)batch * cls_num * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, input, size, input_mem);
    DEVICE_MEM_NEW_INPUT(handle, target, size, target_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input, size, grad_input_mem);

    sg_api_crossentropy_backward_t api = {
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(target_mem),
        bm_mem_get_device_addr(grad_input_mem),
        batch, cls_num, reduction, dtype};

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_cross_entropy_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, target, target_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);

    return BM_SUCCESS;
}

#ifdef ENABLE_PYBIND
// pybind11 register c++ half-precision floating point as numpy.float16
// https://github.com/pybind/pybind11/issues/1776

//#include <pybind11/embed.h>
//#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pybind11_numpy_scalar.h"

namespace py = pybind11;

typedef fp16 float16;

namespace pybind11 { namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

// Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <>
struct type_caster<float16> : npy_scalar_caster<float16> {
  static constexpr auto name = _("float16");
};

}}  // namespace pybind11::detail

PYBIND11_MODULE(sgdnn_pybind, m)
{
    m.doc() = "pybind11 sgdnn backward plugin";
    m.def("conv_backward", [](py::array_t<float16> grad_output,
                              py::array_t<float16> input,
                              py::array_t<float16> weight,
                              py::array_t<float16> grad_input,
                              py::array_t<float16> grad_weight,
                              py::array_t<float16> grad_bias,
                              int n, int ic, int ih, int iw,
                              int oc, int oh, int ow, int groups,
                              int kh, int kw,
                              int stride_h, int stride_w,
                              int dh, int dw,
                              int pad_ht, int pad_hb,
                              int pad_wl, int pad_wr,
                              bool if_relu,
                              bool input_grad_enable,
                              bool weight_grad_enable,
                              bool bias_grad_enable) {
        py::buffer_info grad_out_buf = grad_output.request();
        float16 *grad_out_fp16 = (float16 *)grad_out_buf.ptr;
        py::buffer_info input_buf = input.request();
        float16 *input_fp16 = (float16 *)input_buf.ptr;
        py::buffer_info weight_buf = weight.request();
        float16 *weight_fp16 = (float16 *)weight_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float16 *grad_input_fp16 = (float16 *)grad_input_buf.ptr;
        py::buffer_info grad_weight_buf = grad_weight.request();
        float16 *grad_weight_fp16 = (float16 *)grad_weight_buf.ptr;
        py::buffer_info grad_bias_buf = grad_bias.request();
        float16 *grad_bias_fp16 = (float16 *)grad_bias_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        bm_status_t status = sgdnn_conv_backward(handle,
                            bm_mem_from_system(grad_out_fp16),
                            bm_mem_from_system(input_fp16),
                            bm_mem_from_system(weight_fp16),
                            bm_mem_from_system(grad_input_fp16),
                            bm_mem_from_system(grad_weight_fp16),
                            bm_mem_from_system(grad_bias_fp16),
                            n, ic, ih, iw, oc, oh, ow,
                            groups, kh, kw,
                            stride_h, stride_w, dh, dw,
                            pad_ht, pad_hb, pad_wl, pad_wr,
                            if_relu,
                            input_grad_enable,
                            weight_grad_enable,
                            bias_grad_enable,
                            (sg_data_type_t)1);

        UNUSED(status);
        //assert(status == BM_SUCCESS);
        bm_dev_free(handle);
    });


    m.def("batchnorm_forward", [](py::array_t<float> input,
                                    py::array_t<float> running_mean,
                                    py::array_t<float> running_var,
                                    py::array_t<float> weight,
                                    py::array_t<float> bias,
                                    py::array_t<float> updated_mean,
                                    py::array_t<float> updated_var,
                                    py::array_t<float> batch_mean,
                                    py::array_t<float> batch_invstd,
                                    py::array_t<float> output,
                                    int n, int c, int h, int w,
                                    float momentum,
                                    float eps) {
            py::buffer_info input_buf = input.request();
            float *input_fp = (float *)input_buf.ptr;
            py::buffer_info running_mean_buf = running_mean.request();
            float *running_mean_fp = (float *)running_mean_buf.ptr;
            py::buffer_info running_var_buf = running_var.request();
            float *running_var_fp = (float *)running_var_buf.ptr;
            py::buffer_info weight_buf = weight.request();
            float *weight_fp = (float *)weight_buf.ptr;
            py::buffer_info bias_buf = bias.request();
            float *bias_fp = (float *)bias_buf.ptr;
            py::buffer_info updated_mean_buf = updated_mean.request();
            float *updated_mean_fp = (float *)updated_mean_buf.ptr;
            py::buffer_info updated_var_buf = updated_var.request();
            float *updated_var_fp = (float *)updated_var_buf.ptr;
            py::buffer_info batch_mean_buf = batch_mean.request();
            float *batch_mean_fp = (float *)batch_mean_buf.ptr;
            py::buffer_info batch_invstd_buf = batch_invstd.request();
            float *batch_invstd_fp = (float *)batch_invstd_buf.ptr;
            py::buffer_info output_buf = output.request();
            float *output_fp = (float *)output_buf.ptr;

            bm_handle_t handle;
            bm_dev_request(&handle, 0);

            bm_status_t status = sgdnn_batchnorm_forward(handle,
                                bm_mem_from_system(input_fp),
                                bm_mem_from_system(running_mean_fp),
                                bm_mem_from_system(running_var_fp),
                                bm_mem_from_system(weight_fp),
                                bm_mem_from_system(bias_fp),
                                bm_mem_from_system(updated_mean_fp),
                                bm_mem_from_system(updated_var_fp),
                                bm_mem_from_system(batch_mean_fp),
                                bm_mem_from_system(batch_invstd_fp),
                                bm_mem_from_system(output_fp),
                                n, c, h, w,
                                momentum,
                                eps,
                                (sg_data_type_t)0);

            UNUSED(status);
            assert(status == BM_SUCCESS);
            bm_dev_free(handle);
        });

    m.def("batchnorm_backward", [](py::array_t<float16> grad_output,
                                    py::array_t<float16> input,
                                    py::array_t<float16> weight,
                                    py::array_t<float16> mean,
                                    py::array_t<float16> invstd,
                                    py::array_t<float16> grad_input,
                                    py::array_t<float16> grad_weight,
                                    py::array_t<float16> grad_bias,
                                    int n, int c, int h, int w,
                                    bool input_grad_enable,
                                    bool weight_grad_enable,
                                    bool bias_grad_enable) {
          py::buffer_info grad_output_buf = grad_output.request();
          float16 *grad_output_fp16 = (float16 *)grad_output_buf.ptr;
          py::buffer_info input_buf = input.request();
          float16 *input_fp16 = (float16 *)input_buf.ptr;
          py::buffer_info weight_buf = weight.request();
          float16 *weight_fp16 = (float16 *)weight_buf.ptr;
          py::buffer_info mean_buf = mean.request();
          float16 *mean_fp16 = (float16 *)mean_buf.ptr;
          py::buffer_info invstd_buf = invstd.request();
          float16 *invstd_fp16 = (float16 *)invstd_buf.ptr;
          py::buffer_info grad_input_buf = grad_input.request();
          float16 *grad_input_fp16 = (float16 *)grad_input_buf.ptr;
          py::buffer_info grad_weight_buf = grad_weight.request();
          float16 *grad_weight_fp16 = (float16 *)grad_weight_buf.ptr;
          py::buffer_info grad_bias_buf = grad_bias.request();
          float16 *grad_bias_fp16 = (float16 *)grad_bias_buf.ptr;

          bm_handle_t handle;
          bm_dev_request(&handle, 0);

          bm_status_t status = sgdnn_batchnorm_backward(handle,
                              bm_mem_from_system(grad_output_fp16),
                              bm_mem_from_system(input_fp16),
                              bm_mem_from_system(weight_fp16),
                              bm_mem_from_system(mean_fp16),
                              bm_mem_from_system(invstd_fp16),
                              bm_mem_from_system(grad_input_fp16),
                              bm_mem_from_system(grad_weight_fp16),
                              bm_mem_from_system(grad_bias_fp16),
                              n, c, h, w,
                              input_grad_enable,
                              weight_grad_enable,
                              bias_grad_enable,
                              (sg_data_type_t)1);

          UNUSED(status);
          assert(status == BM_SUCCESS);
          bm_dev_free(handle);
    });

    m.def("batchnorm_backward_fp32", [](py::array_t<float> grad_output,
                                        py::array_t<float> input,
                                        py::array_t<float> weight,
                                        py::array_t<float> mean,
                                        py::array_t<float> invstd,
                                        py::array_t<float> grad_input,
                                        py::array_t<float> grad_weight,
                                        py::array_t<float> grad_bias,
                                        int n, int c, int h, int w,
                                        bool input_grad_enable,
                                        bool weight_grad_enable,
                                        bool bias_grad_enable) {
          py::buffer_info grad_output_buf = grad_output.request();
          float *grad_output_fp32 = (float *)grad_output_buf.ptr;
          py::buffer_info input_buf = input.request();
          float *input_fp32 = (float *)input_buf.ptr;
          py::buffer_info weight_buf = weight.request();
          float *weight_fp32 = (float *)weight_buf.ptr;
          py::buffer_info mean_buf = mean.request();
          float *mean_fp32 = (float *)mean_buf.ptr;
          py::buffer_info invstd_buf = invstd.request();
          float *invstd_fp32 = (float *)invstd_buf.ptr;
          py::buffer_info grad_input_buf = grad_input.request();
          float *grad_input_fp32 = (float *)grad_input_buf.ptr;
          py::buffer_info grad_weight_buf = grad_weight.request();
          float *grad_weight_fp32 = (float *)grad_weight_buf.ptr;
          py::buffer_info grad_bias_buf = grad_bias.request();
          float *grad_bias_fp32 = (float *)grad_bias_buf.ptr;

          bm_handle_t handle;
          bm_dev_request(&handle, 0);

          bm_status_t status = sgdnn_batchnorm_backward(handle,
                              bm_mem_from_system(grad_output_fp32),
                              bm_mem_from_system(input_fp32),
                              bm_mem_from_system(weight_fp32),
                              bm_mem_from_system(mean_fp32),
                              bm_mem_from_system(invstd_fp32),
                              bm_mem_from_system(grad_input_fp32),
                              bm_mem_from_system(grad_weight_fp32),
                              bm_mem_from_system(grad_bias_fp32),
                              n, c, h, w,
                              input_grad_enable,
                              weight_grad_enable,
                              bias_grad_enable,
                              (sg_data_type_t)0);

          UNUSED(status);
          assert(status == BM_SUCCESS);
          bm_dev_free(handle);
    });

    m.def("avgpool_backward", [](py::array_t<float16> grad_output,
                                 py::array_t<float16> grad_input,
                                 int n, int c, int ih, int iw,
                                 int oh, int ow,
                                 int kh, int kw,
                                 int stride_h, int stride_w,
                                 int pad_h, int pad_w,
                                 bool ceil_mode,
                                 bool count_include_pad,
                                 int divisor_override) {

        py::buffer_info grad_out_buf = grad_output.request();
        float16 *grad_out_fp16 = (float16 *)grad_out_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float16 *grad_input_fp16 = (float16 *)grad_input_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        bm_status_t status = sgdnn_avgpool_backward(handle,
                            bm_mem_from_system(grad_out_fp16),
                            bm_mem_from_system(grad_input_fp16),
                            n, c, ih, iw, oh, ow,
                            kh, kw,
                            stride_h, stride_w,
                            pad_h, pad_w,
                            ceil_mode,
                            count_include_pad,
                            divisor_override,
                            (sg_data_type_t)1);

        UNUSED(status);
        //assert(status == BM_SUCCESS);
        bm_dev_free(handle);
    });

    m.def("maxpool_backward_f16", [](py::array_t<float16> forward_input,
                                     py::array_t<float16> forward_output,
                                     py::array_t<float16> grad_output,
                                     py::array_t<float16> grad_input,
                                     int n, int c, int ih, int iw,
                                     int oh, int ow,
                                     int kh, int kw,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int dilation_h, int dilation_w,
                                     bool ceil_mode) {

        py::buffer_info forward_input_buf = forward_input.request();
        float16 *forward_input_fp16 = (float16 *)forward_input_buf.ptr;
        py::buffer_info forward_output_buf = forward_output.request();
        float16 *forward_output_fp16 = (float16 *)forward_output_buf.ptr;
        py::buffer_info grad_out_buf = grad_output.request();
        float16 *grad_out_fp16 = (float16 *)grad_out_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float16 *grad_input_fp16 = (float16 *)grad_input_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        bm_status_t status = sgdnn_maxpool_backward(handle,
                            bm_mem_from_system(forward_input_fp16),
                            bm_mem_from_system(forward_output_fp16),
                            bm_mem_from_system(grad_out_fp16),
                            bm_mem_from_system(grad_input_fp16),
                            n, c, ih, iw, oh, ow,
                            kh, kw,
                            stride_h, stride_w,
                            pad_h, pad_w,
                            dilation_h, dilation_w,
                            ceil_mode,
                            (sg_data_type_t)1);

        UNUSED(status);
        //assert(status == BM_SUCCESS);
        bm_dev_free(handle);
    });

    m.def("maxpool_backward_f32", [](py::array_t<float> forward_input,
                                     py::array_t<float> forward_output,
                                     py::array_t<float> grad_output,
                                     py::array_t<float> grad_input,
                                     int n, int c, int ih, int iw,
                                     int oh, int ow,
                                     int kh, int kw,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int dilation_h, int dilation_w,
                                     bool ceil_mode) {

        py::buffer_info forward_input_buf = forward_input.request();
        float *forward_input_fp32 = (float *)forward_input_buf.ptr;
        py::buffer_info forward_output_buf = forward_output.request();
        float *forward_output_fp32 = (float *)forward_output_buf.ptr;
        py::buffer_info grad_out_buf = grad_output.request();
        float *grad_out_fp32 = (float *)grad_out_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float *grad_input_fp32 = (float *)grad_input_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        bm_status_t status = sgdnn_maxpool_backward(handle,
                            bm_mem_from_system(forward_input_fp32),
                            bm_mem_from_system(forward_output_fp32),
                            bm_mem_from_system(grad_out_fp32),
                            bm_mem_from_system(grad_input_fp32),
                            n, c, ih, iw, oh, ow,
                            kh, kw,
                            stride_h, stride_w,
                            pad_h, pad_w,
                            dilation_h, dilation_w,
                            ceil_mode,
                            (sg_data_type_t)0);

        UNUSED(status);
        //assert(status == BM_SUCCESS);
        bm_dev_free(handle);
    });

    m.def("eltwise_backward", [](py::array_t<float16> input_a,
                                 py::array_t<float16> input_b,
                                 py::array_t<float16> grad_output,
                                 py::array_t<float16> grad_input_a,
                                 py::array_t<float16> grad_input_b,
                                 int n, int c, int h, int w,
                                 int op_code,
                                 int coeff_a,
                                 int coeff_b,
                                 bool grad_input_a_enable,
                                 bool grad_input_b_enable) {
          py::buffer_info input_a_buf = input_a.request();
          float16 *input_a_fp16 = (float16 *)input_a_buf.ptr;
          py::buffer_info input_b_buf = input_b.request();
          float16 *input_b_fp16 = (float16 *)input_b_buf.ptr;
          py::buffer_info grad_output_buf = grad_output.request();
          float16 *grad_output_fp16 = (float16 *)grad_output_buf.ptr;
          py::buffer_info grad_input_a_buf = grad_input_a.request();
          float16 *grad_input_a_fp16 = (float16 *)grad_input_a_buf.ptr;
          py::buffer_info grad_input_b_buf = grad_input_b.request();
          float16 *grad_input_b_fp16 = (float16 *)grad_input_b_buf.ptr;

          bm_handle_t handle;
          bm_dev_request(&handle, 0);

          auto dtype_size = [](int dtype) {
              int size = 1;
              if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
              else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
                       dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
                  size = 2;
              else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
                  size = 4;
              return size;};

          bm_device_mem_t input_a_mem, input_b_mem, grad_output_mem;
          bm_device_mem_t grad_input_a_mem, grad_input_b_mem;
 
          u64 param_size = (u64)n * c * h * w * dtype_size(SG_DTYPE_FP16);

          DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(input_a_fp16), param_size, input_a_mem);
          DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(input_b_fp16), param_size, input_b_mem);
          DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(grad_output_fp16), param_size, grad_output_mem);
          DEVICE_MEM_NEW_OUTPUT(handle, bm_mem_from_system(grad_input_a_fp16), param_size, grad_input_a_mem);
          DEVICE_MEM_NEW_OUTPUT(handle, bm_mem_from_system(grad_input_b_fp16), param_size, grad_input_b_mem);

          OpTensorDescriptor_t opTensorDesc;
          opTensorDesc.op_code = op_code;

          TensorDescriptor_t aDesc;
          aDesc.dtype = SG_DTYPE_FP16;
          aDesc.ndims = 4;
          aDesc.shape[0] = n;
          aDesc.shape[1] = c;
          aDesc.shape[2] = h;
          aDesc.shape[3] = w;

          int beta[1] = {0};

          bm_status_t status = sgdnn_eltwise_backward(handle,
                               &coeff_a, aDesc,
                               ((void*)(input_a_mem.u.device.device_addr)),
                               &coeff_b, aDesc,// use real bDesc later
                               ((void*)(input_b_mem.u.device.device_addr)),
                               &beta, aDesc,// use real cDesc later
                               ((void*)(grad_output_mem.u.device.device_addr)),
                               ((void*)(grad_input_a_mem.u.device.device_addr)),
                               ((void*)(grad_input_b_mem.u.device.device_addr)),
                               grad_input_a_enable,
                               grad_input_b_enable,
                               opTensorDesc);

          DEVICE_MEM_DEL_OUTPUT(handle, bm_mem_from_system(grad_input_a_fp16), grad_input_a_mem);
          DEVICE_MEM_DEL_OUTPUT(handle, bm_mem_from_system(grad_input_b_fp16), grad_input_b_mem);
          DEVICE_MEM_DEL_INPUT(handle, bm_mem_from_system(input_a_fp16), input_a_mem);
          DEVICE_MEM_DEL_INPUT(handle, bm_mem_from_system(input_b_fp16), input_b_mem);
          DEVICE_MEM_DEL_INPUT(handle, bm_mem_from_system(grad_output_fp16), grad_output_mem);

          UNUSED(status);
          assert(status == BM_SUCCESS);
          bm_dev_free(handle);
    });

    m.def("linear_backward", [](py::array_t<float16> input,
                                py::array_t<float16> weight,
                                py::array_t<float16> grad_output,
                                py::array_t<float16> grad_input,
                                py::array_t<float16> grad_weight,
                                py::array_t<float16> grad_bias,
                                int batch,
                                int in_features,
                                int out_features,
                                bool input_grad_enable,
                                bool weight_grad_enable,
                                bool bias_grad_enable) {
        py::buffer_info input_buf = input.request();
        float16 *input_fp16 = (float16 *)input_buf.ptr;
        py::buffer_info weight_buf = weight.request();
        float16 *weight_fp16 = (float16 *)weight_buf.ptr;
        py::buffer_info grad_output_buf = grad_output.request();
        float16 *grad_output_fp16 = (float16 *)grad_output_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float16 *grad_input_fp16 = (float16 *)grad_input_buf.ptr;
        py::buffer_info grad_weight_buf = grad_weight.request();
        float16 *grad_weight_fp16 = (float16 *)grad_weight_buf.ptr;
        py::buffer_info grad_bias_buf = grad_bias.request();
        float16 *grad_bias_fp16 = (float16 *)grad_bias_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        bm_status_t status = sgdnn_linear_backward(handle,
                            bm_mem_from_system(input_fp16),
                            bm_mem_from_system(weight_fp16),
                            bm_mem_from_system(grad_output_fp16),
                            bm_mem_from_system(grad_input_fp16),
                            bm_mem_from_system(grad_weight_fp16),
                            bm_mem_from_system(grad_bias_fp16),
                            batch,
                            in_features,
                            out_features,
                            input_grad_enable,
                            weight_grad_enable,
                            bias_grad_enable,
                            (sg_data_type_t)1);

        UNUSED(status);
        assert(status == BM_SUCCESS);
        bm_dev_free(handle);
    });

    m.def("relu_backward", [](py::array_t<float16> input,
                              py::array_t<float16> grad_output,
                              py::array_t<float16> grad_input,
                              int n, int c, int h, int w,
                              bool input_grad_enable) {
        py::buffer_info input_buf = input.request();
        float16 *input_fp16 = (float16 *)input_buf.ptr;
        py::buffer_info grad_output_buf = grad_output.request();
        float16 *grad_output_fp16 = (float16 *)grad_output_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float16 *grad_input_fp16 = (float16 *)grad_input_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        bm_status_t status = sgdnn_relu_backward(handle,
                            bm_mem_from_system(input_fp16),
                            bm_mem_from_system(grad_output_fp16),
                            bm_mem_from_system(grad_input_fp16),
                            n, c, h, w,
                            input_grad_enable,
                            (sg_data_type_t)1);

        UNUSED(status);
        assert(status == BM_SUCCESS);
        bm_dev_free(handle);
    });

    m.def("cross_entropy_backward", [](py::array_t<float> input,
                                    py::array_t<float> target,
                                    py::array_t<float> grad_input,
                                    int batch,
                                    int cls_num,
                                    int reduction) {
        py::buffer_info input_buf = input.request();
        float *input_fp = (float *)input_buf.ptr;
        py::buffer_info target_buf = target.request();
        float *target_fp = (float *)target_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float *grad_input_fp = (float *)grad_input_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        bm_status_t status = sgdnn_cross_entropy_backward(handle,
                            bm_mem_from_system(input_fp),
                            bm_mem_from_system(target_fp),
                            bm_mem_from_system(grad_input_fp),
                            batch, cls_num, reduction,
                            (sg_data_type_t)0);

        UNUSED(status);
        assert(status == BM_SUCCESS);
        bm_dev_free(handle);
    });
    // add new ops backward here
    // m.def("batchnorm_backward", ...);
}
#endif
