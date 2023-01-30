#include "sgdnn_api.h"
#include "sgdnn_helpers.h"
#include <assert.h>
#include <memory>
#include <string.h>
#include "sg_fp16.h"

#ifdef USING_CMODEL
#include "sg_stas_gen_util.h"
#endif

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

    //sg_set_profile_dump(true);
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
    u64 weight_size = (u64)oc * ALIGN(ic, 32) * kh * kw * dtype_size(dtype);
    u64 grad_input_size = input_size;
    //grad_weight arrange as [ic, oc, kh, kw]
    u64 grad_weight_size = (u64)ic * oc * kh * kw * dtype_size(dtype);
    u64 grad_bias_size = (u64)oc * dtype_size(dtype);
    u64 weight_32oc_size = ALIGN(oc, 32) * kh * kw * ic * dtype_size(dtype);
    u64 grad_out_32n_size = ALIGN(n, 32) * oh * ow * oc * dtype_size(dtype);
    u64 buffer_size = sg_max(weight_32oc_size, grad_out_32n_size);//use for weight reorder

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
        {kh, kw},
        {stride_h, stride_w},
        {dh, dw},
        {pad_ht, pad_hb, pad_wl, pad_wr},
        1, 1, 1
    };

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
    sg_set_profile_dump(true);
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
    };

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
        dtype
    };

    tpu_kernel_launch_sync(handle, "tpu_kernel_api_maxpool_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);
    DEVICE_MEM_DEL_INPUT(handle, forward_input, forward_input_mem);
    DEVICE_MEM_DEL_INPUT(handle, forward_output, forward_output_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    return BM_SUCCESS;
}

// C = op(alpha1[0] * A, alpha2[0] * B) + beta[0] * C
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

    float alpha_A = ((float*)alpha1)[0];
    float alpha_B = ((float*)alpha2)[0];
    float beta_C = ((float*)beta)[0];
    assert(beta_C == 0.0f);

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
        alpha_A, alpha_B,
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

    float alpha_A = ((float*)alpha1)[0];
    float alpha_B = ((float*)alpha2)[0];
    float beta_C = ((float*)beta)[0];
    assert(beta_C == 0.0f);

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
