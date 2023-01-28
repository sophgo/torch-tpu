#include "sgdnn_api.h"
#include "sgdnn_helpers.h"
#include <assert.h>
#include <memory>
#include <string.h>
#include "sg_fp16.h"

#ifdef USING_CMODEL
#include "sg_stas_gen_util.h"
#endif

//using namespace py::literals;
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

