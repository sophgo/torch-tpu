#include "sgdnn_api.h"
#include "sgdnn_helpers.h"
#include <assert.h>
#include <memory>
#include <string.h>
#include "sg_stas_gen_util.h"
#include "sg_fp16.h"

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

PYBIND11_MODULE(sgdnn, m)
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

    // add new ops backward here
    // m.def("batchnorm_backward", ...);
}
