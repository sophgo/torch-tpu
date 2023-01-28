#include "sgdnn_api.h"

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

    // add new ops backward here
    // m.def("batchnorm_backward", ...);
}
