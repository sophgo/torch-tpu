#include "sgdnn_api.h"
#include "sgdnn_helpers.h"
#include <assert.h>
#include <memory>
#include <string.h>
#include "kernel_module_data.h"

static inline int dtype_size(sg_data_type_t dtype) {
    int size = 1;
    if (dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8) size = 1;
    else if (dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
             dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16)
        size = 2;
    else if (dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32)
        size = 4;
    return size;
}

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

    m.def("conv_backward", [](py::array_t<float> grad_output,
                              py::array_t<float> input,
                              py::array_t<float> weight,
                              py::array_t<float> grad_input,
                              py::array_t<float> grad_weight,
                              py::array_t<float> grad_bias,
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
        float *grad_out_float = (float *)grad_out_buf.ptr;
        py::buffer_info input_buf = input.request();
        float *input_float = (float *)input_buf.ptr;
        py::buffer_info weight_buf = weight.request();
        float *weight_float = (float *)weight_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float *grad_input_float = (float *)grad_input_buf.ptr;
        py::buffer_info grad_weight_buf = grad_weight.request();
        float *grad_weight_float = (float *)grad_weight_buf.ptr;
        py::buffer_info grad_bias_buf = grad_bias.request();
        float *grad_bias_float = (float *)grad_bias_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);
        tpu_module_init(handle);

#if 1
        bm_status_t status = sgdnn_conv_backward(handle,
                            bm_mem_from_system(grad_out_float),
                            bm_mem_from_system(input_float),
                            bm_mem_from_system(weight_float),
                            bm_mem_from_system(grad_input_float),
                            bm_mem_from_system(grad_weight_float),
                            bm_mem_from_system(grad_bias_float),
                            n, ic, ih, iw, oc, oh, ow,
                            groups, kh, kw,
                            stride_h, stride_w, dh, dw,
                            pad_ht, pad_hb, pad_wl, pad_wr,
                            if_relu,
                            input_grad_enable,
                            weight_grad_enable,
                            bias_grad_enable,
                            (sg_data_type_t)0);
#else
        sg_data_type_t dtype = SG_DTYPE_FP32;
        bm_device_mem_t input_mem, weight_mem, grad_out_mem;
        bm_device_mem_t grad_input_mem, grad_weight_mem, grad_bias_mem;
        u64 grad_output_size = (u64)n * oc * oh * ow * dtype_size(dtype);
        u64 input_size = (u64)n * ic * ih * iw * dtype_size(dtype);
        //u64 weight_size = (u64)oc * (dtype == SG_DTYPE_FP32 ? ic : ALIGN(ic, 32)) * kh * kw * dtype_size(dtype);
        u64 weight_size = (u64)oc * ic * kh * kw * dtype_size(dtype);
        u64 grad_input_size = input_size;
        u64 grad_weight_size = (u64)oc * ic * kh * kw * dtype_size(dtype);
        u64 grad_bias_size = (u64)oc * dtype_size(dtype);

        DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(grad_out_float), grad_output_size, grad_out_mem);
        DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(input_float), input_size, input_mem);
        DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(weight_float), weight_size, weight_mem);
        DEVICE_MEM_NEW_OUTPUT(handle, bm_mem_from_system(grad_input_float), grad_input_size, grad_input_mem);
        DEVICE_MEM_NEW_OUTPUT(handle, bm_mem_from_system(grad_weight_float), grad_weight_size, grad_weight_mem);
        DEVICE_MEM_NEW_OUTPUT(handle, bm_mem_from_system(grad_bias_float), grad_bias_size, grad_bias_mem);

        TensorDescriptor_t xDesc, dyDesc, dbDesc;
        xDesc.dtype = SG_DTYPE_FP32;
        xDesc.ndims = 4;
        xDesc.shape[0] = n;
        xDesc.shape[1] = ic;
        xDesc.shape[2] = ih;
        xDesc.shape[3] = iw;

        dyDesc.dtype = SG_DTYPE_FP32;
        dyDesc.ndims = 4;
        dyDesc.shape[0] = n;
        dyDesc.shape[1] = oc;
        dyDesc.shape[2] = oh;
        dyDesc.shape[3] = ow;

        dbDesc.dtype = SG_DTYPE_FP32;
        dbDesc.ndims = 4;
        dbDesc.shape[0] = 1;
        dbDesc.shape[1] = oc;
        dbDesc.shape[2] = 1;
        dbDesc.shape[3] = 1;

        FilterDescriptor_t wDesc;
        wDesc.dtype = SG_DTYPE_FP32;
        wDesc.oc = oc;
        wDesc.ic = ic;
        wDesc.kh = kh;
        wDesc.kw = kw;

        assert(pad_ht == pad_hb);
        assert(pad_wl == pad_wr);
        ConvolutionDescriptor_t convDesc;
        convDesc.pad_h = pad_ht;
        convDesc.pad_w = pad_wl;
        convDesc.stride_h = stride_h;
        convDesc.stride_w = stride_w;
        convDesc.dilation_h = dh;
        convDesc.dilation_w = dw;
        convDesc.groups = 1;
        convDesc.computeType = SG_DTYPE_FP32;//SG_DTYPE_FP16 if AMP

        float alpha[1] = {1.0f};
        float beta[1] = {0.0f};

        bm_status_t status = sgdnn_conv_backward_cudnn(
                                handle,
                                alpha,
                                beta,
                                xDesc,
                                ((void*)(input_mem.u.device.device_addr)),
                                ((void*)(grad_input_mem.u.device.device_addr)),
                                wDesc,
                                ((void*)(weight_mem.u.device.device_addr)),
                                ((void*)(grad_weight_mem.u.device.device_addr)),
                                dbDesc,
                                ((void*)(grad_bias_mem.u.device.device_addr)),
                                dyDesc,
                                ((void*)(grad_out_mem.u.device.device_addr)),
                                convDesc,
                                input_grad_enable,
                                weight_grad_enable,
                                bias_grad_enable);

        DEVICE_MEM_DEL_OUTPUT(handle, bm_mem_from_system(grad_input_float), grad_input_mem);
        DEVICE_MEM_DEL_OUTPUT(handle, bm_mem_from_system(grad_weight_float), grad_weight_mem);
        DEVICE_MEM_DEL_OUTPUT(handle, bm_mem_from_system(grad_bias_float), grad_bias_mem);
        DEVICE_MEM_DEL_INPUT(handle, bm_mem_from_system(grad_out_float), grad_out_mem);
        DEVICE_MEM_DEL_INPUT(handle, bm_mem_from_system(input_float), input_mem);
        DEVICE_MEM_DEL_INPUT(handle, bm_mem_from_system(weight_float), weight_mem);
#endif

        UNUSED(status);
        //assert(status == BM_SUCCESS);
        tpu_module_deinit(handle);
        bm_dev_free(handle);
    });

    m.def("conv_backward_fp16", [](py::array_t<float16> grad_output,
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
        tpu_module_init(handle);

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
        tpu_module_deinit(handle);
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
            tpu_module_init(handle);

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
            tpu_module_deinit(handle);
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
          tpu_module_init(handle);

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
          tpu_module_deinit(handle);
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
          tpu_module_init(handle);

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
          tpu_module_deinit(handle);
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
        tpu_module_init(handle);

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
        tpu_module_deinit(handle);
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
        tpu_module_init(handle);

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
        tpu_module_deinit(handle);
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
        tpu_module_init(handle);

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
        tpu_module_deinit(handle);
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
                                 bool input_a_grad_enable,
                                 bool input_b_grad_enable) {
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
          tpu_module_init(handle);

          bm_status_t status = sgdnn_eltwise_backward(handle,
                              bm_mem_from_system(input_a_fp16),
                              bm_mem_from_system(input_b_fp16),
                              bm_mem_from_system(grad_output_fp16),
                              bm_mem_from_system(grad_input_a_fp16),
                              bm_mem_from_system(grad_input_b_fp16),
                              n, c, h, w,
                              op_code,
                              coeff_a,
                              coeff_b,
                              input_a_grad_enable,
                              input_b_grad_enable,
                              (sg_data_type_t)1);

          UNUSED(status);
          assert(status == BM_SUCCESS);
          tpu_module_deinit(handle);
          bm_dev_free(handle);
    });

    m.def("eltwise_backward_cudnn", [](py::array_t<float16> input_a,
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
          tpu_module_init(handle);

          bm_device_mem_t input_a_mem, input_b_mem, grad_output_mem;
          bm_device_mem_t grad_input_a_mem, grad_input_b_mem;

          u64 param_size = (u64)n * c * h * w * dtype_size(SG_DTYPE_FP16);

          DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(input_a_fp16), param_size, input_a_mem);
          DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(input_b_fp16), param_size, input_b_mem);
          DEVICE_MEM_NEW_INPUT(handle, bm_mem_from_system(grad_output_fp16), param_size, grad_output_mem);
          DEVICE_MEM_NEW_OUTPUT(handle, bm_mem_from_system(grad_input_a_fp16), param_size, grad_input_a_mem);
          DEVICE_MEM_NEW_OUTPUT(handle, bm_mem_from_system(grad_input_b_fp16), param_size, grad_input_b_mem);

          EltwiseOpMode_t opTensorDesc;
          if(op_code==0){opTensorDesc = OP_ELTWISE_PRODUCT;}
          if(op_code==1){opTensorDesc = OP_ELTWISE_COEF_ADD;}
          if(op_code==2){opTensorDesc = OP_ELTWISE_MAX;}

          TensorDescriptor_t aDesc;
          aDesc.dtype = SG_DTYPE_FP16;
          aDesc.ndims = 4;
          aDesc.shape[0] = n;
          aDesc.shape[1] = c;
          aDesc.shape[2] = h;
          aDesc.shape[3] = w;

          int beta[1] = {0};

          bm_status_t status = sgdnn_eltwise_backward_cudnn(handle,
                               &coeff_a,
                               aDesc,
                               ((void*)(input_a_mem.u.device.device_addr)),
                               &coeff_b,
                               aDesc,// use real bDesc later
                               ((void*)(input_b_mem.u.device.device_addr)),
                               &beta,
                               aDesc,// use real cDesc later
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
          tpu_module_deinit(handle);
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
        tpu_module_init(handle);

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
        tpu_module_deinit(handle);
        bm_dev_free(handle);
    });

    m.def("relu_forward", [](py::array_t<float16> input,
                             py::array_t<float16> output,
                             float upper_limit,
                             int n, int c, int h, int w) {
        py::buffer_info input_buf = input.request();
        float16 *input_fp16 = (float16 *)input_buf.ptr;
        py::buffer_info output_buf = output.request();
        float16 *output_fp16 = (float16 *)output_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);

        bm_status_t status = sgdnn_relu_forward(handle,
                            bm_mem_from_system(input_fp16),
                            bm_mem_from_system(output_fp16),
                            upper_limit,
                            n, c, h, w,
                            (sg_data_type_t)1);
        UNUSED(status);
        assert(status == BM_SUCCESS);
        bm_dev_free(handle);
    });

    m.def("relu_backward", [](py::array_t<float> input,
                              py::array_t<float> grad_output,
                              py::array_t<float> grad_input,
                              int n, int c, int h, int w,
                              bool input_grad_enable) {
        py::buffer_info input_buf = input.request();
        float *input_fp = (float *)input_buf.ptr;
        py::buffer_info grad_output_buf = grad_output.request();
        float *grad_output_fp = (float *)grad_output_buf.ptr;
        py::buffer_info grad_input_buf = grad_input.request();
        float *grad_input_fp = (float *)grad_input_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);
        tpu_module_init(handle);

        bm_status_t status = sgdnn_relu_backward(handle,
                            bm_mem_from_system(input_fp),
                            bm_mem_from_system(grad_output_fp),
                            bm_mem_from_system(grad_input_fp),
                            n, c, h, w,
                            input_grad_enable,
                            (sg_data_type_t)0);

        UNUSED(status);
        assert(status == BM_SUCCESS);
        tpu_module_deinit(handle);
        bm_dev_free(handle);
    });

    m.def("cross_entropy_forward", [](py::array_t<float> input,
                                    py::array_t<float> target,
                                    py::array_t<float> loss,
                                    int batch,
                                    int cls_num,
                                    int reduction) {
        py::buffer_info input_buf = input.request();
        float *input_fp = (float *)input_buf.ptr;
        py::buffer_info target_buf = target.request();
        float *target_fp = (float *)target_buf.ptr;
        py::buffer_info loss_buf = loss.request();
        float *loss_fp = (float *)loss_buf.ptr;

        bm_handle_t handle;
        bm_dev_request(&handle, 0);
        tpu_module_init(handle);

        bm_status_t status = sgdnn_cross_entropy_forward(handle,
                            bm_mem_from_system(input_fp),
                            bm_mem_from_system(target_fp),
                            bm_mem_from_system(loss_fp),
                            batch, cls_num, reduction,
                            (sg_data_type_t)0);

        UNUSED(status);
        assert(status == BM_SUCCESS);
        tpu_module_deinit(handle);
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
        tpu_module_init(handle);

        bm_status_t status = sgdnn_cross_entropy_backward(handle,
                            bm_mem_from_system(input_fp),
                            bm_mem_from_system(target_fp),
                            bm_mem_from_system(grad_input_fp),
                            batch, cls_num, reduction,
                            (sg_data_type_t)0);

        UNUSED(status);
        assert(status == BM_SUCCESS);
        tpu_module_deinit(handle);
        bm_dev_free(handle);
    });
    // add new ops backward here
    // m.def("batchnorm_backward", ...);
}
