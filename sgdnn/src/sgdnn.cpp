#include "sgdnn_api.h"
#include "sgdnn_helpers.h"
#include <assert.h>
#include <map>
#include <memory>
#include <string.h>
#include "kernel_module_data.h"
#include "bmodel.hpp"

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

static inline sg_active_type_t tpu_active_type_convert(ActivationMode_t active_type) {
    sg_active_type_t atype = ACTIVE_RELU;
    switch (active_type) {
        case Activation_Sigmoid:    atype = ACTIVE_SIGMOID;     break;
        case Activation_Relu:       atype = ACTIVE_RELU;        break;
        case Activation_Tanh:       atype = ACTIVE_TANH;        break;
        case Activation_Elu:        atype = ACTIVE_ELU;         break;
        case Activation_Gelu:       atype = ACTIVE_GELU;        break;
        case Activation_Swish:      atype = ACTIVE_SWISH;       break;
    default:
        assert(0);
        break;
    }
    return atype;
}

static inline sg_binary_type_t tpu_binary_type_convert(BinaryOpMode_t binary_type) {
    sg_binary_type_t optype = BINARY_ADD;
    switch (binary_type) {
    case OP_BINARY_ADD:          optype = BINARY_ADD;           break;
    case OP_BINARY_SUB:          optype = BINARY_SUB;           break;
    case OP_BINARY_MUL:          optype = BINARY_MUL;           break;
    case OP_BINARY_DIV:          optype = BINARY_DIV;           break;
    case OP_BINARY_MAX:          optype = BINARY_MAX;           break;
    case OP_BINARY_MIN:          optype = BINARY_MIN;           break;
    case OP_BINARY_GT:           optype = BINARY_GT;            break;
    case OP_BINARY_GE:           optype = BINARY_GE;            break;
    case OP_BINARY_LT:           optype = BINARY_LT;            break;
    case OP_BINARY_LE:           optype = BINARY_LE;            break;
    case OP_BINARY_EQ:           optype = BINARY_EQ;            break;
    case OP_BINARY_NE:           optype = BINARY_NE;            break;
    case OP_BINARY_SQUARED_DIFF: optype = BINARY_SQUARED_DIFF;  break;
    case OP_BINARY_FLOOR_MOD:    optype = BINARY_FLOOR_MOD;     break;
    case OP_BINARY_FLOOR_DIV:    optype = BINARY_FLOOR_DIV;     break;
    default:
        assert(0);
        break;
    }
    return optype;
}

#define ASSERT_SAME_DIMS(A, B)            \
  assert(A.ndims == B.ndims);             \

#define ASSERT_SAME_SHAPE(A, B)           \
  assert(A.ndims == B.ndims);             \
  for (int dim = 0; dim < A.ndims; dim++) \
    assert(A.shape[dim] == B.shape[dim]); \

static std::map<bm_handle_t, tpu_kernel_module_t> tpu_kernel_module;

void tpu_module_init(bm_handle_t handle) {
    if (tpu_kernel_module.find(handle) != tpu_kernel_module.end()) return;
    const unsigned int *p = kernel_module_data;
    size_t length = sizeof(kernel_module_data);
    tpu_kernel_module_t tpu_module = tpu_kernel_load_module(handle, (const char *)p, length);
    tpu_kernel_module.insert(std::pair<bm_handle_t, tpu_kernel_module_t>(handle, tpu_module));
}

void tpu_module_deinit(bm_handle_t handle) {
    if (tpu_kernel_module.find(handle) == tpu_kernel_module.end()) return;
    assert(tpu_kernel_module.erase(handle));
}

static void sgdnn_tpu_kernel_launch(
        bm_handle_t     handle,
        const char*     func_name,
        const void*     api,
        size_t          api_size) {

    tpu_kernel_function_t func_id;
    tpu_kernel_module_t tpu_module = tpu_kernel_module[handle];
    func_id = tpu_kernel_get_function(handle, tpu_module, func_name);
    bm_status_t ret = tpu_kernel_launch(handle, func_id, (void*)api, api_size);
    if (ret != BM_SUCCESS) throw("tpu_kernel_launch failed");
}

#define SP(D, T) (std::shared_ptr<T>((D), std::default_delete<T []>()))
void get_coeff_data(const std::string& modelpath, u64 addr_offset, int coeff_size, float* coeff) {

    bmodel::ModelCtx model_ctx(modelpath);
    //just assume resnet50 has one net
    auto params = model_ctx.model()->net()->Get(0)->parameter();
    //just assume resnet50 has one netparam
    const bmodel::CoeffMem* coeff_mem = params->Get(0)->coeff_mem();

#define COEFF_BLK_SIZE 0x1000000
    u8* data = new u8[COEFF_BLK_SIZE];
    auto data_sp = SP(data, u8);
    u64 left_size = coeff_size;
    u64 offset = 0;
    while (left_size > 0) {
      u64 data_size = (left_size >= COEFF_BLK_SIZE ? COEFF_BLK_SIZE : left_size);
      model_ctx.read_binary(coeff_mem->binary_coeff(), offset, data, data_size);
      memcpy(coeff + offset * sizeof(float), data, data_size * sizeof(float));
      offset += data_size;
      left_size -= data_size;
    }
}

void set_coeff_data(const std::string& modelpath, u64 addr_offset, int coeff_size, float* coeff) {}

//use for pybind test, deprecate after
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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_forward", &api, sizeof(api));

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
    UNUSED(alpha_);
    float beta_ = ((float*)beta)[0];
    assert(beta_ == 0.0f || beta_ == 1.0f);
    bool result_add = beta_ == 1.0f;

    sg_data_type_t idtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t wdtype = (sg_data_type_t)(wDesc.dtype);
    sg_data_type_t bdtype = (sg_data_type_t)(bDesc.dtype);
    sg_data_type_t odtype = (sg_data_type_t)(yDesc.dtype);
    assert(bdtype == SG_DTYPE_FP32);

    sg_data_type_t compute_type = (sg_data_type_t)(convDesc.computeType);

    if (compute_type == SG_DTYPE_FP32) {

        assert(idtype == SG_DTYPE_FP32 &&
               wdtype == SG_DTYPE_FP32 &&
               odtype == SG_DTYPE_FP32);

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

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_forward", &api, sizeof(api));

    } else if (compute_type == SG_DTYPE_FP16) {

        if (idtype == SG_DTYPE_FP32) {

            assert(wdtype == SG_DTYPE_FP32 && odtype == SG_DTYPE_FP32);

            int dtype_size = 2;

            bm_device_mem_t x_fp16, w_fp16, w_32ic_fp16;
            u64 x_fp16_size = (u64)n * ic * ih * iw * dtype_size;
            u64 w_fp16_size = (u64)oc * ic * kh * kw * dtype_size;
            u64 w_32ic_fp16_size = (u64)oc * ALIGN(ic, 32) * kh * kw * dtype_size;

            DEVICE_MEM_NEW_BUFFER(handle, x_fp16, x_fp16_size);
            DEVICE_MEM_NEW_BUFFER(handle, w_fp16, w_fp16_size);
            DEVICE_MEM_NEW_BUFFER(handle, w_32ic_fp16, w_32ic_fp16_size);

            sg_api_dtype_convert_t cast_x_api;
            cast_x_api.input_global_addr = (unsigned long long)x;
            cast_x_api.output_global_addr = bm_mem_get_device_addr(x_fp16);
            memcpy(cast_x_api.shape, xDesc.shape, xDesc.ndims * sizeof(int));
            cast_x_api.dims = xDesc.ndims;
            cast_x_api.idtype = SG_DTYPE_FP32;
            cast_x_api.odtype = SG_DTYPE_FP16;
            cast_x_api.round_mode = SG_ROUND_EVEN;

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_x_api, sizeof(cast_x_api));

            sg_api_dtype_convert_t cast_w_api;
            cast_w_api.input_global_addr = (unsigned long long)w;
            cast_w_api.output_global_addr = bm_mem_get_device_addr(w_fp16);
            int w_shape[4] = {oc, ic, kh, kw};
            memcpy(cast_w_api.shape, w_shape, 4 * sizeof(int));
            cast_w_api.dims = 4;
            cast_w_api.idtype = SG_DTYPE_FP32;
            cast_w_api.odtype = SG_DTYPE_FP16;
            cast_w_api.round_mode = SG_ROUND_EVEN;

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_w_api, sizeof(cast_w_api));

            sg_api_conv_weight_reorder_t conv_weight_reorder_api = {
                bm_mem_get_device_addr(w_fp16),
                bm_mem_get_device_addr(w_32ic_fp16),
                {oc, ic, kh, kw},
                Reorder_To_32ic};

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_weight_reorder", &conv_weight_reorder_api, sizeof(conv_weight_reorder_api));

            sg_api_conv_forward_t api = {
                bm_mem_get_device_addr(x_fp16),
                bm_mem_get_device_addr(w_32ic_fp16),
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
                SG_DTYPE_FP16,
                odtype};

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_forward", &api, sizeof(api));

            bm_free_device(handle, x_fp16);
            bm_free_device(handle, w_fp16);
            bm_free_device(handle, w_32ic_fp16);

        } else if (idtype == SG_DTYPE_FP16) {

            assert(wdtype == SG_DTYPE_FP16);

            int dtype_size = 2;
            bm_device_mem_t w_32ic_fp16;
            u64 w_32ic_fp16_size = (u64)oc * ALIGN(ic, 32) * kh * kw * dtype_size;

            DEVICE_MEM_NEW_BUFFER(handle, w_32ic_fp16, w_32ic_fp16_size);

            sg_api_conv_weight_reorder_t conv_weight_reorder_api = {
                (unsigned long long)w,
                bm_mem_get_device_addr(w_32ic_fp16),
                {oc, ic, kh, kw},
                Reorder_To_32ic};

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_weight_reorder", &conv_weight_reorder_api, sizeof(conv_weight_reorder_api));

            sg_api_conv_forward_t api = {
                (unsigned long long)x,
                bm_mem_get_device_addr(w_32ic_fp16),
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

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_forward", &api, sizeof(api));
            bm_free_device(handle, w_32ic_fp16);
        }
    }

    return BM_SUCCESS;
}

//use for pybind test, deprecate after
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
    bool               grad_input_enable,
    bool               grad_weight_enable,
    bool               grad_bias_enable,
    sg_data_type_t     dtype
  ) {

    bm_device_mem_t grad_output_mem;
    bm_device_mem_t input_mem, weight_mem, buffer_mem;
    bm_device_mem_t grad_input_mem, grad_weight_mem, grad_bias_mem;
    u64 grad_output_size = (u64)n * oc * oh * ow * dtype_size(dtype);
    u64 input_size = (u64)n * ic * ih * iw * dtype_size(dtype);
    //u64 weight_size = (u64)oc * (dtype == SG_DTYPE_FP32 ? ic : ALIGN(ic, 32)) * kh * kw * dtype_size(dtype);
    u64 weight_size = (u64)oc * ic * kh * kw * dtype_size(dtype);
    u64 grad_input_size = input_size;
    //grad_weight arrange as [ic, oc, kh, kw]
    u64 grad_weight_size = (u64)ic * oc * kh * kw * dtype_size(dtype);
    u64 grad_bias_size = (u64)oc * dtype_size(dtype);

    // cal buffer size
    u64 weight_reorder_size = ALIGN(oc, 32) * kh * kw * ic * dtype_size(SG_DTYPE_FP16);
    u64 grad_out_reorder_size = ALIGN(n, 32) * oh * ow * oc * dtype_size(SG_DTYPE_FP16);
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
        grad_input_enable,
        grad_weight_enable,
        grad_bias_enable,
        dtype};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_backward", &api, sizeof(api));

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
    UNUSED(wdtype);
    UNUSED(odtype);

    // cal buffer size
    sg_data_type_t compute_type = (sg_data_type_t)(convDesc.computeType);

    bool need_buffer = idtype == SG_DTYPE_FP16 || compute_type == SG_DTYPE_FP16;

    u64 weight_reorder_size = ALIGN(oc, 32) * kh * kw * ic * dtype_size(SG_DTYPE_FP16);
    u64 grad_out_reorder_size = ALIGN(n, 32) * oh * ow * oc * dtype_size(SG_DTYPE_FP16);
    u64 buffer_size = need_buffer ? sg_max(weight_reorder_size, grad_out_reorder_size) : 0;//use for weight reorder

    bm_device_mem_t buffer_mem;
    if (buffer_size > 0) {
        DEVICE_MEM_NEW_BUFFER(handle, buffer_mem, buffer_size);
    }

    if ((idtype == SG_DTYPE_FP32 && compute_type == SG_DTYPE_FP32) ||
        (idtype == SG_DTYPE_FP16 && compute_type == SG_DTYPE_FP16)) {

        if (dx_enable || dw_enable || db_enable) {
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

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_backward", &api, sizeof(api));
        }

        if (buffer_size > 0) {
            bm_free_device(handle, buffer_mem);
        }

    } else if (idtype == SG_DTYPE_FP32 && compute_type == SG_DTYPE_FP16) {

        int dtype_size = 2;

        bm_device_mem_t x_fp16, w_fp16, dy_fp16;
        bm_device_mem_t dx_fp16, dw_fp16, db_fp16;
        u64 x_fp16_size = (u64)n * ic * ih * iw * dtype_size;
        u64 w_fp16_size = (u64)oc * ic * kh * kw * dtype_size;
        u64 dy_fp16_size = (u64)n * oc * oh * ow * dtype_size;
        u64 dx_fp16_size = x_fp16_size;
        u64 dw_fp16_size = w_fp16_size;
        u64 db_fp16_size = oc * dtype_size;

        if (dx_enable || dw_enable || db_enable) {
            DEVICE_MEM_NEW_BUFFER(handle, x_fp16, x_fp16_size);
            DEVICE_MEM_NEW_BUFFER(handle, w_fp16, w_fp16_size);
            DEVICE_MEM_NEW_BUFFER(handle, dy_fp16, dy_fp16_size);
        }
        if (dx_enable) DEVICE_MEM_NEW_BUFFER(handle, dx_fp16, dx_fp16_size);
        if (dw_enable) DEVICE_MEM_NEW_BUFFER(handle, dw_fp16, dw_fp16_size);
        if (db_enable) DEVICE_MEM_NEW_BUFFER(handle, db_fp16, db_fp16_size);

        if (dx_enable || dw_enable || db_enable) {

            sg_api_dtype_convert_t cast_x_api;
            cast_x_api.input_global_addr = (unsigned long long)x;
            cast_x_api.output_global_addr = bm_mem_get_device_addr(x_fp16);
            memcpy(cast_x_api.shape, xDesc.shape, xDesc.ndims * sizeof(int));
            cast_x_api.dims = xDesc.ndims;
            cast_x_api.idtype = SG_DTYPE_FP32;
            cast_x_api.odtype = SG_DTYPE_FP16;
            cast_x_api.round_mode = SG_ROUND_EVEN;

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_x_api, sizeof(cast_x_api));

            sg_api_dtype_convert_t cast_w_api;
            cast_w_api.input_global_addr = (unsigned long long)w;
            cast_w_api.output_global_addr = bm_mem_get_device_addr(w_fp16);
            int w_shape[4] = {oc, ic, kh, kw};
            memcpy(cast_w_api.shape, w_shape, 4 * sizeof(int));
            cast_w_api.dims = 4;
            cast_w_api.idtype = SG_DTYPE_FP32;
            cast_w_api.odtype = SG_DTYPE_FP16;
            cast_w_api.round_mode = SG_ROUND_EVEN;

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_w_api, sizeof(cast_w_api));

            sg_api_dtype_convert_t cast_dy_api;
            cast_dy_api.input_global_addr = (unsigned long long)dy;
            cast_dy_api.output_global_addr = bm_mem_get_device_addr(dy_fp16);
            memcpy(cast_dy_api.shape, dyDesc.shape, dyDesc.ndims * sizeof(int));
            cast_dy_api.dims = dyDesc.ndims;
            cast_dy_api.idtype = SG_DTYPE_FP32;
            cast_dy_api.odtype = SG_DTYPE_FP16;
            cast_dy_api.round_mode = SG_ROUND_EVEN;

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_dy_api, sizeof(cast_dy_api));

            sg_api_conv_backward_t api = {
                bm_mem_get_device_addr(x_fp16),
                bm_mem_get_device_addr(w_fp16),
                bm_mem_get_device_addr(dy_fp16),
                bm_mem_get_device_addr(dx_fp16),
                bm_mem_get_device_addr(dw_fp16),
                bm_mem_get_device_addr(db_fp16),
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
                SG_DTYPE_FP16};

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_backward", &api, sizeof(api));
        }

        if (dx_enable) {

            sg_api_dtype_convert_t cast_dx_api;
            cast_dx_api.input_global_addr = bm_mem_get_device_addr(dx_fp16);
            cast_dx_api.output_global_addr = (unsigned long long)dx;
            memcpy(cast_dx_api.shape, xDesc.shape, xDesc.ndims * sizeof(int));
            cast_dx_api.dims = xDesc.ndims;
            cast_dx_api.idtype = SG_DTYPE_FP16;
            cast_dx_api.odtype = SG_DTYPE_FP32;
            cast_dx_api.round_mode = SG_ROUND_EVEN;

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_dx_api, sizeof(cast_dx_api));
        }

        if (dw_enable) {

            sg_api_dtype_convert_t cast_dw_api;
            cast_dw_api.input_global_addr = bm_mem_get_device_addr(dw_fp16);
            cast_dw_api.output_global_addr = (unsigned long long)dw;
            int dw_shape[4] = {oc, ic, kh, kw};
            memcpy(cast_dw_api.shape, dw_shape, 4 * sizeof(int));
            cast_dw_api.dims = 4;
            cast_dw_api.idtype = SG_DTYPE_FP16;
            cast_dw_api.odtype = SG_DTYPE_FP32;
            cast_dw_api.round_mode = SG_ROUND_EVEN;

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_dw_api, sizeof(cast_dw_api));
        }

        if (db_enable) {

            sg_api_dtype_convert_t cast_db_api;
            cast_db_api.input_global_addr = bm_mem_get_device_addr(db_fp16);
            cast_db_api.output_global_addr = (unsigned long long)db;
            memcpy(cast_db_api.shape, dbDesc.shape, dbDesc.ndims * sizeof(int));
            cast_db_api.dims = dbDesc.ndims;
            cast_db_api.idtype = SG_DTYPE_FP16;
            cast_db_api.odtype = SG_DTYPE_FP32;
            cast_db_api.round_mode = SG_ROUND_EVEN;

            sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_db_api, sizeof(cast_db_api));
        }

        if (buffer_size > 0) {
            bm_free_device(handle, buffer_mem);
        }
        if (dx_enable) bm_free_device(handle, dx_fp16);
        if (dw_enable) bm_free_device(handle, dw_fp16);
        if (db_enable) bm_free_device(handle, db_fp16);
        if (dx_enable || dw_enable || db_enable) {
            bm_free_device(handle, x_fp16);
            bm_free_device(handle, w_fp16);
            bm_free_device(handle, dy_fp16);
        }
    } else {
        //not support input is FP16 but compute type is FP32
        assert(0);
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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batchnorm_forward", &api, sizeof(api));

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
    BatchNormMode_t                  mode,
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
    float alpha_ = ((float*)alpha)[0];
    assert(alpha_ == 1.0f);
    float beta_ = ((float*)beta)[0];
    assert(beta_ == 0.0f || beta_ == 1.0f);

    int n, c, h, w;

    float eps = epsilon;
    sg_data_type_t idtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t odtype = (sg_data_type_t)(yDesc.dtype);

    if( mode == BatchNorm_Spatial )
    {
        unsigned long long input        = (unsigned long long)x;
        unsigned long long weight       = (unsigned long long)bnScale;
        unsigned long long bias         = (unsigned long long)bnBias;
        unsigned long long running_mean = (unsigned long long)resultRunningMean;
        unsigned long long running_var  = (unsigned long long)resultRunningVariance;
        unsigned long long batch_mean   = (unsigned long long)resultSaveMean;
        unsigned long long batch_invstd = (unsigned long long)resultSaveInvVariance;
        unsigned long long output       = (unsigned long long)y;

        assert((xDesc.ndims == 4 && yDesc.ndims == 4) || (xDesc.ndims == 3 && yDesc.ndims == 3) );
        if (xDesc.ndims == 4)
        {
            n = xDesc.shape[0];
            c = xDesc.shape[1];
            h = xDesc.shape[2];
            w = xDesc.shape[3];
        }
        else if (xDesc.ndims == 3)
        {
            n = xDesc.shape[0];
            c = xDesc.shape[1];
            h = 1;
            w = xDesc.shape[2];
        }

        float momentum = exponentialAverageFactor;

        if ( bnScale != nullptr || bnBias != nullptr || resultRunningMean != nullptr || resultRunningVariance != nullptr )
        {
            sg_data_type_t wdtype = (sg_data_type_t)(bnScaleBiasMeanVarDesc.dtype);
            assert(idtype == wdtype && wdtype == odtype);
            assert(bnScaleBiasMeanVarDesc.ndims == 1 );
            assert(bnScaleBiasMeanVarDesc.shape[0] == xDesc.shape[1] );
        }

        sg_api_batchnorm_forward_t api;
        api.input_global_addr           = input;
        api.running_mean_global_addr    = running_mean;
        api.running_var_global_addr     = running_var;
        api.weight_global_addr          = weight;
        api.bias_global_addr            = bias;
        api.updated_mean_global_addr    = running_mean;
        api.updated_var_global_addr     = running_var;
        api.batch_mean_global_addr      = batch_mean;
        api.batch_invstd_global_addr    = batch_invstd;
        api.output_global_addr          = output;
        api.momentum                    = momentum;
        api.eps                         = eps;
        api.dtype                       = idtype;
        api.shape[0]                    = n;
        api.shape[1]                    = c;
        api.shape[2]                    = h;
        api.shape[3]                    = w;

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batchnorm_forward_v2", &api, sizeof(api));
        return BM_SUCCESS;
    }
    else if ( mode == BatchNorm_Per_Layer )
    {
        unsigned long long input   = (unsigned long long)x;
        unsigned long long weight  = (unsigned long long)bnScale;
        unsigned long long bias    = (unsigned long long)bnBias;
        unsigned long long mean    = (unsigned long long)resultSaveMean;
        unsigned long long rstd    = (unsigned long long)resultSaveInvVariance;
        unsigned long long output  = (unsigned long long)y;

        assert( resultRunningMean == nullptr && resultRunningVariance == nullptr );

        int affine = 0, save_stat = 0;
        if ( resultSaveMean != nullptr && resultSaveInvVariance != nullptr )
        {
            save_stat = 1;
        }
        if ( bnScale != nullptr && bnBias != nullptr )
        {
            affine = 1;
            sg_data_type_t wdtype = (sg_data_type_t)(bnScaleBiasMeanVarDesc.dtype);
            assert(idtype == wdtype && wdtype == odtype);
        }

        int normalized_ndim = bnScaleBiasMeanVarDesc.ndims;
        int input_ndim = xDesc.ndims;
        int axis = input_ndim - normalized_ndim;
        sg_api_layernorm_forward_t api;
        api.input_global_addr   = input;
        api.weight_global_addr  = weight;
        api.bias_global_addr    = bias;
        api.output_global_addr  = output;
        api.mean_global_addr    = mean;
        api.rstd_global_addr    = rstd;
        api.dims                = input_ndim;
        api.axis                = axis;
        api.eps                 = eps;
        api.affine              = affine;
        api.save_stat           = save_stat;
        api.dtype               = idtype;
        for (int i =0; i < input_ndim; i++){
            api.shape[i] = xDesc.shape[i];
        }

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_layernorm_forward", &api, sizeof(api));
        return BM_SUCCESS;
    }

    return BM_ERR_NOFEATURE;
}

//use for pybind test, deprecate after
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
    bool               if_ln,
    bool               input_need_grad,
    bool               weight_need_grad,
    bool               bias_need_grad,
    sg_data_type_t     dtype)
{
    bm_device_mem_t grad_output_mem, input_mem, weight_mem, mean_mem, invstd_mem;
    bm_device_mem_t grad_input_mem, grad_weight_mem, grad_bias_mem;
    u64 param_size = (u64)n * c * h * w * dtype_size(dtype);
    u64 mv_size = (u64)( if_ln ? n * c : c) * dtype_size(dtype);
    u64 wb_size = (u64)( if_ln ? w : c) * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT (handle, grad_output, param_size,  grad_output_mem);
    DEVICE_MEM_NEW_INPUT (handle, input,       param_size,  input_mem);
    DEVICE_MEM_NEW_INPUT (handle, weight,      wb_size,     weight_mem);
    DEVICE_MEM_NEW_INPUT (handle, mean,        mv_size,     mean_mem);
    DEVICE_MEM_NEW_INPUT (handle, invstd,      mv_size,     invstd_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input,  param_size,  grad_input_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_weight, wb_size,     grad_weight_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_bias,   wb_size,     grad_bias_mem);

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
        1, 1, 1,
        dtype};

    if(if_ln) sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_layernorm_backward", &api, sizeof(api));
    else      sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batchnorm_backward", &api, sizeof(api));

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
    BatchNormMode_t                  mode,
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

    float alpha_data = ((float*)alphaDataDiff)[0];
    assert(alpha_data == 1.0f);
    float alpha_param = ((float*)alphaParamDiff)[0];
    assert(alpha_param == 1.0f);
    float beta_data = ((float*)betaDataDiff)[0];
    assert(beta_data == 0.0f);
    float beta_param = ((float*)betaParamDiff)[0];
    assert(beta_param == 0.0f);

    int n, c, h, w;
    if(mode == BatchNorm_Spatial)
    {
        assert(dyDesc.ndims == 4);
        assert( xDesc.ndims == 4);
        assert(dxDesc.ndims == 4);

        n = xDesc.shape[0];
        c = xDesc.shape[1];
        h = xDesc.shape[2];
        w = xDesc.shape[3];
    }
    else if(mode == BatchNorm_Per_Layer)
    {
        assert(dyDesc.ndims == 3);
        assert( xDesc.ndims == 3);
        if(dx_enable) assert(dxDesc.ndims == 3);

        n = xDesc.shape[0];
        c = xDesc.shape[1];
        h = 1;
        w = xDesc.shape[2];
    }
    else
    {
        assert(0);
    }

    assert(bnScaleBiasDiffDesc.ndims == 1);

    sg_data_type_t dydtype = (sg_data_type_t)(dyDesc.dtype);
    sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t dxdtype = (sg_data_type_t)(dxDesc.dtype);
    sg_data_type_t wdtype = (sg_data_type_t)(bnScaleBiasDiffDesc.dtype);

    // if dtype is fp16, it will convert to fp32 in local
    assert(xdtype == dydtype);
    if (dx_enable)              { assert(xdtype == dxdtype);}
    if (dw_enable || db_enable) { assert(xdtype == wdtype);}

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
        db_enable,
        xdtype};

    if(mode == BatchNorm_Spatial) sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batchnorm_backward", &api, sizeof(api));
    else if(mode == BatchNorm_Per_Layer) sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_layernorm_backward", &api, sizeof(api));
    else {assert(0);}

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_pooling_forward", &api, sizeof(api));

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
    assert(*(float *)alpha == 1.f);
    assert(*(float *)beta == 0.f);
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
    UNUSED(odtype);

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_pooling_forward", &api, sizeof(api));

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_avgpool_backward", &api, sizeof(api));

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_maxpool_backward", &api, sizeof(api));

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
    const TensorDescriptor_t    yDesc,
    const void                 *y,
    const TensorDescriptor_t    dyDesc,
    const void                 *dy,
    const TensorDescriptor_t    xDesc,
    const void                 *x,
    const void                 *beta,
    const TensorDescriptor_t    dxDesc,
    void                       *dx
 ) {
    assert(*(float *)alpha == 1.f);
    assert(*(float *)beta == 0.f);
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

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_maxpool_backward", &api, sizeof(api));

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

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_avgpool_backward", &api, sizeof(api));
    }

    return BM_SUCCESS;
}

bm_status_t sgdnn_binary_cudnn(
    bm_handle_t                 handle,
    const TensorDescriptor_t    aDesc,
    const void*                 A,
    const TensorDescriptor_t    bDesc,
    const void*                 B,
    const TensorDescriptor_t    cDesc,
    void*                       C,
    BinaryOpMode_t              opTensorDesc)
{
    sg_binary_type_t binary_type = tpu_binary_type_convert(opTensorDesc);

    // if (aDesc.ndims > 0 && bDesc.ndims > 0 && aDesc.ndims > bDesc.ndims)
    // {
    //     TensorDescriptor_t bDescSaved = bDesc;
    //     int dimgap = aDesc.ndims - bDesc.ndims;
    //     int i = 0;
    //     for (; i < dimgap; ++i)
    //     {
    //         const_cast<TensorDescriptor_t &>(bDesc).shape[i] = 1;
    //     }
    //     for (; i < aDesc.ndims; ++i)
    //     {
    //         const_cast<TensorDescriptor_t &>(bDesc).shape[i] = bDescSaved.shape[i - dimgap];
    //     }
    //     const_cast<TensorDescriptor_t &>(bDesc).ndims = aDesc.ndims;
    // }
    // else if (aDesc.ndims > 0 && bDesc.ndims > 0 && aDesc.ndims < bDesc.ndims)
    // {
    //     TensorDescriptor_t aDescSaved = aDesc;
    //     int dimgap = bDesc.ndims - aDesc.ndims;
    //     int i = 0;
    //     for (; i < dimgap; ++i)
    //     {
    //         const_cast<TensorDescriptor_t &>(aDesc).shape[i] = 1;
    //     }
    //     for (; i < bDesc.ndims; ++i)
    //     {
    //         const_cast<TensorDescriptor_t &>(aDesc).shape[i] = aDescSaved.shape[i - dimgap];
    //     }
    //     const_cast<TensorDescriptor_t &>(aDesc).ndims = bDesc.ndims;
    // }

    if(aDesc.ndims && bDesc.ndims && cDesc.ndims)
    {
        sg_data_type_t dtype_A = (sg_data_type_t)(aDesc.dtype);
        sg_data_type_t dtype_B = (sg_data_type_t)(bDesc.dtype);
        sg_data_type_t dtype_C = (sg_data_type_t)(cDesc.dtype);
        assert(dtype_A == dtype_B && dtype_B == dtype_C);

        sg_api_bcbinary_float_t api;
        api.A_global_addr = (unsigned long long)A;
        api.B_global_addr = (unsigned long long)B;
        api.res_global_addr = (unsigned long long)C;
        for (int i = 0; i< aDesc.ndims; ++i)
        {
            api.A_shape[i] = aDesc.shape[i];
        }
        for (int i = 0; i< bDesc.ndims; ++i)
        {
            api.B_shape[i] = bDesc.shape[i];
        }
        api.A_dims = aDesc.ndims;
        api.B_dims = bDesc.ndims;
        api.dtype = dtype_A;
        api.binary_type = binary_type;

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_bcbinary_float", &api, sizeof(api));
    }
    else if((!aDesc.ndims && bDesc.ndims) || (!bDesc.ndims && aDesc.ndims))
    {
        const void* tensor = !aDesc.ndims ? B : A;
        const void* scalar = !aDesc.ndims ? A : B;
        TensorDescriptor_t tensorDesc = !aDesc.ndims ? bDesc : aDesc;
        TensorDescriptor_t scalarDesc = !aDesc.ndims ? aDesc : bDesc;

        float const_value = ((float*)scalar)[0];
        sg_data_type_t tensor_dtype = (sg_data_type_t)(tensorDesc.dtype);
        sg_data_type_t scalar_dtype = (sg_data_type_t)(scalarDesc.dtype);
        switch (scalar_dtype)
        {
            case SG_DTYPE_INT8:     const_value = ((s8*) scalar)[0];  break;
            case SG_DTYPE_UINT8:    const_value = ((u8*) scalar)[0];  break;
            case SG_DTYPE_INT16:    const_value = ((s16*)scalar)[0];  break;
            case SG_DTYPE_UINT16:   const_value = ((u16*)scalar)[0];  break;
            case SG_DTYPE_INT32:    const_value = ((s32*)scalar)[0];  break;
            case SG_DTYPE_UINT32:   const_value = ((u32*)scalar)[0];  break;
            case SG_DTYPE_FP32:                                       break;
            case SG_DTYPE_FP16:     assert(0);                        break;
            case SG_DTYPE_BFP16:    assert(0);                        break;
            default:                assert(0);                        break;
        }
        int n, c, h, w;
        if (tensorDesc.ndims == 4)
        {
            n = tensorDesc.shape[0];
            c = tensorDesc.shape[1];
            h = tensorDesc.shape[2];
            w = tensorDesc.shape[3];
        }
        else if (tensorDesc.ndims == 2)
        {
            n = 1;
            c = tensorDesc.shape[0];
            h = 1;
            w = tensorDesc.shape[1];
        }
        else if (tensorDesc.ndims == 1)
        {
            n = 1;
            c = tensorDesc.shape[0];
            h = 1;
            w = 1;
        }
        else
        {
            assert(false);
        }
        bool is_inversed = !aDesc.ndims;
        sg_api_const_binary_float_t api = {
            (unsigned long long)tensor,
            (unsigned long long)C,
            {n, c, h, w},
            4,
            binary_type,
            tensor_dtype,
            const_value,
            is_inversed};
        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_const_binary", &api, sizeof(api));
    }
    else
    {
        assert(0);
    }
    return BM_SUCCESS;
}

// C = op(alpha1[0] * A, alpha2[0] * B) + beta[0] * C
bm_status_t sgdnn_eltwise_forward_cudnn(
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
    const EltwiseOpMode_t       opTensorDesc
) {

    int op_code = opTensorDesc;
    int A_n, A_c, A_h, A_w, B_n, B_c, B_h, B_w, C_n, C_c, C_h, C_w;
    if (aDesc.ndims == 4 && bDesc.ndims == 4 && cDesc.ndims == 4)
    {
       A_n = aDesc.shape[0];
       A_c = aDesc.shape[1];
       A_h = aDesc.shape[2];
       A_w = aDesc.shape[3];

       B_n = bDesc.shape[0];
       B_c = bDesc.shape[1];
       B_h = bDesc.shape[2];
       B_w = bDesc.shape[3];

       C_n = cDesc.shape[0];
       C_c = cDesc.shape[1];
       C_h = cDesc.shape[2];
       C_w = cDesc.shape[3];
    }
    else if (aDesc.ndims == 1 && bDesc.ndims == 1 && cDesc.ndims == 1)
    {
       A_n = 1;
       A_c = aDesc.shape[0];
       A_h = 1;
       A_w = 1;

       B_n = 1;
       B_c = bDesc.shape[0];
       B_h = 1;
       B_w = 1;

       C_n = 1;
       C_c = cDesc.shape[0];
       C_h = 1;
       C_w = 1;
     }
     else if (aDesc.ndims == 2 && bDesc.ndims == 2 && cDesc.ndims == 2)
     {
       A_n = 1;
       A_c = aDesc.shape[0];
       A_h = 1;
       A_w = aDesc.shape[1];

       B_n = 1;
       B_c = bDesc.shape[0];
       B_h = 1;
       B_w = bDesc.shape[1];

       C_n = 1;
       C_c = cDesc.shape[0];
       C_h = 1;
       C_w = cDesc.shape[1];
     }
     else
     {
       assert(false);
     }

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
    UNUSED(dtype_B);

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_eltwise_forward", &api, sizeof(api));

    return BM_SUCCESS;
}

bm_status_t sgdnn_eltwise_backward(
    bm_handle_t        handle,
    bm_device_mem_t    input_a,
    bm_device_mem_t    input_b,
    bm_device_mem_t    grad_output,
    bm_device_mem_t    grad_input_a,
    bm_device_mem_t    grad_input_b,
    int                n,
    int                c,
    int                h,
    int                w,
    int                op_code,
    int                coeff_a,
    int                coeff_b,
    bool               input_a_need_grad,
    bool               input_b_need_grad,
    sg_data_type_t     dtype)
{

    bm_device_mem_t input_a_mem, input_b_mem, grad_output_mem;
    bm_device_mem_t grad_input_a_mem, grad_input_b_mem;
    u64 param_size = (u64)n * c * h * w * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, input_a, param_size, input_a_mem);
    DEVICE_MEM_NEW_INPUT(handle, input_b, param_size, input_b_mem);
    DEVICE_MEM_NEW_INPUT(handle, grad_output, param_size, grad_output_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input_a, param_size, grad_input_a_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, grad_input_b, param_size, grad_input_b_mem);

    sg_api_eltwise_backward_t api = {
        bm_mem_get_device_addr(input_a_mem),
        bm_mem_get_device_addr(input_b_mem),
        bm_mem_get_device_addr(grad_output_mem),
        bm_mem_get_device_addr(grad_input_a_mem),
        bm_mem_get_device_addr(grad_input_b_mem),
        {n, c, h, w},
        op_code,
        coeff_a,
        coeff_b,
        1,1};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_eltwise_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input_a, input_a_mem);
    DEVICE_MEM_DEL_INPUT(handle, input_b, input_b_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input_a, grad_input_a_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input_b, grad_input_b_mem);

    return BM_SUCCESS;
}

bm_status_t sgdnn_eltwise_backward_cudnn(
    bm_handle_t                 handle,
    const void*                 alpha1,
    const TensorDescriptor_t    aDesc,
    const void*                 input_a,
    const void*                 alpha2,
    const TensorDescriptor_t    bDesc,
    const void*                 input_b,
    const void*                 beta,
    const TensorDescriptor_t    cDesc,
    const void*                 grad_output,
    void*                       grad_input_a,
    void*                       grad_input_b,
    bool                        grad_input_a_enable,
    bool                        grad_input_b_enable,
    const EltwiseOpMode_t       opTensorDesc
 ) {

    int op_code = opTensorDesc;

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
    if(op_code!=1)
    {
        // TODO:coeff product & coeff max
        assert(alpha_A==1 && alpha_B==1);
    }

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_eltwise_backward", &api, sizeof(api));

    return BM_SUCCESS;
}

bm_status_t sgdnn_relu_forward(
    bm_handle_t        handle,
    bm_device_mem_t    input,
    bm_device_mem_t    output,
    int                n,
    int                c,
    int                h,
    int                w,
    float              upper_limit,
    sg_data_type_t     dtype)
{

    bm_device_mem_t input_mem;
    bm_device_mem_t output_mem;
    u64 size = (u64)n * c * h * w * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, input, size, input_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, output, size, output_mem);

    sg_api_relu_forward_t api = {
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(output_mem),
        {n, c, h, w},
        upper_limit,
        (sg_data_type_t)1};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_relu_forward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, output, output_mem);

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
        dtype};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_relu_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);

    return BM_SUCCESS;
}

bm_status_t sgdnn_activation_forward_cudnn(
    bm_handle_t                     handle,
    ActivationDescriptor_t          activationDesc,
    const void                     *alpha,
    const TensorDescriptor_t        xDesc,
    const void                     *x,
    const void                     *beta,
    const TensorDescriptor_t        yDesc,
    void                           *y)
{
    if(activationDesc.mode == Activation_Relu)
    {
        unsigned long long input    = (unsigned long long)x;
        unsigned long long output   = (unsigned long long)y;

        float alpha_ = ((float*)alpha)[0];
        assert(alpha_ == 1.0f);
        float beta_ = ((float*)beta)[0];
        assert(beta_ == 0.0f || beta_ == 1.0f);

        float upper_limit = activationDesc.coef;
        assert(xDesc.ndims == 4);
        assert(yDesc.ndims == 4);
        int n = xDesc.shape[0];
        int c = xDesc.shape[1];
        int h = xDesc.shape[2];
        int w = xDesc.shape[3];

        sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
        sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
        assert(xdtype == ydtype);

        sg_api_relu_forward_t api = {
            input,
            output,
            {n, c, h, w},
            upper_limit,
            xdtype};

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_relu_forward", &api, sizeof(api));

        return BM_SUCCESS;
    }
    else if ( activationDesc.mode == Activation_Gelu && xDesc.dtype == SG_DTYPE_FP16)
    {
        assert(xDesc.ndims == yDesc.ndims);
        sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
        sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
        assert(xdtype == ydtype);

        sg_api_gelu_forward_t api;
        api.x_global_addr =  (unsigned long long)x;
        api.y_global_addr =  (unsigned long long)y;
        api.dim = xDesc.ndims;
        api.dtype = xdtype;

        for (int i=0; i<xDesc.ndims; ++i)
        {
            assert(xDesc.shape[i] == yDesc.shape[i] );
            api.shape[i] = xDesc.shape[i];
        }

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_gelu_forward", &api, sizeof(api));
        return BM_SUCCESS;
    }
    else
    {
        sg_active_type_t active_type =  tpu_active_type_convert(activationDesc.mode) ;

        unsigned long long input    = (unsigned long long)x;
        unsigned long long output   = (unsigned long long)y;

        assert(xDesc.ndims == yDesc.ndims);

        sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
        sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
        assert(xdtype == ydtype);

        sg_api_active_forward_t api;
        api.in_global_addr = input;
        api.out_global_addr = output;
        api.shape_dim = xDesc.ndims;
        api.dtype = xdtype;
        api.active_type = active_type;

        for (int i=0; i<xDesc.ndims; ++i)
        {
            assert(xDesc.shape[i] == yDesc.shape[i] );
            api.shape[i] = xDesc.shape[i];
        }

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_active_forward", &api, sizeof(api));
        return BM_SUCCESS;
    }

    return BM_ERR_NOFEATURE;
}

bm_status_t sgdnn_activation_backward_cudnn(
    bm_handle_t                      handle,
    ActivationDescriptor_t           activationDesc,
    const void                      *alpha,
    const TensorDescriptor_t         yDesc,
    const void                      *y,
    const TensorDescriptor_t         dyDesc,
    const void                      *dy,
    const TensorDescriptor_t         xDesc,
    const void                      *x,
    const void                      *beta,
    const TensorDescriptor_t         dxDesc,
    void                            *dx)
{
    assert(activationDesc.mode == Activation_Relu);

    if(activationDesc.mode == Activation_Relu)
    {
        unsigned long long input         = (unsigned long long)x;
        unsigned long long grad_output   = (unsigned long long)dy;
        unsigned long long grad_input    = (unsigned long long)dx;

        float alpha_ = ((float*)alpha)[0];
        assert(alpha_ == 1.0f);
        float beta_ = ((float*)beta)[0];
        assert(beta_ == 0.0f || beta_ == 1.0f);

        float upper_limit = activationDesc.coef;
        //TODO: clipped_relu_backward
        assert(upper_limit==0);

        assert(dyDesc.ndims == 4);
        assert( xDesc.ndims == 4);
        assert(dxDesc.ndims == 4);
        int n = xDesc.shape[0];
        int c = xDesc.shape[1];
        int h = xDesc.shape[2];
        int w = xDesc.shape[3];

        sg_data_type_t dydtype = (sg_data_type_t)(dyDesc.dtype);
        sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
        sg_data_type_t dxdtype = (sg_data_type_t)(dxDesc.dtype);
        assert(dydtype == xdtype && xdtype == dxdtype);

        sg_api_relu_backward_t api = {
            input,
            grad_output,
            grad_input,
            {n, c, h, w},
            xdtype};

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_relu_backward", &api, sizeof(api));

        return BM_SUCCESS;
    }
    return BM_ERR_NOFEATURE;
}

bm_status_t sgdnn_cross_entropy_forward(
    bm_handle_t        handle,
    bm_device_mem_t    input,
    bm_device_mem_t    target,
    bm_device_mem_t    loss,
    int                batch,
    int                cls_num,
    int                reduction,
    sg_data_type_t     dtype)
{

    bm_device_mem_t input_mem, target_mem;
    bm_device_mem_t loss_mem;
    u64 size = (u64)batch * cls_num * dtype_size(dtype);

    DEVICE_MEM_NEW_INPUT(handle, input, size, input_mem);
    DEVICE_MEM_NEW_INPUT(handle, target, size, target_mem);
    DEVICE_MEM_NEW_OUTPUT(handle, loss, dtype_size(dtype), loss_mem);

    sg_api_crossentropy_forward_t api = {
        bm_mem_get_device_addr(input_mem),
        bm_mem_get_device_addr(target_mem),
        bm_mem_get_device_addr(loss_mem),
        batch, cls_num, reduction, dtype};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_cross_entropy_forward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, target, target_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, loss, loss_mem);

    return BM_SUCCESS;
}

bm_status_t sgdnn_cross_entropy_forward_cudnn(
    bm_handle_t                      handle,
    SoftmaxMode_t                    softmax_mode,
    CrossEntropyMode_t               crossentropy_mode,
    const void                      *alpha,
    const TensorDescriptor_t         xDesc,
    const void                      *x,
    const void                      *beta,
    const TensorDescriptor_t         yDesc,
    void                            *y,
    const TensorDescriptor_t         labelDesc,
    void                            *label)
{
    unsigned long long input  = (unsigned long long)x;
    unsigned long long target = (unsigned long long)label;
    unsigned long long loss   = (unsigned long long)y;

    float alpha_ = ((float*)alpha)[0];
    assert(alpha_ == 1.0f);
    float beta_ = ((float*)beta)[0];
    assert(beta_ == 0.0f);

    assert( yDesc.ndims == 1 && yDesc.shape[0] == 1);
    assert( xDesc.ndims == 2);
    assert( labelDesc.ndims == 2);
    int batch = xDesc.shape[0];
    int cls_num = xDesc.shape[1];
    int reduction = (int)crossentropy_mode;

    sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
    sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
    assert(xdtype == ydtype && xdtype == 0);

    sg_api_crossentropy_forward_t api = {
        input, target, loss,
        batch, cls_num, reduction, xdtype};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_cross_entropy_forward", &api, sizeof(api));

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_cross_entropy_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, target, target_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);

    return BM_SUCCESS;
}

bm_status_t sgdnn_cross_entropy_backward_cudnn(
    bm_handle_t                      handle,
    SoftmaxMode_t                    softmax_mode,
    CrossEntropyMode_t               crossentropy_mode,
    const void                      *alpha,
    const TensorDescriptor_t         xDesc,
    const void                      *xData,
    const TensorDescriptor_t         labelDesc,
    void                            *label,
    const void                      *beta,
    const TensorDescriptor_t         dxDesc,
    void                            *dx)
{
    float alpha_ = ((float*)alpha)[0];
    assert(alpha_ == 1.0f);
    float beta_ = ((float*)beta)[0];
    assert(beta_ == 0.0f);

    unsigned long long input        = (unsigned long long)xData;
    unsigned long long target       = (unsigned long long)label;
    unsigned long long grad_input   = (unsigned long long)dx;

    assert(    xDesc.ndims == 2);
    assert(   dxDesc.ndims == 2);
    assert(labelDesc.ndims == 2);
    int batch = xDesc.shape[0];
    int cls_num = xDesc.shape[1];
    int reduction = (int)crossentropy_mode;
    assert(reduction == 0 || reduction == 1);

    sg_data_type_t dxdtype = (sg_data_type_t)(dxDesc.dtype);
    sg_data_type_t  xdtype = (sg_data_type_t)( xDesc.dtype);
    assert(xdtype == dxdtype && xdtype == 0);

    sg_api_crossentropy_backward_t api = {
        input, target, grad_input,
        batch, cls_num, reduction, xdtype};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_cross_entropy_backward", &api, sizeof(api));

    return BM_SUCCESS;
}

//y = cast(x)
bm_status_t sgdnn_dtype_convert(
    bm_handle_t                      handle,
    const TensorDescriptor_t         xDesc,
    const void                      *xData,
    const TensorDescriptor_t         yDesc,
    const void                      *yData,
    sg_round_mode_t                  round_mode
 ) {
    assert(xDesc.ndims == yDesc.ndims);
    for (int idx = 0; idx < xDesc.ndims; idx++) {
        assert(xDesc.shape[idx] == yDesc.shape[idx]);
    }

    sg_api_dtype_convert_t api;
    api.input_global_addr = (unsigned long long)xData;
    api.output_global_addr = (unsigned long long)yData;
    memcpy(api.shape, xDesc.shape, xDesc.ndims * sizeof(int));
    api.dims = xDesc.ndims;
    api.idtype = (sg_data_type_t)xDesc.dtype;
    api.odtype = (sg_data_type_t)yDesc.dtype;
    api.round_mode = (sg_round_mode_t)round_mode;

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &api, sizeof(api));

    return BM_SUCCESS;
}

//y = x -> 32ic or y = x -> 32oc
bm_status_t sgdnn_conv_weight_reorder(
    bm_handle_t                      handle,
    const TensorDescriptor_t         xDesc,
    const void                      *xData,
    const TensorDescriptor_t         yDesc,
    const void                      *yData,
    ConvWeightReorderMode_t          reorder_mode
 ) {

    assert(xDesc.ndims == 4 && yDesc.ndims == 4);

    int n = xDesc.shape[0];
    int c = xDesc.shape[1];
    int h = xDesc.shape[2];
    int w = xDesc.shape[3];

    sg_api_conv_weight_reorder_t api = {
        (unsigned long long)xData,
        (unsigned long long)yData,
        {n, c, h, w},
        reorder_mode};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_weight_reorder", &api, sizeof(api));
    return BM_SUCCESS;
}

bm_status_t sgdnn_general_matmul(
    bm_handle_t                      handle,
    const TensorDescriptor_t         LDesc,
    const void                      *L,
    const TensorDescriptor_t         RDesc,
    const void                      *R,
    const TensorDescriptor_t         YDesc,
    void                            *Y,
    int                              R_transpose,
    sg_data_type_t                   compute_type)
{
    assert(LDesc.ndims == 2 && RDesc.ndims == 2 && YDesc.ndims == 2);

    int L_row = LDesc.shape[0];
    int L_col = LDesc.shape[1];
    int R_col = RDesc.shape[1];

    sg_data_type_t Ldtype = (sg_data_type_t)(LDesc.dtype);
    sg_data_type_t Rdtype = (sg_data_type_t)(RDesc.dtype);
    sg_data_type_t Ydtype = (sg_data_type_t)(YDesc.dtype);
    assert(Ldtype == Rdtype && Ldtype == Ydtype);

    if( Ldtype != compute_type )
    {
        int datasize = dtype_size(compute_type);

        bm_device_mem_t L_cast, R_cast, Y_cast;
        u64 L_cast_size = (u64)L_row * L_col * datasize;
        u64 R_cast_size = (u64)L_col * R_col * datasize;
        u64 Y_cast_size = (u64)L_row * R_col * datasize;

        DEVICE_MEM_NEW_BUFFER(handle, L_cast, L_cast_size);
        DEVICE_MEM_NEW_BUFFER(handle, R_cast, R_cast_size);
        DEVICE_MEM_NEW_BUFFER(handle, Y_cast, Y_cast_size);

        sg_api_dtype_convert_t cast_L_api;
        cast_L_api.input_global_addr = (unsigned long long)L;
        cast_L_api.output_global_addr = bm_mem_get_device_addr(L_cast);
        cast_L_api.dims = LDesc.ndims;
        cast_L_api.idtype = Ldtype;//SG_DTYPE_FP32;
        cast_L_api.odtype = compute_type;//SG_DTYPE_FP16;
        cast_L_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_L_api.shape, LDesc.shape, LDesc.ndims * sizeof(int));

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_L_api, sizeof(cast_L_api));

        sg_api_dtype_convert_t cast_R_api;
        cast_R_api.input_global_addr = (unsigned long long)R;
        cast_R_api.output_global_addr = bm_mem_get_device_addr(R_cast);
        cast_R_api.dims = RDesc.ndims;
        cast_R_api.idtype = Rdtype;//SG_DTYPE_FP32;
        cast_R_api.odtype = compute_type;//SG_DTYPE_FP16;
        cast_R_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_R_api.shape, RDesc.shape, RDesc.ndims * sizeof(int));

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_R_api, sizeof(cast_R_api));

        sg_api_general_matmul_t api = {
            bm_mem_get_device_addr(L_cast),
            bm_mem_get_device_addr(R_cast),
            bm_mem_get_device_addr(Y_cast),
            bm_mem_get_device_addr(Y_cast),
            L_row,
            L_col,
            R_col,
            R_transpose,
            0,
            compute_type};
        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_general_matmul", &api, sizeof(api));

        sg_api_dtype_convert_t cast_Y_api;
        cast_Y_api.input_global_addr = bm_mem_get_device_addr(Y_cast);
        cast_Y_api.output_global_addr = (unsigned long long)Y;
        cast_Y_api.dims = YDesc.ndims;
        cast_Y_api.idtype = compute_type;//SG_DTYPE_FP16;
        cast_Y_api.odtype = Ydtype;//SG_DTYPE_FP32;
        cast_Y_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_Y_api.shape, YDesc.shape, YDesc.ndims * sizeof(int));

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_Y_api, sizeof(cast_Y_api));

        bm_free_device(handle, L_cast);
        bm_free_device(handle, R_cast);
        bm_free_device(handle, Y_cast);
    }
    else
    {
        sg_api_general_matmul_t api = {
            (unsigned long long)L,
            (unsigned long long)R,
            (unsigned long long)Y,
            (unsigned long long)Y,
            L_row,
            L_col,
            R_col,
            R_transpose,
            0,
            Ldtype};
        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_general_matmul", &api, sizeof(api));
    }

    return BM_SUCCESS;
}

bm_status_t sgdnn_batch_matmul(
    bm_handle_t                      handle,
    const TensorDescriptor_t         LDesc,
    const void                      *L,
    const TensorDescriptor_t         RDesc,
    const void                      *R,
    const TensorDescriptor_t         YDesc,
    void                            *Y,
    int                              L_transpose,
    int                              R_transpose,
    sg_data_type_t                   compute_type)
{
    assert(LDesc.ndims == 3 && RDesc.ndims == 3 && YDesc.ndims == 3);

    assert(LDesc.shape[0] == RDesc.shape[0]);
    assert(LDesc.shape[0] == YDesc.shape[0]);
    assert(LDesc.shape[2] == RDesc.shape[1]);

    int batch_num = LDesc.shape[0];
    int L_row = LDesc.shape[1];
    int L_col = LDesc.shape[2];
    int R_col = RDesc.shape[2];

    sg_data_type_t Ldtype = (sg_data_type_t)(LDesc.dtype);
    sg_data_type_t Rdtype = (sg_data_type_t)(RDesc.dtype);
    sg_data_type_t Ydtype = (sg_data_type_t)(YDesc.dtype);

    if( (Ldtype==Rdtype||Rdtype==Ydtype) && (Ldtype != compute_type))
    {
        int datasize = dtype_size(compute_type);

        bm_device_mem_t L_cast, R_cast, Y_cast;
        u64 L_cast_size = (u64)batch_num * L_row * L_col * datasize;
        u64 R_cast_size = (u64)batch_num * L_col * R_col * datasize;
        u64 Y_cast_size = (u64)batch_num * L_row * R_col * datasize;

        DEVICE_MEM_NEW_BUFFER(handle, L_cast, L_cast_size);
        DEVICE_MEM_NEW_BUFFER(handle, R_cast, R_cast_size);
        DEVICE_MEM_NEW_BUFFER(handle, Y_cast, Y_cast_size);

        sg_api_dtype_convert_t cast_L_api;
        cast_L_api.input_global_addr = (unsigned long long)L;
        cast_L_api.output_global_addr = bm_mem_get_device_addr(L_cast);
        cast_L_api.dims = LDesc.ndims;
        cast_L_api.idtype = Ldtype;//SG_DTYPE_FP32;
        cast_L_api.odtype = compute_type;//SG_DTYPE_FP16;
        cast_L_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_L_api.shape, LDesc.shape, LDesc.ndims * sizeof(int));

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_L_api, sizeof(cast_L_api));

        sg_api_dtype_convert_t cast_R_api;
        cast_R_api.input_global_addr = (unsigned long long)R;
        cast_R_api.output_global_addr = bm_mem_get_device_addr(R_cast);
        cast_R_api.dims = RDesc.ndims;
        cast_R_api.idtype = Rdtype;//SG_DTYPE_FP32;
        cast_R_api.odtype = compute_type;//SG_DTYPE_FP16;
        cast_R_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_R_api.shape, RDesc.shape, RDesc.ndims * sizeof(int));

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_R_api, sizeof(cast_R_api));

        sg_api_batch_matmul_t api = {
            bm_mem_get_device_addr(L_cast),
            bm_mem_get_device_addr(R_cast),
            bm_mem_get_device_addr(Y_cast),
            batch_num,
            L_row,
            L_col,
            R_col,
            L_transpose,
            R_transpose,
            compute_type,
            compute_type,
            compute_type};

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batch_matmul", &api, sizeof(api));

        sg_api_dtype_convert_t cast_Y_api;
        cast_Y_api.input_global_addr = bm_mem_get_device_addr(Y_cast);
        cast_Y_api.output_global_addr = (unsigned long long)Y;
        cast_Y_api.dims = YDesc.ndims;
        cast_Y_api.idtype = compute_type;//SG_DTYPE_FP16;
        cast_Y_api.odtype = Ydtype;//SG_DTYPE_FP32;
        cast_Y_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_Y_api.shape, YDesc.shape, YDesc.ndims * sizeof(int));

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_Y_api, sizeof(cast_Y_api));

        bm_free_device(handle, L_cast);
        bm_free_device(handle, R_cast);
        bm_free_device(handle, Y_cast);
    }
    else
    {
        sg_api_batch_matmul_t api = {
            (unsigned long long)L,
            (unsigned long long)R,
            (unsigned long long)Y,
            batch_num,
            L_row,
            L_col,
            R_col,
            L_transpose,
            R_transpose,
            Ldtype,
            Rdtype,
            Ydtype};

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batch_matmul", &api, sizeof(api));
    }
    return BM_SUCCESS;
}

bm_status_t sgdnn_linear(
    bm_handle_t                      handle,
    const TensorDescriptor_t         LDesc,
    const void                      *L,
    const TensorDescriptor_t         RDesc,
    const void                      *R,
    const TensorDescriptor_t         BDesc,
    void                            *B,
    const TensorDescriptor_t         YDesc,
    void                            *Y,
    int                              R_transpose,
    sg_data_type_t                   compute_type)
{
    assert(LDesc.ndims == 2 && RDesc.ndims == 2 && YDesc.ndims == 2);
    assert(BDesc.ndims == 1);

    int L_row = LDesc.shape[0];
    int L_col = LDesc.shape[1];
    int R_col = RDesc.shape[1];
    assert(BDesc.shape[0] == R_col);

    sg_data_type_t Ldtype = (sg_data_type_t)(LDesc.dtype);
    sg_data_type_t Rdtype = (sg_data_type_t)(RDesc.dtype);
    sg_data_type_t Bdtype = (sg_data_type_t)(BDesc.dtype);
    sg_data_type_t Ydtype = (sg_data_type_t)(YDesc.dtype);
    assert(Ldtype == Rdtype && Ldtype == Bdtype && Ldtype == Ydtype);

    if( Ldtype != compute_type )
    {
        int datasize = dtype_size(compute_type);

        bm_device_mem_t L_cast, R_cast, B_cast, Y_cast;
        u64 L_cast_size = (u64)L_row * L_col * datasize;
        u64 R_cast_size = (u64)L_col * R_col * datasize;
        u64 B_cast_size = (u64)R_col * datasize;
        u64 Y_cast_size = (u64)L_row * R_col * datasize;

        DEVICE_MEM_NEW_BUFFER(handle, L_cast, L_cast_size);
        DEVICE_MEM_NEW_BUFFER(handle, R_cast, R_cast_size);
        DEVICE_MEM_NEW_BUFFER(handle, B_cast, B_cast_size);
        DEVICE_MEM_NEW_BUFFER(handle, Y_cast, Y_cast_size);

        sg_api_dtype_convert_t cast_L_api;
        cast_L_api.input_global_addr = (unsigned long long)L;
        cast_L_api.output_global_addr = bm_mem_get_device_addr(L_cast);
        cast_L_api.dims = LDesc.ndims;
        cast_L_api.idtype = Ldtype;
        cast_L_api.odtype = compute_type;
        cast_L_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_L_api.shape, LDesc.shape, LDesc.ndims * sizeof(int));
        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_L_api, sizeof(cast_L_api));

        sg_api_dtype_convert_t cast_R_api;
        cast_R_api.input_global_addr = (unsigned long long)R;
        cast_R_api.output_global_addr = bm_mem_get_device_addr(R_cast);
        cast_R_api.dims = RDesc.ndims;
        cast_R_api.idtype = Rdtype;
        cast_R_api.odtype = compute_type;
        cast_R_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_R_api.shape, RDesc.shape, RDesc.ndims * sizeof(int));
        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_R_api, sizeof(cast_R_api));

        sg_api_dtype_convert_t cast_B_api;
        cast_B_api.input_global_addr = (unsigned long long)B;
        cast_B_api.output_global_addr = bm_mem_get_device_addr(B_cast);
        cast_B_api.dims = BDesc.ndims;
        cast_B_api.idtype = Bdtype;
        cast_B_api.odtype = compute_type;
        cast_B_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_B_api.shape, BDesc.shape, BDesc.ndims * sizeof(int));
        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_B_api, sizeof(cast_B_api));

        sg_api_general_matmul_t api = {
            bm_mem_get_device_addr(L_cast),
            bm_mem_get_device_addr(R_cast),
            bm_mem_get_device_addr(B_cast),
            bm_mem_get_device_addr(Y_cast),
            L_row,
            L_col,
            R_col,
            R_transpose,
            1,
            compute_type};
        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_general_matmul", &api, sizeof(api));

        sg_api_dtype_convert_t cast_Y_api;
        cast_Y_api.input_global_addr = bm_mem_get_device_addr(Y_cast);
        cast_Y_api.output_global_addr = (unsigned long long)Y;
        cast_Y_api.dims = YDesc.ndims;
        cast_Y_api.idtype = compute_type;
        cast_Y_api.odtype = Ydtype;
        cast_Y_api.round_mode = SG_ROUND_EVEN;
        memcpy(cast_Y_api.shape, YDesc.shape, YDesc.ndims * sizeof(int));

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_dtype_convert", &cast_Y_api, sizeof(cast_Y_api));

        bm_free_device(handle, L_cast);
        bm_free_device(handle, R_cast);
        bm_free_device(handle, B_cast);
        bm_free_device(handle, Y_cast);
    }
    else
    {
        sg_api_general_matmul_t api = {
            (unsigned long long)L,
            (unsigned long long)R,
            (unsigned long long)B,
            (unsigned long long)Y,
            L_row,
            L_col,
            R_col,
            R_transpose,
            1,
            Ldtype};
        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_general_matmul", &api, sizeof(api));
    }

    return BM_SUCCESS;
}

bm_status_t sgdnn_softmax_forward_cudnn(
    bm_handle_t                      handle,
    int                              dim,
    const void                      *alpha,
    const TensorDescriptor_t         xDesc,
    const void                      *x,
    const void                      *beta,
    const TensorDescriptor_t         yDesc,
    void                            *y)
{
    assert ( *( ( float * ) alpha ) == 1.f );
    assert ( *( ( float * ) beta ) == 0.f );
    sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
    sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
    assert(xdtype == ydtype);
    assert(xDesc.ndims == yDesc.ndims);
    for (int i = 0; i < xDesc.ndims; ++i)
    {
        assert(xDesc.shape[i] == yDesc.shape[i]);
    }

    if ( xdtype == SG_DTYPE_FP32 || xdtype == SG_DTYPE_FP16)
    {
        sg_api_softmax_forward_t api;
        api.input_global_addr = (unsigned long long)x;
        api.output_global_addr = (unsigned long long)y;
        for (int i=0; i<xDesc.ndims; ++i)
        {
            api.shape[i] = xDesc.shape[i];
        }
        api.dims = xDesc.ndims;
        api.compute_dim = dim;
        api.scale_val = 1.f;
        api.dtype = xdtype;

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_softmax_forward", &api, sizeof(api));
    }
    else
    {
        assert(false);
    }
    return BM_SUCCESS;
}

bm_status_t sgdnn_softmax_backward_cudnn(
    bm_handle_t                      handle,
    int                              dim,
    const TensorDescriptor_t         yDesc,
    const void                      *y,
    const TensorDescriptor_t         dyDesc,
    const void                      *dy,
    const TensorDescriptor_t         dxDesc,
    void                            *dx)
{
    sg_data_type_t ydtype  = (sg_data_type_t)( yDesc.dtype);
    sg_data_type_t dydtype = (sg_data_type_t)(dyDesc.dtype);
    sg_data_type_t dxdtype = (sg_data_type_t)(dxDesc.dtype);
    assert(ydtype == dydtype && dydtype == dxdtype);

    assert(yDesc.ndims == dyDesc.ndims && dyDesc.ndims == dxDesc.ndims && yDesc.ndims == 4);
    assert(dim == 3);
    for (int i = 0; i < yDesc.ndims; ++i)
    {
        assert(yDesc.shape[i] == dyDesc.shape[i]);
        assert(yDesc.shape[i] == dxDesc.shape[i]);
    }

    sg_api_softmax_backward_t api;
    api.output_global_addr      = (unsigned long long)y;
    api.grad_output_global_addr = (unsigned long long)dy;
    api.grad_input_global_addr  = (unsigned long long)dx;
    api.input_n = yDesc.shape[0];
    api.input_c = yDesc.shape[1];
    api.input_h = yDesc.shape[2];
    api.input_w = yDesc.shape[3];
    api.dim = dim;
    api.dtype = ydtype;

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_softmax_backward", &api, sizeof(api));
    return BM_SUCCESS;
}

bm_status_t sgdnn_transpose(
    bm_handle_t                      handle,
    const TensorDescriptor_t         xDesc,
    const void                      *xData,
    const TensorDescriptor_t         yDesc,
    void                            *yData)
{
    assert(xDesc.ndims == yDesc.ndims && xDesc.ndims == 2);
    assert(xDesc.shape[0] == yDesc.shape[1]);
    assert(xDesc.shape[1] == yDesc.shape[0]);

    sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
    assert(xdtype == ydtype);

    sg_api_transpose_t api;
    api.input_global_mem_addr = (unsigned long long)xData;
    api.output_global_mem_addr = (unsigned long long)yData;
    api.buffer_global_mem_addr = 0;
    api.dims = 2;
    api.sgdtype = xdtype;
    api.input_shape[0] = xDesc.shape[0];
    api.input_shape[1] = xDesc.shape[1];
    api.order[0] = 1;
    api.order[1] = 0;

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_transpose", &api, sizeof(api));
    return BM_SUCCESS;
}

bm_status_t sgdnn_permute(
    bm_handle_t                      handle,
    const TensorDescriptor_t         xDesc,
    const void                      *xData,
    const TensorDescriptor_t         yDesc,
    void                            *yData,
    const int                       *order)
{
    assert(xDesc.ndims == yDesc.ndims);
    int dims = xDesc.ndims;

    sg_api_transpose_t api;
    int x_size = 1, y_size = 1;
    int change_num = 0;
    for(int i=0; i<dims; ++i)
    {
        x_size *= xDesc.shape[i];
        y_size *= yDesc.shape[i];
        api.input_shape[i] = xDesc.shape[i];
        api.order[i] = order[i];
        change_num += order[i]!=i;
    }
    assert(x_size == y_size);

    sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
    assert(xdtype == ydtype);

    int step_num = (change_num-1)/2+1;
    u64 buffer_size = (step_num>1) * x_size * dtype_size(xdtype);
    bm_device_mem_t buffer_mem;
    if (buffer_size > 0) {
        DEVICE_MEM_NEW_BUFFER(handle, buffer_mem, buffer_size);
    }

    api.input_global_mem_addr = (unsigned long long)xData;
    api.output_global_mem_addr = (unsigned long long)yData;
    api.buffer_global_mem_addr = bm_mem_get_device_addr(buffer_mem);
    api.dims = dims;
    api.sgdtype = xdtype;

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_transpose", &api, sizeof(api));

    if (buffer_size > 0) {
        bm_free_device(handle, buffer_mem);
    }
    return BM_SUCCESS;
}

bm_status_t sgdnn_gelu_backward_cudnn(
    bm_handle_t                     handle,
    const TensorDescriptor_t        xDesc,
    const void                     *x,
    const TensorDescriptor_t        dyDesc,
    const void                     *dy,
    const TensorDescriptor_t        dxDesc,
    void                           *dx)
{
    assert(xDesc.ndims == dxDesc.ndims && xDesc.ndims == dyDesc.ndims);
    sg_data_type_t dxdtype = (sg_data_type_t)(dxDesc.dtype);
    sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t dydtype = (sg_data_type_t)(dyDesc.dtype);
    assert(dxdtype == xdtype && xdtype == dydtype);

    sg_api_gelu_backward_t api;
    api.dx_global_addr = (unsigned long long)dx;
    api.dy_global_addr = (unsigned long long)dy;
    api.x_global_addr = (unsigned long long)x;
    api.dim = xDesc.ndims;
    api.dtype = xdtype;

    for (int i=0; i<xDesc.ndims; ++i)
    {
        assert(xDesc.shape[i] == dyDesc.shape[i] );
        assert(xDesc.shape[i] == dxDesc.shape[i] );
        api.shape[i] = xDesc.shape[i];
    }

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_gelu_backward", &api, sizeof(api));
    return BM_SUCCESS;
}

bm_status_t sgdnn_reduce_sum_cudnn(
    bm_handle_t                     handle,
    const TensorDescriptor_t        xDesc,
    const void                     *x,
    const TensorDescriptor_t        yDesc,
    void                           *y,
    int                             reduce_dim,
    int                             keep_dim)
{
    if (keep_dim)
    {
        assert (xDesc.ndims == yDesc.ndims);
    }
    else
    {
        assert (xDesc.ndims == yDesc.ndims + 1);
    }
    sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
    assert (xdtype == ydtype);
    if (reduce_dim < 0)
    {
        reduce_dim += xDesc.ndims;
    }
     sg_api_reduce_sum_t api;
    int y_shape[FW_MAX_SHAPE_DIMS];
    if (keep_dim)
    {
         for (int i = 0; i < yDesc.ndims; ++i)
         {
             y_shape[i] = yDesc.shape[i];
         }
    }
    else
    {
        for (int i = 0; i < reduce_dim; ++i)
        {
            y_shape[i] = yDesc.shape[i];
        }
        y_shape[reduce_dim] = 1;
        for (int i = reduce_dim; i < yDesc.ndims; ++i)
        {
            y_shape[i + 1] = yDesc.shape[i];
        }
    }
    for (int i = 0; i < xDesc.ndims; ++i)
    {
        if (i != reduce_dim)
        {
            assert(xDesc.shape[i] == y_shape[i]);
        }
        else
        {
            assert(y_shape[i] == 1);
        }
        api.shape[i] = xDesc.shape[i];
    }

    api.shape_dim = xDesc.ndims;
    api.input_global_addr = (unsigned long long)x;
    api.output_global_addr = (unsigned long long)y;
    api.reduce_dim = reduce_dim;
    api.dtype = xdtype;

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_reduce_sum", &api, sizeof(api));
    return BM_SUCCESS;
}
bm_status_t sgdnn_strided_copy_cudnn(
    bm_handle_t                     handle,
    const TensorDescriptor_t        srcDesc,
    const void                      *src,
    const TensorDescriptor_t        dstDesc,
    void                            *dst)
{
    sg_data_type_t srcDtype = (sg_data_type_t)(srcDesc.dtype);
    sg_data_type_t dstDtype = (sg_data_type_t)(srcDesc.dtype);
    assert(srcDtype == dstDtype);
    assert(srcDesc.ndims == dstDesc.ndims);

    sg_api_strided_copy_t api;
    api.dtype = srcDtype;
    api.shape_dim = srcDesc.ndims;
    api.in_global_addr = (unsigned long long) src;
    api.out_global_addr = (unsigned long long) dst;

    for (int i = 0; i < srcDesc.ndims; i++){
        assert(srcDesc.shape[i] == dstDesc.shape[i]);
        api.shape[i]      = srcDesc.shape[i];
        api.in_stride[i]  = srcDesc.stride[i];
        api.out_stride[i] = dstDesc.stride[i];
     }
    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_strided_copy", &api, sizeof(api));

    return BM_SUCCESS;
}

bm_status_t sgdnn_where (
    bm_handle_t                     handle,
    const TensorDescriptor_t        condDesc,
    const void                     *cond,
    const TensorDescriptor_t        selfDesc,
    const void                     *self,
    const TensorDescriptor_t        otherDesc,
    const void                     *other,
    const TensorDescriptor_t        outDesc,
    void                           *out)
{
    sg_data_type_t condDtype = (sg_data_type_t)(condDesc.dtype);
    sg_data_type_t selfDtype = (sg_data_type_t)(selfDesc.dtype);
    sg_data_type_t otherDtype = (sg_data_type_t)(otherDesc.dtype);
    sg_data_type_t outDtype = (sg_data_type_t)(outDesc.dtype);
    assert(selfDtype == otherDtype);
    assert(selfDtype == outDtype);
    assert(condDesc.ndims > 0);
    assert(condDesc.ndims == outDesc.ndims);
    if (selfDesc.ndims > 0)
    {
        assert(selfDesc.ndims == outDesc.ndims);
    }
    if (otherDesc.ndims > 0)
    {
        assert(otherDesc.ndims == outDesc.ndims);
    }
    sg_api_where_t api;
    api.cond_global_addr = (unsigned long long)cond;
    if (selfDesc.ndims > 0)
    {
        api.self_is_scalar = false;
        api.self_global_addr = (unsigned long long)self;
    }
    else
    {
        api.self_is_scalar = true;
        memcpy ( &api.self_global_addr, self, dtype_size(selfDtype) );
    }
    if (otherDesc.ndims > 0)
    {
        api.other_is_scalar = false;
        api.other_global_addr = (unsigned long long)other;
    }
    else
    {
        api.other_is_scalar = true;
        memcpy ( &api.other_global_addr, other, dtype_size(otherDtype) );
    }
    api.out_global_addr = (unsigned long long)out;
    api.cond_dtype = condDtype;
    api.dtype = selfDtype;
    api.shape_dim = selfDesc.ndims;
    for (int i = 0; i < outDesc.ndims; i++) {
        api.cond_shape[i] = condDesc.shape[i];
        api.out_shape[i] = outDesc.shape[i];
        if (selfDesc.ndims > 0)
        {
           api.self_shape[i] = selfDesc.shape[i];
        }
        if (otherDesc.ndims > 0)
        {
            api.other_shape[i] = otherDesc.shape[i];
        }
     }
    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_where", &api, sizeof(api));
    return BM_SUCCESS;
}

bm_status_t sgdnn_concat (
    bm_handle_t                     handle,
    const TensorDescriptor_t       *inputDescs,
    const void * const *            inputs,
    int                             input_num,
    const TensorDescriptor_t        outputDesc,
    void                           *output,
    int                             concat_dim)
{
    sg_data_type_t outDtype = (sg_data_type_t)(outputDesc.dtype);
    if (concat_dim < 0)
    {
        concat_dim += outputDesc.ndims;
    }
    int concat_dim_shape = 0;
    for (int i = 0; i < input_num; ++i)
    {
        sg_data_type_t inDtype = (sg_data_type_t)(inputDescs[i].dtype);
        assert(inDtype == outDtype);
        assert(inputDescs[i].ndims == outputDesc.ndims);
        for (int j = 0; j < outputDesc.ndims; ++j)
        {
            if (j != concat_dim)
            {
                assert(inputDescs[i].shape[j] == outputDesc.shape[j]);
            }
            else
            {
                concat_dim_shape += inputDescs[i].shape[j];
            }
        }
    }
    assert(concat_dim_shape == outputDesc.shape[concat_dim]);
    sg_api_concat_t api;
    for (int i = 0; i < input_num; ++i)
    {
        api.input_global_addrs[i] = (unsigned long long)inputs[i];
        for (int j = 0; j < inputDescs[i].ndims; ++j)
        {
            api.input_shapes[i][j] = inputDescs[i].shape[j];
        }
    }
    api.output_global_addr = (unsigned long long)output;
    api.input_num = input_num;
    api.shape_dim = outputDesc.ndims;
    api.concat_dim = concat_dim;
    api.dtype = outDtype;
    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_concat", &api, sizeof(api));
    return BM_SUCCESS;
}

bm_status_t sgdnn_index_select_cudnn(
    bm_handle_t                     handle,
    const TensorDescriptor_t        tableDesc,
    const void                      *table,
    const TensorDescriptor_t        indexDesc,
    const void                      *index,
    const TensorDescriptor_t        outDesc,
    void                            *out,
    int                             dim){
    sg_api_index_select_t api;
    api.input_global_addr     = (unsigned long long) table;
    api.index_global_addr     = (unsigned long long) index;
    api.output_global_addr    = (unsigned long long) out;
    api.shape_dims            = tableDesc.ndims;
    api.index_num             = 1;
    api.axis                  = dim;
    api.const_val             = 0;
    api.dtype                 = (sg_data_type_t)tableDesc.dtype;
    for(int i = 0; i < tableDesc.ndims; i++ ){
        api.input_shape[i] = tableDesc.shape[i];
    }
    for (int i = 0; i < indexDesc.ndims; i++){
        api.index_num *= indexDesc.shape[i];
    }
    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_index_select", &api, sizeof(api));
    return BM_SUCCESS;
}

bm_status_t sgdnn_const_fill_cudnn(
    bm_handle_t                     handle,
    const TensorDescriptor_t        srcDesc,
    void                           *src,
    const void*                     fill_value){
    sg_api_constant_fill_t api;
    api.out_global_addr       = (unsigned long long) src;
    api.shape_dim             = srcDesc.ndims;
    api.filled_sgdtype        = (sg_data_type_t)srcDesc.dtype;
    api.filled_value          = 0;

    memcpy( &api.filled_value, fill_value, dtype_size(api.filled_sgdtype) );
        for(int i = 0; i < srcDesc.ndims; i++ ){
        api.shape[i] = srcDesc.shape[i];
    }
    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_const_fill", &api, sizeof(api));
    return BM_SUCCESS;
} 

bm_status_t sgdnn_sqrt(
    bm_handle_t                     handle,
    const TensorDescriptor_t        inputDesc,
    const void                     *input,
    const TensorDescriptor_t        outputDesc,
    void                           *output )
{
    sg_api_sqrt_t api;
    api.input_global_addr = (unsigned long long) input;
    api.output_global_addr = (unsigned long long) output;
    api.dim = inputDesc.ndims;
    api.dtype = (sg_data_type_t)inputDesc.dtype;
    assert(inputDesc.ndims == outputDesc.ndims);
    assert(inputDesc.dtype == outputDesc.dtype);

    for(int i = 0; i < inputDesc.ndims; i++ ){
        assert(inputDesc.shape[i] == outputDesc.shape[i]);
        api.shape[i] = inputDesc.shape[i];
    }
    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_sqrt", &api, sizeof(api));
    return BM_SUCCESS;
}

bm_status_t sgdnn_addcdiv(
    bm_handle_t                     handle,
    const TensorDescriptor_t        inputDesc,
    const void                     *input,
    const TensorDescriptor_t        tensor1Desc,
    const void                     *tensor1,
    const TensorDescriptor_t        tensor2Desc,
    const void                     *tensor2,
    const TensorDescriptor_t        outputDesc,
    void                           *output,
    double                          value)
{
    sg_api_addcdiv_t api;
    api.input_global_addr = (unsigned long long) input;
    api.tensor1_global_addr = (unsigned long long) tensor1;
    api.tensor2_global_addr = (unsigned long long) tensor2;
    api.output_global_addr = (unsigned long long) output;
    api.dim = inputDesc.ndims;
    api.dtype = (sg_data_type_t)inputDesc.dtype;
    api.value = (float) value;
    assert(inputDesc.ndims == outputDesc.ndims);
    assert(inputDesc.ndims == tensor1Desc.ndims);
    assert(inputDesc.ndims == tensor2Desc.ndims);
    assert(inputDesc.dtype == outputDesc.dtype);
    assert(inputDesc.dtype == tensor1Desc.dtype);
    assert(inputDesc.dtype == tensor2Desc.dtype);

    for(int i = 0; i < inputDesc.ndims; i++ ){
        assert(inputDesc.shape[i] == outputDesc.shape[i]);
        assert(inputDesc.shape[i] == tensor1Desc.shape[i]);
        assert(inputDesc.shape[i] == tensor2Desc.shape[i]);
        api.shape[i] = inputDesc.shape[i];
    }
    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_addcdiv", &api, sizeof(api));
    return BM_SUCCESS;
}
