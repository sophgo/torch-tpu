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
    sg_data_type_t odtype = (sg_data_type_t)(yDesc.dtype);

    sg_data_type_t compute_type = (sg_data_type_t)(convDesc.computeType);

    if (compute_type == SG_DTYPE_FP32) {

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
            SG_DTYPE_FP32};

        sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_conv_forward", &api, sizeof(api));

        bm_free_device(handle, x_fp16);
        bm_free_device(handle, w_fp16);
        bm_free_device(handle, w_32ic_fp16);
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

    assert((xDesc.ndims == 4 && yDesc.ndims == 4) || (xDesc.ndims == 3 && yDesc.ndims == 3));
    if ( bnScale != nullptr || bnBias != nullptr || resultRunningMean != nullptr || resultRunningVariance != nullptr )
    {
      assert(bnScaleBiasMeanVarDesc.ndims == 1 );
      assert(bnScaleBiasMeanVarDesc.shape[0] == xDesc.shape[1] );
    }
    int n, c, h, w;
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
    float eps = epsilon;
    
    sg_data_type_t idtype = (sg_data_type_t)(xDesc.dtype);
    sg_data_type_t odtype = (sg_data_type_t)(yDesc.dtype);
    if ( bnScale != nullptr || bnBias != nullptr || resultRunningMean != nullptr || resultRunningVariance != nullptr )
    {
      sg_data_type_t wdtype = (sg_data_type_t)(bnScaleBiasMeanVarDesc.dtype);
      assert(idtype == wdtype && wdtype == odtype);
    }
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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batchnorm_forward_v2", &api, sizeof(api));

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batchnorm_backward", &api, sizeof(api));

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_batchnorm_backward", &api, sizeof(api));

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

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_linear_backward", &api, sizeof(api));

    DEVICE_MEM_DEL_INPUT(handle, input, input_mem);
    DEVICE_MEM_DEL_INPUT(handle, weight, weight_mem);
    DEVICE_MEM_DEL_INPUT(handle, grad_output, grad_output_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_input, grad_input_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_weight, grad_weight_mem);
    DEVICE_MEM_DEL_OUTPUT(handle, grad_bias, grad_bias_mem);

    return BM_SUCCESS;
}

bm_status_t sgdnn_relu_forward(
    bm_handle_t        handle,
    bm_device_mem_t    input,
    bm_device_mem_t    output,
    float              upper_limit,
    int                n,
    int                c,
    int                h,
    int                w, 
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
else
    {
        sg_active_type_t active_type =  tpu_active_type_convert(activationDesc.mode) ;
        
        unsigned long long input    = (unsigned long long)x;
        unsigned long long output   = (unsigned long long)y;

        assert(xDesc.ndims == yDesc.ndims);
        
        sg_data_type_t ydtype = (sg_data_type_t)(yDesc.dtype);
        sg_data_type_t xdtype = (sg_data_type_t)(xDesc.dtype);
        assert(xdtype == ydtype);

        sg_api_active_forward_t api = {
            .in_global_addr = input,
            .out_global_addr = output,
            .shape_dim = xDesc.ndims,
            .dtype = xdtype,
            .active_type = active_type};

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
    int                              R_transpose)
{
    assert(LDesc.ndims == 2 && RDesc.ndims == 2 && YDesc.ndims == 2);

    int L_row = LDesc.shape[0];
    int L_col = LDesc.shape[1];
    int R_col = RDesc.shape[1];

    sg_data_type_t Ldtype = (sg_data_type_t)(LDesc.dtype);
    sg_data_type_t Rdtype = (sg_data_type_t)(RDesc.dtype);
    sg_data_type_t Ydtype = (sg_data_type_t)(YDesc.dtype);
    assert(Ldtype == Rdtype && Ldtype == Ydtype);

    sg_api_general_matmul_t api = {
        (unsigned long long)L,
        (unsigned long long)R,
        (unsigned long long)Y,
        L_row,
        L_col,
        R_col,
        R_transpose,
        Ldtype};

    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_general_matmul", &api, sizeof(api));
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
    int                              R_transpose)
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
    return BM_SUCCESS;
}

bm_status_t sgdnn_softmax_forward_cudnn(
    bm_handle_t                      handle,
    //SoftmaxMode_t                    softmax_mode,
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
    assert(xdtype == ydtype && xdtype == 0);
    assert(xDesc.ndims == yDesc.ndims);
    for (int i = 0; i < xDesc.ndims; ++i)
    {
        assert(xDesc.shape[i] == yDesc.shape[i]);
    }
    sg_api_softmax_forward_t api;
    api.input_global_addr = (unsigned long long)x;
    api.output_global_addr = (unsigned long long)y;
    api.input_n = 1;
    for (int i = 0; i < dim; ++i)
    {
        api.input_n *= xDesc.shape[i];
    }
    api.input_c = xDesc.shape[dim];
    api.input_inner_dim = 1;
    for (int i = dim + 1; i < xDesc.ndims; ++i)
    {
        api.input_inner_dim *= xDesc.shape[i];
    }

    api.scale_val = 1.f;
    api.dtype = xdtype;
    sgdnn_tpu_kernel_launch(handle, "tpu_kernel_api_softmax_forward", &api, sizeof(api));
    return BM_SUCCESS;
}
