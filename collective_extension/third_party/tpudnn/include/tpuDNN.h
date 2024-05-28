#pragma once

#include "common_def.h"
#include <cstdint>

extern "C"
{

    enum tpudnnStatus_t
    {
        TPUDNN_STATUS_SUCCESS,
        TPUDNN_STATUS_FAILED
    };

    typedef void *tpudnnHandle_t;

    tpudnnHandle_t tpudnnCreate(int deviceID = 0);
    void tpudnnDestroy(tpudnnHandle_t handle);

    void *tpudnnPhysToVirt(tpudnnHandle_t handle, uint64_t addr);
    uint64_t tpudnnVirtToPhys(tpudnnHandle_t handle, void *addr);

    // TODO
    // tpudnnHandle_t handle_from_bmlib();
    // tpudnnHandle_t handle_from_tpurt();

    tpudnnStatus_t tpudnnActive(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int shape_dim,
        sg_active_type_t active_type,
        const float *coeff,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnPooling(
        tpudnnHandle_t handle,
        // input
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int output_h,
        int output_w,
        int kh,
        int kw,
        int pad_h,
        int pad_w,
        int pad_h_after,
        int pad_w_after,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int is_avg_pooling,
        int avg_pooling_mode,
        int max_mask,
        int if_relu,
        float relu_upper_limit,
        sg_data_type_t dtype,
        // output
        void *output,
        void *max_mask_addr);

    tpudnnStatus_t tpudnnPoolingFix8b(
        tpudnnHandle_t handle,
        // input
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int kh,
        int kw,
        int pad_h_top,
        int pad_h_bottom,
        int pad_w_left,
        int pad_w_right,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int is_avg_pooling,
        int avg_pooling_mode,
        int ceil_mode,
        sg_data_type_t output_dtype,
        sg_data_type_t input_dtype,
        // output
        void *output);

    tpudnnStatus_t tpudnnAdaptivePool(
        tpudnnHandle_t handle,
        void *input_mem,
        const int *input_shape,
        int input_dims,
        const int oh,
        const int ow,
        const int mode,
        void *output_mem,
        sg_data_type_t sgdtype);

    tpudnnStatus_t tpudnnInterpForward(
        tpudnnHandle_t handle,
        // input
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int output_h,
        int output_w,
        int pad_bag,
        int pad_end,
        bool align_corners,
        bool half_pixel_centers,
        PLATFORM_SUPPORT platform_sp,
        sg_data_type_t dtype,
        // output
        void *output);

    tpudnnStatus_t tpudnnScaleForward(
        tpudnnHandle_t handle,
        // input
        void *input,
        void *scale,
        void *bias,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int axis,     // scale begin axis
        int axis_num, // scale axis num
        int has_bias,
        int if_relu,
        float relu_upper_limit,
        sg_data_type_t dtype,
        // output
        void *output);

    tpudnnStatus_t tpudnnShift(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int shape_dim,
        int shift_axis,
        sg_shift_type_t shift_dir,
        int shift_num,
        sg_data_type_t in_dtype);

    tpudnnStatus_t tpudnnShuffleChannel(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int shape_dim,
        int group,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnSoftmax(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        int input_n,
        int input_c,
        int input_inner_dim,
        float scale_val,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnSoftmaxTfliteFix8b(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        void *table,
        int input_n,
        int input_c,
        int input_inner_dim,
        int zero_point,
        float scale_val,
        sg_data_type_t input_dtype,
        sg_data_type_t output_dtype);

    tpudnnStatus_t tpudnnSort(
        tpudnnHandle_t handle,
        void *src_data,
        void *src_idx,
        void *dst_data,
        void *dst_idx,
        int len,
        int topk,
        bool is_descend,
        bool idx_en,
        bool auto_idx,
        int dtype);

    tpudnnStatus_t tpudnnSparseConv3d(
        tpudnnHandle_t handle,
        void *input_features_global_addr,
        void *input_coor_global_addr,
        void *weight_global_addr,
        void *origin_input_shape_global_addr,
        void *intermedia_mem_pool_global_addr,
        void *intermedia_mem_pool_ex_global_addr,
        void *output_features_global_addr,
        void *output_coor_global_addr,
        void *origin_output_shape_global_addr,
        void *debug_pool_global_addr,
        int case_num,
        int batch_num,
        int limit_active_out_num,
        int ndim,
        int output_channel,
        int input_channel,
        int kz, // kernel
        int ky,
        int kx,
        int sz, // stride
        int sy,
        int sx,
        int pz, // padding
        int py,
        int px,
        int dz, // dilation
        int dy,
        int dx,
        unsigned long long input_feature_sz,
        unsigned long long input_coor_sz,
        unsigned long long weight_sz,
        unsigned long long origin_input_shape_sz,
        unsigned long long intermedia_mem_pool_sz,
        unsigned long long intermedia_mem_pool_ex_sz,
        unsigned long long output_feature_sz,
        unsigned long long output_coor_sz,
        unsigned long long origin_output_shape_sz,
        unsigned long long debug_sz,
        int has_bias,
        int subm,
        int opz, // output padding
        int opy,
        int opx,
        sg_data_type_t feature_dtype);

    tpudnnStatus_t tpudnnSplitForward(
        tpudnnHandle_t handle,
        // input
        void *input,
        const int *input_shape,
        int input_dim,
        int split_axis,
        const int *split_size,
        int split_num,
        sg_data_type_t dtype,
        // output
        void **output,
        int test_glb);

    tpudnnStatus_t tpudnnSortPerDim(
        tpudnnHandle_t handle,
        void *input_data,
        void *output_data,
        void *output_index,
        const int *input_shape,
        int input_dims,
        int sort_dim,
        bool is_argsort,
        bool stable,
        bool descending,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnSsdDetectOut(tpudnnHandle_t handle,
                                      void *location,
                                      void *confidence,
                                      void *prior,
                                      int batch_num,
                                      int num_prior,
                                      int num_classes,
                                      int num_loc_classes,
                                      int share_location,
                                      int background_label_id,
                                      int top_k,
                                      int code_type,
                                      int keep_top_k,
                                      int variance_encoded_in_target,
                                      float nms_threshold,
                                      float conf_threshold,
                                      float eta,
                                      int onnx_nms,
                                      void *output);

    tpudnnStatus_t tpudnnStrideSlice(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *input_shape,
        int shape_dim,
        int begin_mask,
        int end_mask,
        const int *begin_index,
        const int *end_index,
        const int *strides,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnTileForward(
        tpudnnHandle_t handle,
        // input
        void *input,
        const int *input_shape,
        const int *tile_coeff,
        int input_dim,
        int type,
        sg_data_type_t dtype,
        // output
        void *output,
        int test_glb);

    tpudnnStatus_t tpudnnTiuLoopTest(
        tpudnnHandle_t handle,
        const void *input_data,
        void *output_data,
        sg_data_type_t dtype,
        int full_local_mem_size,
        int loop_num,
        int save_last,
        int test_power);

    tpudnnStatus_t tpudnnTopk(
        tpudnnHandle_t handle,
        void *input_data,
        void *input_index,
        void *output_data,
        void *output_index,
        bool input_index_valid,
        int k,
        int descending,
        int batchs,
        int batch_num,
        int batch_stride,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnTpuFullTest(
        tpudnnHandle_t handle,
        const void *input_data,
        void *output_data,
        int full_local_mem_size,
        int loop_num,
        int max_run_num,
        unsigned long long disable_mask);

    tpudnnStatus_t tpudnnTranspose(
        tpudnnHandle_t handle,
        void *input_mem,
        int *input_shape,
        int *order,
        int input_dims,
        void *output_mem,
        sg_data_type_t sgdtype);

    tpudnnStatus_t tpudnnTriangularize(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        int *shape,
        int dims,
        int is_upper,
        int diagonal,
        sg_data_type_t dtype);

    tpudnnStatus_t sgdnn_upsample(
        tpudnnHandle_t handle,
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int size,
        int if_relu,
        sg_data_type_t dtype,
        void *output);

    tpudnnStatus_t tpudnnActiveMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int dims,
        sg_active_type_t active_type,
        const float *coeff,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnMatmulMultiCore(
        tpudnnHandle_t handle,
        void *left,
        void *right,
        void *output,
        const int *L_shape,
        const int *R_shape,
        const int L_dims,
        const int R_dims,
        const int L_trans,
        const int R_trans,
        sg_data_type_t in_dtype,
        sg_data_type_t out_dtype,
        int slice_m_core,
        int slice_n_core,
        int slice_m,
        int slice_n,
        int slice_k,
        const int left_slice_dim,
        const int right_slice_dim,
        const int result_slice_dim,
        const int left_8ch_buf_size,
        const int right_8ch_buf_size,
        const int result_8ch_buf_size,
        char *left_8ch_buf[8],
        char *right_8ch_buf[8],
        char *result_8ch_buf[8]);

    tpudnnStatus_t tpudnnRmsnormForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *output,
        int *shape,
        int dims,
        int axis,
        float partial,
        float eps,
        int with_weight,
        int with_bias,
        sg_data_type_t dtype,
        const int enable_8ch,
        const int input_slice_dim,
        const int weight_slice_dim,
        const int bias_slice_dim,
        const int output_slice_dim,
        const int input_buffer_size,
        const int weight_buffer_size,
        const int bias_buffer_size,
        const int output_buffer_size,
        char **input_8ch_buffer,
        char **weight_8ch_buffer,
        char **bias_8ch_buffer,
        char **output_8ch_buffer);

    // tpudnnStatus_t tpudnnGroupNormMultiCore(
    //     tpudnnHandle_t handle,
    //     void * input,
    //     void * weight,
    //     void * bias,
    //     void * output,
    //     const int *shape,
    //     int dims,
    //     int axis,
    //     int group_num,
    //     float eps,
    //     int affine,
    //     sg_data_type_t dtype);

    typedef struct
    {
        void *data;
        unsigned long long addr;
        unsigned int size;
    } io_mem_t;

    typedef struct
    {
        void *sys_mem_tpu_cmd;
        void *sys_mem_gdma_cmd;
        void *sys_mem_hau_cmd;
        void *sys_mem_sdma_cmd;
        void *sys_mem_imm_buf;
        void *sys_mem_pio_buffer;
        int tpu_cmd_num;
        int gdma_cmd_num;
        int hau_cmd_num;
        int sdma_cmd_num;
        unsigned int tpu_cmd_size;
        unsigned int gdma_cmd_size;
        unsigned int hau_cmd_size;
        unsigned int sdma_cmd_size;
        unsigned int imm_buf_size;
        unsigned int pio_buffer_size;
    } system_msg_sync_core_param_t;
    typedef struct
    {
        system_msg_sync_core_param_t core_param[8];
        int loop;
        int enable_pio_des_interleave;
        int core_num;
        io_mem_t input[20];  // Align with test_msg_sync_multi_core MAX_IO_NUM
        io_mem_t output[20]; // Align with test_msg_sync_multi_core MAX_IO_NUM
        int input_num;
        int output_num;
        unsigned long long total_io_size;
        void *placeholder;
    } system_msg_sync_multi_core_param_t;

    tpudnnStatus_t tpudnnMsg_sync_multi_core(
        tpudnnHandle_t handle,
        system_msg_sync_multi_core_param_t system_msg_sync_param);

    tpudnnStatus_t tpudnnLlamaMultiCore(
        tpudnnHandle_t handle,
        int loop);
    tpudnnStatus_t tpudnnCdmaMsgCentralTestMultiCore(
        tpudnnHandle_t handle);
    tpudnnStatus_t tpudnnMsgCentralMultiCore(
        tpudnnHandle_t handle,
        void *blob_A_0,
        void *blob_B_0,
        void *blob_T_0,
        void *blob_A_1,
        void *blob_B_1,
        void *blob_T_1,
        int op,
        int n,
        int c,
        int h,
        int w,
        int test_core_idx0,
        int test_core_idx1);

    tpudnnStatus_t tpudnnMgmMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight0,
        void *weight1,
        void *bias0,
        void *bias1,
        void *output,
        const int *in_shape,
        const int *w0_shape,
        const int *w1_shape,
        int in_dims,
        int w0_dims,
        int w1_dims,
        sg_data_type_t in_dtype,
        sg_data_type_t out_dtype,
        int has_bias,
        bool use_fast);

    tpudnnStatus_t tpudnnMgmBackwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight0,
        void *weight1,
        void *mat0,
        void *gelu,
        void *grad_output,
        void *grad_weight0,
        void *grad_weight1,
        void *grad_bias0,
        void *grad_bias1,
        void *grad_mat0,
        void *grad_input,
        const int *in_shape,
        const int *w0_shape,
        const int *w1_shape,
        int in_dims,
        int w0_dims,
        int w1_dims,
        int has_bias,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnMlp0FuseMultiCore(
        tpudnnHandle_t handle,
        void *input0,
        void *input1,
        void *gamma,
        void *beta,
        void *weight,
        void *bias,
        void *output,
        void *gelu_output,
        void *norm_output,
        void *norm_mean,
        void *norm_rstd,
        const int *in_shape,
        const int *w_shape,
        int in_dims,
        int w_dims,
        sg_data_type_t dtype,
        float eps,
        int has_bias,
        bool use_fast);

    tpudnnStatus_t tpudnnLayernormMatmulFuseMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *gamma,
        void *beta,
        void *weight,
        void *bias,
        void *output,
        void *norm_mean,
        void *norm_rstd,
        const int *in_shape,
        const int *w_shape,
        int in_dims,
        int w_dims,
        sg_data_type_t dtype,
        float eps,
        int has_bias);

    tpudnnStatus_t tpudnnSoftmaxWhereBackwardFuseMultiCore(
        tpudnnHandle_t handle,
        void *grad_output,
        void *softmax_output,
        void *cond,
        void *grad_input,
        const int *in_shape,
        const int *cond_shape,
        int in_dims,
        sg_data_type_t dtype,
        float value);

    tpudnnStatus_t tpudnnWhereMultiCore(
        tpudnnHandle_t handle,
        void *output,
        void *cond,
        void *self,
        void *other,
        float self_val,
        float other_val,
        const int *out_shape,
        const int *cond_shape,
        const int *self_shape,
        const int *other_shape,
        int dim,
        bool self_is_scalar,
        bool other_is_scalar,
        sg_data_type_t cond_dtype,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnCrossEntropyMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *target,
        void *weight,
        void *output,
        void *sum,
        void *max,
        int batch_num,
        int class_num,
        int reduction,
        float label_smoothing,
        sg_data_type_t dtype,
        bool target_is_int64);

    tpudnnStatus_t tpudnnCrossEntropyBackwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *target,
        void *weight,
        void *grad_output,
        void *sum,
        void *max,
        void *grad_input,
        int batch_num,
        int class_num,
        int reduction,
        float label_smoothing,
        sg_data_type_t dtype,
        bool target_is_int64);

    tpudnnStatus_t tpudnnMatmulBackwardMultiCore(
        tpudnnHandle_t handle,
        void *left,
        void *right,
        void *grad_out,
        void *grad_left,
        void *grad_right,
        const int *L_shape,        // left && grad_left shape
        const int *R_shape,        // right && grad_right shape
        const int *Y_shape,        // out && grad_out shape
        const int L_dims,          // left && grad_left dims
        const int R_dims,          // right && grad_right dims
        const int Y_dims,          // out && grad_out dims
        sg_data_type_t in_dtype,   // fwd l&r data type
        sg_data_type_t out_dtype); // fwd out data type

    tpudnnStatus_t tpudnnGptQkvMultiCore(
        tpudnnHandle_t handle,
        void *Q,
        void *K,
        void *V,
        void *Y,
        void *where_cond,
        float C,
        float where_other_val,
        const int batch,
        const int N,
        const int d,
        sg_data_type_t dtype,
        sg_data_type_t where_cond_dtype);

    tpudnnStatus_t tpudnnGeluForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int dims,
        const float *coeff,
        sg_data_type_t dtype,
        int input_slice_dim,
        int output_slice_dim,
        int input_8ch_buf_size,
        int output_8ch_buf_size,
        char *input_8ch_buf[8],
        char *output_8ch_buf[8]);

    tpudnnStatus_t tpudnnPoolingFp8(
        tpudnnHandle_t handle,
        // input
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int output_h,
        int output_w,
        int kh,
        int kw,
        int pad_h,
        int pad_w,
        int pad_h_after,
        int pad_w_after,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int is_avg_pooling,
        int avg_pooling_mode,
        float re_scale,
        sg_data_type_t output_dtype,
        sg_data_type_t input_dtype,
        // output
        void *output);

    tpudnnStatus_t tpudnnGeluBackwardMultiCore(
        tpudnnHandle_t handle,
        void *grad_input,
        void *grad_output,
        void *input,
        const int *shape,
        const int dims,
        sg_data_type_t dtype,
        int input_slice_dim,
        int output_slice_dim,
        int input_8ch_buf_size,
        int output_8ch_buf_size,
        char *grad_input_8ch_buf[8],
        char *grad_output_8ch_buf[8],
        char *input_output_8ch_buf[8]);

    tpudnnStatus_t tpudnnSoftmaxForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        const int dims,
        int begin_dim,
        int end_dim,
        float scale_val,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnSoftmaxBackwardMultiCore(
        tpudnnHandle_t handle,
        void *grad_input,
        void *grad_output,
        void *output,
        int n,
        int c,
        int h,
        int w,
        int axis,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnLayernormForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *output,
        void *mean,
        void *rstd,
        int *shape,
        int dims,
        int axis,
        float eps,
        int affine,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnLayernormBackwardMultiCore(
        tpudnnHandle_t handle,
        void *grad_input,
        void *grad_weight,
        void *grad_bias,
        void *grad_output,
        void *input,
        void *weight,
        void *mean,
        void *rstd,
        int *shape,
        int dims,
        int axis,
        int affine,
        int requires_grad_input,
        sg_data_type_t dtype,
        int input_slice_dim,
        int output_slice_dim,
        int input_8ch_buf_size,
        int output_8ch_buf_size,
        char *grad_input_8ch_buf[8],
        char *grad_output_8ch_buf[8],
        char *input_8ch_buf[8]);

    tpudnnStatus_t tpudnnAdamBackwardMultiCore(
        tpudnnHandle_t handle,
        void *weight_out,
        void *m_out,
        void *v_out,
        void *vmax_out,
        void *grad_weight,
        void *weight_in,
        void *m_in,
        void *v_in,
        void *vmax_in,
        void *t,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        bool amsgrad,
        bool maximize,
        int *shape,
        int dims,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnLlamaMlpForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight0,
        void *weight1,
        void *weight2,
        void *output,
        int batch,
        int input_w,
        int middle_w,
        sg_data_type_t dtype,
        const int enable_8ch,
        const int input_slice_dim,
        const int weight0_slice_dim,
        const int weight1_slice_dim,
        const int weight2_slice_dim,
        const int output_slice_dim,
        const int input_buffer_size,
        const int weight0_buffer_size,
        const int weight1_buffer_size,
        const int weight2_buffer_size,
        const int output_buffer_size,
        char **input_8ch_buffer,
        char **weight0_8ch_buffer,
        char **weight1_8ch_buffer,
        char **weight2_8ch_buffer,
        char **output_8ch_buffer);

    tpudnnStatus_t tpudnnBinaryFloatMultiCore(
        tpudnnHandle_t handle,
        void *input_A,
        void *input_B,
        void *output,
        const int *A_shape,
        const int *B_shape,
        int A_dim,
        int B_dim,
        int binary_type,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnBinaryBackwardMultiCore(
        tpudnnHandle_t handle,
        void *input_A,
        void *input_B,
        void *grad_output,
        void *grad_input_A,
        void *grad_input_B,
        const int *A_shape,
        const int *B_shape,
        int A_dim,
        int B_dim,
        int binary_type,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnConstBinaryFloatMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int dims,
        sg_data_type_t dtype,
        sg_binary_type_t binary_type,
        float const_value,
        int is_inversed);

    tpudnnStatus_t tpudnnConstBinaryFloatBackwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *grad_output,
        void *grad_input,
        const int *shape,
        int dims,
        sg_data_type_t dtype,
        sg_binary_type_t binary_type,
        float const_value,
        int is_inversed);

    tpudnnStatus_t tpudnnDropoutMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        const int dims,
        const float drop_rate,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnMatmulAllReduceMultiCore(
        tpudnnHandle_t handle,
        void *left,
        void *right,
        void *output,
        int op_code,
        const int *L_shape,
        const int *R_shape,
        const int L_dims,
        const int R_dims,
        const int L_trans,
        const int R_trans,
        sg_data_type_t in_dtype,
        sg_data_type_t out_dtype);

    tpudnnStatus_t tpudnnLlama2QkvMultiCore(
        tpudnnHandle_t handle,
        void *Q,
        void *K,
        void *V,
        void *Q_buffer,
        void *K_buffer,
        void *V_buffer,
        void *Kcache,
        void *Vcache,
        void *RoPE_cos,
        void *RoPE_sin,
        void *Mask,
        void *Y,
        void *Input_length,
        void *Save_slots,
        void *Fetch_slots,
        int slots_size,
        float C,
        const int batch,
        const int Mask_max,
        const int hidden_size,
        const int num_attention_heads,
        const int num_k_v_heads,
        const int embeddings,
        const int attention_mode,
        const int block_size,
        const int max_blocks,
        sg_data_type_t dtype,
        int qkv_packed);

    typedef struct
    {
        unsigned long long gdma_src_offset;
        unsigned long long gdma_dst_offset;
        unsigned long long gdma_reduce_src_offset[8];
        unsigned long long gdma_reduce_dst_offset[8];
        unsigned long long sdma_src_offset[8];
        unsigned long long sdma_dst_offset[8];
        unsigned long long cdma_src_offset[8];
        unsigned long long cdma_dst_offset[8];
        unsigned int gdma_shape[4];
        unsigned int gdma_reduce_shape[4];
        unsigned int sdma_shape[4];
        unsigned int sdma_reduce_shape[4];
        unsigned int cdma_shape[4];
        sg_data_type_t gdma_sg_dtype;
        sg_data_type_t gdma_reduce_sg_dtype;
        sg_data_type_t sdma_sg_dtype;
        sg_data_type_t sdma_reduce_sg_dtype;
        sg_data_type_t cdma_sg_dtype;
        sg_reduce_method_t gdma_sg_reduce_method;
        sg_reduce_method_t sdma_sg_reduce_method;
    } dma_k2k_stress_multi_core_param_t;

    tpudnnStatus_t tpudnnDmaK2kStressMultiCore(
        tpudnnHandle_t handle,
        dma_k2k_stress_multi_core_param_t dma_k2k_stress_multi_core_param);

#if defined(__sg2260__)
    tpudnnStatus_t tpudnnC2CAllReduce(
        tpudnnHandle_t handle,
        void *send_buff,
        void *recv_buff,
        int count,
        sg_data_type_t dtype,
        sg_reduce_method_t reduce_method,
        sccl_args_t sccl_args);

    tpudnnStatus_t tpudnnC2CReduce(
        tpudnnHandle_t handle,
        void *send_buff,
        void *recv_buff,
        int count,
        sg_data_type_t dtype,
        sg_reduce_method_t reduce_method,
        int root,
        sccl_args_t sccl_args);

    tpudnnStatus_t tpudnnC2CGather(
        tpudnnHandle_t handle,
        void *send_buff,
        int send_count,
        void *recv_buff,
        int recv_count,
        sg_data_type_t dtype,
        int root,
        sccl_args_t sccl_args);

    tpudnnStatus_t tpudnnC2CAllGather(
        tpudnnHandle_t handle,
        void *send_buff,
        int send_count,
        void *recv_buff,
        int recv_count,
        sg_data_type_t dtype,
        sccl_args_t sccl_args);

    tpudnnStatus_t tpudnnC2CBroadcast(
        tpudnnHandle_t handle,
        void *buff,
        int count,
        sg_data_type_t dtype,
        int root,
        sccl_args_t sccl_args);

    tpudnnStatus_t tpudnnC2CScatter(
        tpudnnHandle_t handle,
        void *send_mem,
        int send_count,
        sg_data_type_t send_type,
        void *recv_mem,
        int recv_count,
        sg_data_type_t recv_type,
        int root,
        sccl_args_t sccl_args);

    tpudnnStatus_t tpudnnC2CAllToAll(
        tpudnnHandle_t handle,
        void *send_mem,
        int send_count,
        sg_data_type_t send_type,
        void *recv_mem,
        int recv_count,
        sg_data_type_t recv_type,
        sccl_args_t sccl_args);

    tpudnnStatus_t tpudnnSdmaMultiThread(
        tpudnnHandle_t handle,
        void *input_data,
        void *output_data,
        void *buffer,
        unsigned int element_num,
        sg_data_type_t dtype,
        unsigned int case_);
#endif
}
