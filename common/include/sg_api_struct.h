#ifndef SG_API_STRUCT_H
#define SG_API_STRUCT_H

#pragma pack(push, 1)
#include "common_def.h"

#ifndef FW_MAX_SHAPE_DIMS
#define FW_MAX_SHAPE_DIMS      8
#endif
#define MAX_CONCATLAYER_INPUT_NUM 10

typedef struct {
    unsigned long long in_global_addr;
    unsigned long long out_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int shape_dim;
    sg_data_type_t dtype;
    sg_active_type_t active_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_active_forward_t;
#else
} sg_api_active_forward_t;
#endif

typedef struct {
    unsigned long long  input_global_addr;
    unsigned long long  weight_global_addr;
    unsigned long long  grad_output_global_addr;
    unsigned long long  grad_input_global_addr;
    unsigned long long  grad_weight_global_addr;
    unsigned long long  grad_bias_global_addr;
    unsigned long long  buffer_global_addr;
    int                 input_shape[4];
    int                 output_shape[4];
    int                 groups;
    int                 kernel[2];
    int                 stride[2];
    int                 dilation[2];
    int                 pad[4];
    int                 grad_input_enable;
    int                 grad_weight_enable;
    int                 grad_bias_enable;
    sg_data_type_t      dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_backward_t;
#else
} sg_api_conv_backward_t;
#endif

typedef struct{
    unsigned long long grad_output_global_addr;
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long saved_mean_global_addr;
    unsigned long long saved_invstd_global_addr;
    unsigned long long grad_input_global_addr;
    unsigned long long grad_weight_global_addr;
    unsigned long long grad_bias_global_addr;
    int                shape[4];
    int                grad_input_enable;
    int                grad_weight_enable;
    int                grad_bias_enable;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_batchnorm_backward_t;
#else
} sg_api_batchnorm_backward_t;
#endif

typedef struct {
    unsigned long long grad_output_global_addr;
    unsigned long long grad_input_global_addr;
    int                input_shape[4];
    int                output_shape[4];
    int                kernel[2];
    int                stride[2];
    int                pad[2];
    int                ceil_mode;
    int                count_include_pad;
    int                divisor_override;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_avgpool_backward_t;
#else
} sg_api_avgpool_backward_t;
#endif

typedef struct {
    unsigned long long forward_input_global_addr;
    unsigned long long forward_output_global_addr;
    unsigned long long grad_output_global_addr;
    unsigned long long grad_input_global_addr;
    int                input_shape[4];
    int                output_shape[4];
    int                kernel[2];
    int                stride[2];
    int                pad[2];
    int                dilation[2];
    int                ceil_mode;
    sg_data_type_t     data_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_maxpool_backward_t;
#else
} sg_api_maxpool_backward_t;
#endif

typedef struct{
    unsigned long long inputA_global_addr;
    unsigned long long inputB_global_addr;
    unsigned long long grad_output_global_addr;
    unsigned long long grad_inputA_global_addr;
    unsigned long long grad_inputB_global_addr;
    int                shape[4];
    int                op_code;
    int                coeff_a;
    int                coeff_b;
    int                grad_input_a_enable;
    int                grad_input_b_enable;
    sg_data_type_t     idtype;//dtype for input && grad_input
    sg_data_type_t     odtype;//dtype for output && grad_output
#ifndef WIN32
} __attribute__((packed)) sg_api_eltwise_backward_t;
#else
} sg_api_eltwise_backward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long grad_output_global_addr;
    unsigned long long grad_input_global_addr;
    int                shape[4];
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_relu_backward_t;
#else
} sg_api_relu_backward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long target_global_addr;
    unsigned long long grad_input_global_addr;
    int                batch;
    int                cls_num;
    int                reduction;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_crossentropy_backward_t;
#else
} sg_api_crossentropy_backward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    int ishape[4];
    int groups;
    int output_c;
    int kernel[2];
    int stride[2];
    int dilation[2];
    int pad[4];
    int has_bias;
    int if_relu;
    float upper_limit;
    int result_add;
    sg_data_type_t idtype;
    sg_data_type_t odtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_forward_t;
#else
} sg_api_conv_forward_t;
#endif

typedef struct{
    unsigned long long input_global_addr;
    unsigned long long running_mean_global_addr;
    unsigned long long running_var_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long updated_mean_global_addr;
    unsigned long long updated_var_global_addr;
    unsigned long long batch_mean_global_addr;
    unsigned long long batch_invstd_global_addr;
    unsigned long long output_global_addr;
    int                shape[4];
    float              momentum;
    float              eps;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_batchnorm_forward_t;
#else
} sg_api_batchnorm_forward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    unsigned long long mean_global_addr;
    unsigned long long rstd_global_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    int                axis;
    float              eps;
    int                affine;
    int                save_stat;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_layernorm_forward_t;
#else
} sg_api_layernorm_forward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    unsigned long long max_mask_global_addr;
    int input_n;
    int input_c;
    int input_h;
    int input_w;
    int output_h;
    int output_w;
    int kh;
    int kw;
    int pad_h;
    int pad_w;
    int pad_h_after;
    int pad_w_after;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int is_avg_pooling;
    int avg_pooling_mode;
    int max_with_mask;
    int if_relu;
    float relu_upper_limit;
    sg_data_type_t   data_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_pooling_forward_t;
#else
} sg_api_pooling_forward_t;
#endif

typedef struct{
    unsigned long long inputA_global_addr;
    unsigned long long inputB_global_addr;
    unsigned long long output_global_addr;
    unsigned long long mask_global_addr;
    int                input_num;
    int                tensor_n;
    int                tensor_c;
    int                tensor_h;
    int                tensor_w;
    int                op_code;
    int                coeff_A;
    int                coeff_B;
    int                need_mask;
    int                mask_index_A;
    int                mask_index_B;
    int                if_relu;
    sg_data_type_t     idtype;//dtype for inputA&&inputB
    sg_data_type_t     odtype;//dtype for output
#ifndef WIN32
} __attribute__((packed)) sg_api_eltwise_forward_t;
#else
} sg_api_eltwise_forward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int                shape[4];
    float              upper_limit;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_relu_forward_t;
#else
} sg_api_relu_forward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long target_global_addr;
    unsigned long long loss_global_addr;
    int                batch;
    int                cls_num;
    int                reduction;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_crossentropy_forward_t;
#else
} sg_api_crossentropy_forward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    sg_data_type_t     idtype;
    sg_data_type_t     odtype;
    sg_round_mode_t    round_mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_dtype_convert_t;
#else
} sg_api_dtype_convert_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int                shape[4];
    int                reorder_mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_weight_reorder_t;
#else
} sg_api_conv_weight_reorder_t;
#endif

typedef struct {
    unsigned long long A_global_addr;
    unsigned long long B_global_addr;
    unsigned long long res_global_addr;
    int A_shape[FW_MAX_SHAPE_DIMS];
    int B_shape[FW_MAX_SHAPE_DIMS];
    int A_dims;
    int B_dims;
    sg_data_type_t dtype;
    sg_binary_type_t binary_type;
#ifndef WIN32
} __attribute__((packed)) sg_api_bcbinary_float_t;
#else
} sg_api_bcbinary_float_t;
#endif

typedef struct {
    unsigned long long input_addr;
    unsigned long long output_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dims;
    sg_binary_type_t binary_type;
    sg_data_type_t dtype;
    float const_value;
    int is_inversed;
#ifndef WIN32
} __attribute__((packed)) sg_api_const_binary_float_t;
#else
} sg_api_const_binary_float_t;
#endif

typedef struct {
    unsigned long long L_global_addr;
    unsigned long long R_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long Y_global_addr;
    int                L_row_num;
    int                L_col_num;
    int                R_col_num;
    int                R_transpose;
    int                have_bias;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_general_matmul_t;
#else
} sg_api_general_matmul_t;
#endif

typedef struct {
    unsigned long long L_addr;
    unsigned long long R_addr;
    unsigned long long Y_addr;
    unsigned long long B_addr;
    int                batch_num;
    int                L_row_num;
    int                L_col_num;
    int                R_col_num;
    int                L_trans;
    int                R_trans;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_batch_matmul_t;
#else
} sg_api_batch_matmul_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int                shape[FW_MAX_SHAPE_DIMS];
    int                dims;
    int                compute_dim;
    float              scale_val;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_softmax_forward_t;
#else
} sg_api_softmax_forward_t;
#endif

typedef struct {
    unsigned long long output_global_addr;
    unsigned long long grad_output_global_addr;
    unsigned long long grad_input_global_addr;
    int                input_n;
    int                input_c;
    int                input_h;
    int                input_w;
    int                dim;
    sg_data_type_t     dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_softmax_backward_t;
#else
} sg_api_softmax_backward_t;
#endif

typedef struct {
  unsigned long long input_global_mem_addr;
  unsigned long long output_global_mem_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int order[FW_MAX_SHAPE_DIMS];
  int dims;
  unsigned long long buffer_global_mem_addr;
  sg_data_type_t     sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_transpose_t;
#else
} sg_api_transpose_t;
#endif

typedef struct {
    unsigned long long x_global_addr;
    unsigned long long y_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dim;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_gelu_forward_t;
#else
} sg_api_active_forward_t;
#endif

typedef struct {
    unsigned long long dx_global_addr;
    unsigned long long dy_global_addr;
    unsigned long long x_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dim;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_gelu_backward_t;
#else
} sg_api_gelu_backward_t;
#endif

typedef struct {
    unsigned long long in_global_addr;
    unsigned long long out_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int stride[FW_MAX_SHAPE_DIMS];
    int shape_dim;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_contiguous_forward_t;
#else
} sg_api_contiguous_forward_t;
#endif

typedef struct {
    unsigned long long in_global_addr;
    unsigned long long out_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int in_stride[FW_MAX_SHAPE_DIMS];
    int out_stride[FW_MAX_SHAPE_DIMS];
    int shape_dim;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_strided_copy_t;
#else
} sg_api_strided_copy_t;
#endif

typedef struct {
    unsigned long long cond_global_addr;
    unsigned long long self_global_addr;
    unsigned long long other_global_addr;
    unsigned long long out_global_addr;
    int self_is_scalar;
    int other_is_scalar;
    int cond_shape[FW_MAX_SHAPE_DIMS];
    int self_shape[FW_MAX_SHAPE_DIMS];
    int other_shape[FW_MAX_SHAPE_DIMS];
    int out_shape[FW_MAX_SHAPE_DIMS];
    int shape_dim;
    sg_data_type_t cond_dtype;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_where_t;
#else
} sg_api_where_t;
#endif

typedef struct {
    unsigned long long input_global_addrs[MAX_CONCATLAYER_INPUT_NUM];
    unsigned long long output_global_addr;
    int input_shapes[MAX_CONCATLAYER_INPUT_NUM][FW_MAX_SHAPE_DIMS];
    int input_num;
    int shape_dim;
    int concat_dim;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_concat_t;
#else
} sg_api_concat_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int shape_dim;
    int reduce_dim_start;
    int reduce_dim_end;
    sg_data_type_t dtype;
    int reduction_mode;
#ifndef WIN32
} __attribute__((packed)) sg_api_reduce_t;
#else
} sg_api_reduce_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long index_global_addr;
    unsigned long long output_global_addr;
    int input_shape[8];
    int shape_dims;
    int index_num;
    int axis;
    int const_val;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_index_select_t;
#else
} sg_api_index_select_t;
#endif

typedef struct {
    unsigned long long  out_global_addr;
    int                 shape[FW_MAX_SHAPE_DIMS];
    int                 shape_dim;
    unsigned int        filled_value;
    sg_data_type_t      filled_sgdtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_constant_fill_t;
#else
} sg_api_constant_fill_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long output_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dim;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_sqrt_t;
#else
} sg_api_sqrt_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long tensor1_global_addr;
    unsigned long long tensor2_global_addr;
    unsigned long long output_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dim;
    float value;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_addcdiv_t;
#else
} sg_api_addcdiv_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long tensor1_global_addr;
    unsigned long long tensor2_global_addr;
    unsigned long long output_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dim;
    float value;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_addcmul_t;
#else
} sg_api_addcmul_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long target_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long output_global_addr;
    int batch_num;
    int class_num;
    int reduction;
    float label_smoothing;
    sg_data_type_t dtype;
    int target_is_int64;
#ifndef WIN32
} __attribute__((packed)) sg_api_cross_entropy_loss_forward_t;
#else
} sg_api_cross_entropy_loss_forward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long target_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long grad_output_global_addr;
    unsigned long long grad_input_global_addr;
    int batch_num;
    int class_num;
    int reduction;
    float label_smoothing;
    sg_data_type_t dtype;
    int target_is_int64;
#ifndef WIN32
} __attribute__((packed)) sg_api_cross_entropy_loss_backward_t;
#else
} sg_api_cross_entropy_loss_backward_t;
#endif

typedef struct {
    unsigned long long gradout_global_addr;
    unsigned long long index_global_addr;
    unsigned long long output_global_addr;
    unsigned long long sorted_index_global_addr;
    unsigned long long sorted_index_index_global_addr;
    unsigned long long from_index_global_addr;
    unsigned long long to_index_global_addr;
    int gradout_shape[FW_MAX_SHAPE_DIMS];
    int gradout_dim;
    int idx_shape[FW_MAX_SHAPE_DIMS];
    int idx_dim;
    int out_shape[FW_MAX_SHAPE_DIMS];
    int out_dim;
    int window_size;
    sg_data_type_t grad_dtype;
    sg_data_type_t idx_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_emb_backward_t;
#else
} sg_api_emb_backward_t;
#endif

typedef struct {
    unsigned long long input_global_addr;
    unsigned long long other_global_addr;
    unsigned long long output_global_addr;
    int shape[FW_MAX_SHAPE_DIMS];
    int dim;
    float value;
    sg_data_type_t dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_scale_add_t;
#else
} sg_api_scale_add_t;
#endif

#pragma pack(pop)
#endif  // SG_API_STRUCT_H
