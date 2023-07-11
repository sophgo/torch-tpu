#ifndef SG_API_STRUCT_H
#define SG_API_STRUCT_H

#pragma pack(push, 1)

#ifndef FW_MAX_SHAPE_DIMS
#define FW_MAX_SHAPE_DIMS      8
#endif
#ifndef FW_MAX_CONCAT_NUM
#define FW_MAX_CONCAT_NUM     10
#endif

typedef struct
{
  unsigned long long  input_global_addr;
  unsigned long long  weight_global_addr;
  unsigned long long  grad_output_global_addr;
  unsigned long long  grad_input_global_addr;
  unsigned long long  grad_weight_global_addr;
  unsigned long long  grad_bias_global_addr;
  unsigned long long  buffer_global_addr;
  int input_shape[4];
  int output_shape[4];
  int groups;
  int kernel[2];
  int stride[2];
  int dilation[2];
  int pad[4];
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_conv2d_backward_t;
#else
} sg_api_conv2d_backward_t;
#endif

typedef struct
{
  unsigned long long grad_output_global_addr;
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long saved_mean_global_addr;
  unsigned long long saved_invstd_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long grad_weight_global_addr;
  unsigned long long grad_bias_global_addr;
  int shape[4];
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_batchnorm2d_backward_t;
#else
} sg_api_batchnorm2d_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_relu_backward_t;
#else
}
sg_api_relu_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  int input_shape[4];
  int groups;
  int output_c;
  int kernel[2];
  int stride[2];
  int dilation[2];
  int pad[4];
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_conv2d_t;
#else
}
sg_api_conv2d_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long running_mean_global_addr;
  unsigned long long running_var_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long saved_mean_global_addr;
  unsigned long long saved_invstd_global_addr;
  unsigned long long output_global_addr;
  int shape[4];
  float momentum;
  float eps;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_batchnorm2d_t;
#else
}
sg_api_batchnorm2d_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  float eps;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_layernorm_t;
#else
}
sg_api_layernorm_t;
#endif

typedef struct
{
  unsigned long long grad_output_global_addr;
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long grad_weight_global_addr;
  unsigned long long grad_bias_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_layernorm_backward_t;
#else
}
sg_api_layernorm_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_relu_t;
#else
}
sg_api_relu_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_dtype;
  int output_dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_dtype_convert_t;
#else
}
sg_api_dtype_convert_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[4];
  int mode;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_conv_weight_reorder_t;
#else
}
sg_api_conv_weight_reorder_t;
#endif

typedef struct
{
  unsigned long long left_global_addr;
  unsigned long long right_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  int batch;
  int left_row;
  int left_column;
  int right_column;
  int is_left_transposed;
  int is_right_transposed;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_batch_matmul_t;
#else
}
sg_api_batch_matmul_t;
#endif

typedef struct {
    unsigned long long left_global_addr;
    unsigned long long right_global_addr;
    unsigned long long bias_global_addr;
    unsigned long long output_global_addr;
    int                L_shape[FW_MAX_SHAPE_DIMS];
    int                R_shape[FW_MAX_SHAPE_DIMS];
    int                L_dims;
    int                R_dims;
    int                L_trans;
    int                R_trans;
    int                in_dtype;
    int                out_dtype;
#ifndef WIN32
} __attribute__((packed)) sg_api_matmul_multi_core_t;
#else
} sg_api_matmul_multi_core_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_softmax_t;
#else
}
sg_api_softmax_t;
#endif

typedef struct
{
  unsigned long long output_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_softmax_backward_t;
#else
}
sg_api_softmax_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_gelu_t;
#else
}
sg_api_gelu_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_gelu_backward_t;
#else
}
sg_api_gelu_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_stride[FW_MAX_SHAPE_DIMS];
  int output_stride[FW_MAX_SHAPE_DIMS];
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_strided_copy_t;
#else
}
sg_api_strided_copy_t;
#endif

typedef struct
{
  unsigned long long cond_global_addr;
  unsigned long long self_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int cond_shape[FW_MAX_SHAPE_DIMS];
  int self_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int cond_dtype;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_where_t;
#else
}
sg_api_where_t;
#endif

typedef struct
{
  unsigned long long input_global_addrs[FW_MAX_CONCAT_NUM];
  unsigned long long output_global_addr;
  int input_shapes[FW_MAX_CONCAT_NUM][FW_MAX_SHAPE_DIMS];
  int dim;
  int input_num;
  int axis;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_concat_t;
#else
}
sg_api_concat_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int start_dim;
  int end_dim;
  int dtype;
  int mode;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_reduce_t;
#else
}
sg_api_reduce_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long index_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int index_num;
  int axis;
  int dtype;
  int is_index_int64;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_index_select_t;
#else
}
sg_api_index_select_t;
#endif

typedef struct
{
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  unsigned int value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_constant_fill_t;
#else
}
sg_api_constant_fill_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_sqrt_t;
#else
}
sg_api_sqrt_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  int active_type;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_active_t;
#else
}
sg_api_active_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long tensor1_global_addr;
  unsigned long long tensor2_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_addcdiv_t;
#else
}
sg_api_addcdiv_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long tensor1_global_addr;
  unsigned long long tensor2_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_addcmul_t;
#else
}
sg_api_addcmul_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long target_global_addr;
  unsigned long long output_global_addr;
  int batch;
  int class_;
  int reduction;
  float label_smoothing;
  int dtype;
  int is_target_int64;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_cross_entropy_loss_t;
#else
}
sg_api_cross_entropy_loss_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long target_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int batch;
  int class_;
  int reduction;
  float label_smoothing;
  int dtype;
  int is_target_int64;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_cross_entropy_loss_backward_t;
#else
}
sg_api_cross_entropy_loss_backward_t;
#endif

typedef struct
{
  unsigned long long grad_output_global_addr;
  unsigned long long index_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long sorted_index_global_addr;
  unsigned long long sorted_index_index_global_addr;
  unsigned long long from_index_global_addr;
  unsigned long long to_index_global_addr;
  int grad_output_shape[FW_MAX_SHAPE_DIMS];
  int grad_output_dim;
  int index_shape[FW_MAX_SHAPE_DIMS];
  int index_dim;
  int grad_input_shape[FW_MAX_SHAPE_DIMS];
  int grad_input_dim;
  int window_size;
  int grad_output_dtype;
  int is_index_int64;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_embedding_backward_t;
#else
}
sg_api_embedding_backward_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_add_t;
#else
}
sg_api_add_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_bcast_add_t;
#else
}
sg_api_bcast_add_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_mulc_t;
#else
}
sg_api_mulc_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_addc_t;
#else
}
sg_api_addc_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_cdiv_t;
#else
}
sg_api_cdiv_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_csub_t;
#else
}
sg_api_csub_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_norm2_t;
#else
}
sg_api_norm2_t;
#endif

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  unsigned long long output_global_addr;
  int                shape[FW_MAX_SHAPE_DIMS];
  int                dims;
  int                axis;
  float              eps;
  int                affine;
  int                dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_layernorm_forward_multi_core_t;
#else
}
sg_api_layernorm_forward_multi_core_t;
#endif

typedef struct {
  unsigned long long grad_output_global_addr;
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long grad_weight_global_addr;
  unsigned long long grad_bias_global_addr;
  unsigned long long grad_weight_reduce_buffer;
  unsigned long long grad_bias_reduce_buffer;
  int                shape[FW_MAX_SHAPE_DIMS];
  int                dims;
  int                axis;
  int                affine;
  int                dtype;
#ifndef WIN32
}
__attribute__ ( ( packed ) ) sg_api_layernorm_backward_multi_core_t;
#else
}
sg_api_layernorm_backward_multi_core_t;
#endif

typedef struct {
  unsigned long long output_addr;
  unsigned long long cond_addr;
  unsigned long long self_addr;
  unsigned long long other_addr;
  int out_shape[FW_MAX_SHAPE_DIMS];
  int cond_shape[FW_MAX_SHAPE_DIMS];
  int self_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int cond_dtype;
  int dtype;
  int self_is_scalar;
  int other_is_scalar;
  float self_val;
  float other_val;
#ifndef WIN32
} __attribute__((packed)) sg_api_where_multi_core_t;
#else
} sg_api_where_multi_core_t;
#endif

typedef struct {
    int core_idx;
    int core_num;
    int core_msg_id;
    int name_len;
    int api_id;
    int api_size;
    unsigned char api_data[0];
#ifndef WIN32
} __attribute__((packed)) sg_api_core_info_t;
#else
} sg_api_core_info_t;
#endif

#pragma pack(pop)
#endif  // SG_API_STRUCT_H
