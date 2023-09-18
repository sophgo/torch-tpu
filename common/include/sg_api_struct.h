#include <stdbool.h>
#

#ifndef SG_API_STRUCT_H
#define SG_API_STRUCT_H

#pragma pack(push, 1)

#ifndef FW_MAX_SHAPE_DIMS
#define FW_MAX_SHAPE_DIMS      8
#endif
#ifndef FW_MAX_CONCAT_NUM
#define FW_MAX_CONCAT_NUM     10
#endif


#ifndef WIN32
#define WITH_PLATFORM(x) __attribute__ ((packed)) x
#else
#define WITH_PLATFORM(x) x
#endif


typedef enum {
  ACTIVE_TANH = 0,
  ACTIVE_SIGMOID = 1,
  ACTIVE_RELU = 2,
  ACTIVE_EXP = 3,
  ACTIVE_ELU = 4,
  ACTIVE_SQRT = 5,
  ACTIVE_SQUARE = 6,
  ACTIVE_RSQRT = 7,
  ACTIVE_ABSVAL = 8,
  ACTIVE_LN = 9,
  ACTIVE_ROUND = 10,
  ACTIVE_CEIL = 11,
  ACTIVE_FLOOR = 12,
  ACTIVE_SIN = 13,
  ACTIVE_COS = 14,
  ACTIVE_IS_FINITE = 15,
  ACTIVE_MISH = 16,
  ACTIVE_SWISH = 17,
  ACTIVE_HSWISH = 18,
  ACTIVE_SILU = 19,
  ACTIVE_ARCSIN = 20,
  ACTIVE_ARCCOS = 21,
  ACTIVE_ARCSINH = 22,
  ACTIVE_ARCCOSH = 23,
  ACTIVE_ARCTANH = 24,
  ACTIVE_SINH = 25,
  ACTIVE_COSH = 26,
  ACTIVE_TAN = 27,
  ACTIVE_SIGN = 28,
  ACTIVE_GELU = 29,
  ACTIVE_ERF = 30,
  ACTIVE_HSIGMOID = 31,
  ACTIVE_LOG_SIGMOID = 32,
  ACTIVE_SOFT_PLUS = 33,
  ACTIVE_SOFT_SIGN = 34,
  // only implemented in tpu-train
  ACTIVE_ERFC = 35,
  ACTIVE_ISINF = 36,
  ACTIVE_ISNAN = 37,
  ACTIVE_EXPM1 = 38,
  ACTIVE_RECIPROCAL = 39,
  ACTIVE_EXP2 = 40,
  ACTIVE_TRUNC = 41,
} sg_active_type_t;

typedef enum {
  LOG_E = 0,
  LOG_1P = 1,
  LOG_2 = 2,
  LOG_10 = 10,
} sg_log_type_t;

/**
 * TPU1686/common/include/common_def.h
 * used by upsampling
 */
typedef enum {
  CAFFE_SUPPORT = 0,
  TENSORFLOW_SUPPORT = 1,
  CAFFE_NEAREST = 2,
  TENSORFLOW_NEAREST = 3,
  PYTORCH_SUPPORT = 4,
  PYTORCH_NEAREST = 5,
  OPENCV_BILINEAR = 6,
  ONNX_NEAREST = 7,
} PLATFORM_SUPPORT;

typedef enum {
  POOLING_MAX = 0,
  POOLING_MIN = 1,
  POOLING_AVG = 2,
} sg_pooling_mode_t;

typedef struct {
  int kh;
  int kw;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int output_h;
  int output_w;
  sg_pooling_mode_t mode;
} PoolingDescriptor_t;

typedef enum {
  UPSAMPLING_NEAREST = 0,
  UPSAMPLING_BILINEAR = 1,
} sg_resize_mode_t;

typedef enum {
  BITWISE_XOR,
  BITWISE_AND,
  BITWISE_OR,
} sg_element_bitwise_mode_t;

typedef enum {
  EQUAL,
  NOT_EQUAL,
  GREATER,
  GREATER_OR_EQUAL,
  LESS_THAN,
  LESS_THAN_OR_EQUAL,
} sg_comparision_mode_t;

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long buffer_addr;
  int dim;
  int shape[FW_MAX_SHAPE_DIMS];
  int out_shape[FW_MAX_SHAPE_DIMS];
  int pad_bag;
  int pad_end;
  bool half_pixel_centers;
  bool align_corners;
  bool if_getting_buffer_size;
  int dtype;
  unsigned long long* buffer_size_ptr;
  // PYTORCH_SUPPORT for **bilinear**; PYTORCH_NEAREST for nearest
  PLATFORM_SUPPORT platform_sp;
} WITH_PLATFORM(sg_api_upsampling2d_t);

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  unsigned long long grad_weight_global_addr;
  unsigned long long grad_bias_global_addr;
  unsigned long long buffer_global_addr;
  int input_shape[4];
  int output_shape[4];
  int groups;
  int kernel[2];
  int stride[2];
  int dilation[2];
  int pad[4];
  int dtype;
} WITH_PLATFORM(sg_api_conv2d_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_exp_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_expm1_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int axis;
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_flip_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_logical_and_t);

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
} WITH_PLATFORM(sg_api_batchnorm2d_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_logical_not_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_relu_backward_t);

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
} WITH_PLATFORM(sg_api_conv2d_t);

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
} WITH_PLATFORM(sg_api_batchnorm2d_t);

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
} WITH_PLATFORM(sg_api_layernorm_t);

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
} WITH_PLATFORM(sg_api_layernorm_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int dtype;
  int do_relu;
  PoolingDescriptor_t pooling_desc;
} WITH_PLATFORM(sg_api_pooling_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_relu_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_dtype;
  int output_dtype;
} WITH_PLATFORM(sg_api_dtype_convert_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[4];
  int mode;
} WITH_PLATFORM(sg_api_conv_weight_reorder_t);

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
} WITH_PLATFORM(sg_api_batch_matmul_t);

typedef struct
{
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
} WITH_PLATFORM(sg_api_matmul_multi_core_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
} WITH_PLATFORM(sg_api_softmax_t);

typedef struct
{
  unsigned long long output_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
} WITH_PLATFORM(sg_api_softmax_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_gelu_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_gelu_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float negative_slope;
  int dtype;
} WITH_PLATFORM(sg_api_leakyrelu_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long grad_output_global_addr;
  unsigned long long grad_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float negative_slope;
  int dtype;
} WITH_PLATFORM(sg_api_leakyrelu_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_stride[FW_MAX_SHAPE_DIMS];
  int output_stride[FW_MAX_SHAPE_DIMS];
  int dtype;
} WITH_PLATFORM(sg_api_strided_copy_t);

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
} WITH_PLATFORM(sg_api_where_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_input_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_sigmoid_backward_t);

typedef struct
{
  unsigned long long input_global_addrs[FW_MAX_CONCAT_NUM];
  unsigned long long output_global_addr;
  int input_shapes[FW_MAX_CONCAT_NUM][FW_MAX_SHAPE_DIMS];
  int dim;
  int input_num;
  int axis;
  int dtype;
} WITH_PLATFORM(sg_api_concat_t);

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
} WITH_PLATFORM(sg_api_reduce_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long buffer_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int dtype;
} WITH_PLATFORM(sg_api_reduce_prod_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int src_dtype;
  int dst_dtype;
  int other_dtype;
} WITH_PLATFORM(sg_api_shift_left_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int src_dtype;
  int dst_dtype;
  int other_dtype;
}WITH_PLATFORM(sg_api_shift_left_bcast_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  float min;
  float max;
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_clamp_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  int src_dtype;
  int dst_dtype;
  int other_dtype;
  char const_value;
}WITH_PLATFORM(sg_api_shift_left_c_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int src_dtype;
  int dst_dtype;
  int other_dtype;
} WITH_PLATFORM(sg_api_shift_right_arithmetic_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int src_dtype;
  int dst_dtype;
  int other_dtype;
}WITH_PLATFORM(sg_api_shift_right_arithmetic_bcast_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  int src_dtype;
  int dst_dtype;
  int other_dtype;
  char const_value;
}WITH_PLATFORM(sg_api_shift_right_arithmetic_c_t);

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
} WITH_PLATFORM(sg_api_index_select_t);

typedef struct
{
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  unsigned int value;
  int dtype;
} WITH_PLATFORM(sg_api_constant_fill_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_sqrt_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_rsqrt_t);
typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_sign_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_neg_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  sg_active_type_t active_type;
} WITH_PLATFORM(sg_api_active_t);

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
} WITH_PLATFORM(sg_api_addcdiv_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long tensor1_global_addr;
  unsigned long long tensor2_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int tensor1_shape[FW_MAX_SHAPE_DIMS];
  int tensor2_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int tensor1_dim;
  int tensor2_dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_bcast_addcmul_t);

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
} WITH_PLATFORM(sg_api_addcmul_t);

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
} WITH_PLATFORM(sg_api_cross_entropy_loss_t);

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
} WITH_PLATFORM(sg_api_cross_entropy_loss_backward_t);

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
} WITH_PLATFORM(sg_api_embedding_backward_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_abs_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_trifunc_t); // tan/cos/sin

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  sg_log_type_t log_type; // 0 for log, 1 for log1p, 2 for log2, 10 for log10
} WITH_PLATFORM(sg_api_log_t);

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int tile_axis;
  int tile_num;
  int dtype;
} WITH_PLATFORM(sg_api_squeeze_t);

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int group_num;
  float eps;
  int affine;
  int dtype;
} WITH_PLATFORM(sg_api_native_group_norm_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_add_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_sub_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_div_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_pow_t);

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
} WITH_PLATFORM(sg_api_bcast_add_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int other_dim;
  float value;
  int dtype;
  int binary_type;
} WITH_PLATFORM(sg_api_arithmetic_eltwise_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_mulc_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_addc_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_cdiv_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_csub_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_logical_or_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_norm2_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long mean_global_addr;
  unsigned long long rstd_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int axis;
  float eps;
  int affine;
  int dtype;
} WITH_PLATFORM(sg_api_layernorm_forward_multi_core_t);

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
  unsigned long long grad_weight_reduce_buffer;
  unsigned long long grad_bias_reduce_buffer;
  int shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int axis;
  int affine;
  int dtype;
} WITH_PLATFORM(sg_api_layernorm_backward_multi_core_t);

typedef struct
{
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
} WITH_PLATFORM(sg_api_where_multi_core_t);

typedef struct
{
  unsigned long long input_addr;
  unsigned long long output_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dims;
  int binary_type;
  int dtype;
  float const_value;
  int is_inversed;
} WITH_PLATFORM(sg_api_const_binary_float_t);

typedef struct
{
  unsigned long long input0_addr;
  unsigned long long input1_addr;
  unsigned long long output_addr;
  int in0_shape[FW_MAX_SHAPE_DIMS];
  int in1_shape[FW_MAX_SHAPE_DIMS];
  int in0_dims;
  int in1_dims;
  float in0_scale;
  float in1_scale;
  int binary_type;
  int dtype;
} WITH_PLATFORM(sg_api_binary_multi_core_t);

typedef struct
{
  int core_idx;
  int core_num;
  int core_msg_id;
  int name_len;
  int api_id;
  int api_size;
  unsigned char api_data[0];
} WITH_PLATFORM(sg_api_core_info_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long mask_global_addr;
  unsigned long long out_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int mask_shape[FW_MAX_SHAPE_DIMS];
  int input_dims;
  int mask_dims;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_masked_fill_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long mask_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  unsigned long long seed;
  float threshold;
  int dtype;
} WITH_PLATFORM(sg_api_dropout_t);

typedef struct {
  unsigned long long input_addr;
  unsigned long long weight0_addr;
  unsigned long long weight1_addr;
  unsigned long long bias0_addr;
  unsigned long long bias1_addr;
  unsigned long long output_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int w0_shape[FW_MAX_SHAPE_DIMS];
  int w1_shape[FW_MAX_SHAPE_DIMS];
  int in_dims;
  int w0_dims;
  int w1_dims;
  int in_dtype;
  int out_dtype;
  int has_bias;
  int use_fast;
#ifndef WIN32
} __attribute__((packed)) sg_api_mlp_multi_core_t;
#else
} sg_api_mlp_multi_core_t;
#endif

typedef struct
{
  int start;
  int end;
  int step;
  unsigned long long output_global_addr;
  int dtype;
  int dim;
  int shape[FW_MAX_SHAPE_DIMS];

} WITH_PLATFORM(sg_api_arange_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
  int mode;
} WITH_PLATFORM(sg_api_element_bitwise_t);

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
  int mode;
} WITH_PLATFORM(sg_api_element_bitwise_bcast_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int value;
  int dtype;
  int mode;
} WITH_PLATFORM(sg_api_element_bitwise_c_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  int mode;
} WITH_PLATFORM(sg_api_comparision_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  int mode;
} WITH_PLATFORM(sg_api_comparision_bcast_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float const_value;
  int mode;
} WITH_PLATFORM(sg_api_comparision_c_t);


typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_bitwise_not_t);


typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float scalar;
} WITH_PLATFORM(sg_api_minimumc_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_minimum_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int other_dim;
  int output_dim;
  int dtype;
} WITH_PLATFORM(sg_api_minimum_bcast_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float scalar;
} WITH_PLATFORM(sg_api_maximumc_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_maximum_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int other_dim;
  int output_dim;
  int dtype;
} WITH_PLATFORM(sg_api_maximum_bcast_t);

typedef struct
{
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float scalar;
} WITH_PLATFORM(sg_api_atan2c_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float scalar;
} WITH_PLATFORM(sg_api_atan2_c_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_atan2_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int other_dim;
  int output_dim;
  int dtype;
} WITH_PLATFORM(sg_api_atan2_bcast_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long buffer_global_addr;
  unsigned long long batch1_global_addr;
  unsigned long long batch2_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int batch1_shape[FW_MAX_SHAPE_DIMS];
  int batch2_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int batch1_dim;
  int batch2_dim;
  int output_dim;
  int dtype;
  float alpha;
  float beta;
} WITH_PLATFORM(sg_api_baddbmm_t);
typedef struct
{
  unsigned long long input1_global_addr;
  unsigned long long input2_global_addr;
  unsigned long long output_global_addr;
  int reduction;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_mse_loss_t);

typedef struct{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_silu_t);

typedef struct {
  unsigned long long input_addr;
  unsigned long long gamma_addr;
  unsigned long long beta_addr;
  unsigned long long weight_addr;
  unsigned long long bias_addr;
  unsigned long long mean_addr;
  unsigned long long rstd_addr;
  unsigned long long output_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int w_shape[FW_MAX_SHAPE_DIMS];
  int in_dims;
  int w_dims;
  int in_dtype;
  float eps;
  int has_bias;
#ifndef WIN32
} __attribute__((packed)) sg_api_ln_mm_multi_core_t;
#else
} sg_api_ln_mm_multi_core_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float scalar;
} WITH_PLATFORM(sg_api_fmaxc_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_fmax_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int other_dim;
  int output_dim;
  int dtype;
} WITH_PLATFORM(sg_api_fmax_bcast_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float scalar;
} WITH_PLATFORM(sg_api_fminc_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_fmin_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int output_shape[FW_MAX_SHAPE_DIMS];
  int input_dim;
  int other_dim;
  int output_dim;
  int dtype;
} WITH_PLATFORM(sg_api_fmin_bcast_t);
typedef struct {
  unsigned long long input0_addr;
  unsigned long long input1_addr;
  unsigned long long gamma_addr;
  unsigned long long beta_addr;
  unsigned long long weight_addr;
  unsigned long long bias_addr;
  unsigned long long out_add_addr;
  unsigned long long mean_addr;
  unsigned long long rstd_addr;
  unsigned long long output_addr;
  int in_shape[FW_MAX_SHAPE_DIMS];
  int w_shape[FW_MAX_SHAPE_DIMS];
  int in_dims;
  int w_dims;
  int in_dtype;
  float eps;
  int has_bias;
  int use_fast;
#ifndef WIN32
} __attribute__((packed)) sg_api_add_ln_mm_multi_core_t;
#else
} sg_api_add_ln_mm_multi_core_t;
#endif

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_signbit_t);

typedef struct
{
  unsigned long long self_global_addr;
  unsigned long long out_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float value;
  int dtype;
} WITH_PLATFORM(sg_api_pow_tensor_scalar_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int input_stride[FW_MAX_SHAPE_DIMS];
  int output_stride[FW_MAX_SHAPE_DIMS];
  int dtype;
} WITH_PLATFORM(sg_api_real_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long trans_buffer_global_addr;
  unsigned long long copy_buffer_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int stride[FW_MAX_SHAPE_DIMS];
  int dim_order[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_permute_t);

typedef struct 
{
  unsigned long long input_global_addr;
  unsigned long long value_global_addr;
  unsigned long long index_global_addr;
  unsigned long long index_buffer_global_addr;
  unsigned long long trans_buffer_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int k;
  int dim_order;
  bool largest;
  bool sorted;
  int dtype;
} WITH_PLATFORM(sg_api_topk_t);

typedef struct 
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long index_global_addr;
  unsigned long long num_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_nonzero_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long buffer_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int reduction_dim[FW_MAX_SHAPE_DIMS];
  int reduction_dim_length;
  int dim;
  int mode;
  int dtype;
} WITH_PLATFORM(sg_api_reduce_max_or_min_t);

typedef struct 
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int repeat_times[FW_MAX_SHAPE_DIMS];
  int dim;
  int repeat_dim;
  int dtype;
} WITH_PLATFORM(sg_api_repeat_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long values_global_addr;
  unsigned long long indices_global_addr;
  unsigned long long buffer_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int axis;
  int mode;
  int dtype;
} WITH_PLATFORM(sg_api_reduce_arg_t);

typedef struct 
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
  float min_value;
  float max_value;
} WITH_PLATFORM(sg_api_hardtanh_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_hypot_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long other_global_addr;
  unsigned long long output_global_addr;
  int input_shape[FW_MAX_SHAPE_DIMS];
  int other_shape[FW_MAX_SHAPE_DIMS];
  int dim;
  int dtype;
} WITH_PLATFORM(sg_api_hypot_bcast_t);

typedef struct
{
  unsigned long long input_global_addr;
  unsigned long long output_global_addr;
  int shape[FW_MAX_SHAPE_DIMS];
  int dim;
  float scalar;
  int dtype;
} WITH_PLATFORM(sg_api_hypot_c_t);

#pragma pack(pop)
#endif  // SG_API_STRUCT_H
