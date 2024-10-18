#pragma once

#include <cstdint>
#include <cstddef>
#include<vector>

extern "C"
{

enum tpudnnDataType_t
{
    TPUDNN_DTYPE_FP32 = 0,
    TPUDNN_DTYPE_FP16 = 1,
    TPUDNN_DTYPE_INT8 = 2,
    TPUDNN_DTYPE_UINT8 = 3,
    TPUDNN_DTYPE_INT16 = 4,
    TPUDNN_DTYPE_UINT16 = 5,
    TPUDNN_DTYPE_INT32 = 6,
    TPUDNN_DTYPE_UINT32 = 7,
    TPUDNN_DTYPE_BF16 = 8,
    TPUDNN_DTYPE_INT4 = 9,
    TPUDNN_DTYPE_UINT4 = 10,
    TPUDNN_DTYPE_FP20 = 11,
    TPUDNN_DTYPE_FP8E5M2 = 12,
    TPUDNN_DTYPE_FP8E4M3 = 13,
    TPUDNN_DTYPE_INT64 = 14,
    TPUDNN_DTYPE_TF32 = 15,
    TPUDNN_DTYPE_BOOL = 16,

    TPUDNN_DTYPE_UNKNOWN = -1,
};

enum tpudnnReduceType_t {
    TPUDNN_REDUCE_MEAN = 0,
    TPUDNN_REDUCE_SUM  = 1,
    TPUDNN_REDUCE_MAX  = 2,
    TPUDNN_REDUCE_MIN  = 3,
    TPUDNN_REDUCE_PROD = 4,
    TPUDNN_REDUCE_L2   = 5,
    TPUDNN_REDUCE_L1   = 6,
};

typedef enum {
  TPUDNN_ACTIVE_TANH = 0,
  TPUDNN_ACTIVE_SIGMOID = 1,
  TPUDNN_ACTIVE_RELU = 2,
  TPUDNN_ACTIVE_EXP = 3,
  TPUDNN_ACTIVE_ELU = 4,
  TPUDNN_ACTIVE_SQRT = 5,
  TPUDNN_ACTIVE_SQUARE = 6,
  TPUDNN_ACTIVE_RSQRT = 7,
  TPUDNN_ACTIVE_ABSVAL = 8,
  TPUDNN_ACTIVE_LN = 9,
  TPUDNN_ACTIVE_ROUND = 10,
  TPUDNN_ACTIVE_CEIL = 11,
  TPUDNN_ACTIVE_FLOOR = 12,
  TPUDNN_ACTIVE_SIN = 13,
  TPUDNN_ACTIVE_COS = 14,
  TPUDNN_ACTIVE_IS_FINITE = 15,
  TPUDNN_ACTIVE_MISH = 16,
  TPUDNN_ACTIVE_SWISH = 17,
  TPUDNN_ACTIVE_HSWISH = 18,
  TPUDNN_ACTIVE_SILU = 19,
  TPUDNN_ACTIVE_ARCSIN = 20,
  TPUDNN_ACTIVE_ARCCOS = 21,
  TPUDNN_ACTIVE_ARCSINH = 22,
  TPUDNN_ACTIVE_ARCCOSH = 23,
  TPUDNN_ACTIVE_ARCTANH = 24,
  TPUDNN_ACTIVE_SINH = 25,
  TPUDNN_ACTIVE_COSH = 26,
  TPUDNN_ACTIVE_TAN = 27,
  TPUDNN_ACTIVE_SIGN = 28,
  TPUDNN_ACTIVE_GELU = 29,
  TPUDNN_ACTIVE_ERF = 30,
  TPUDNN_ACTIVE_HSIGMOID = 31,
  TPUDNN_ACTIVE_LOG_SIGMOID = 32,
  TPUDNN_ACTIVE_SOFT_PLUS = 33,
  TPUDNN_ACTIVE_SOFT_SIGN = 34,
  // only implemented in tpu-train
  TPUDNN_ACTIVE_ERFC = 35,
  TPUDNN_ACTIVE_ISINF = 36,
  TPUDNN_ACTIVE_ISNAN = 37,
  TPUDNN_ACTIVE_EXPM1 = 38,
  TPUDNN_ACTIVE_RECIPROCAL = 39,
  TPUDNN_ACTIVE_EXP2 = 40,
  TPUDNN_ACTIVE_TRUNC = 41,
} tensor_active_type_t;

typedef enum {
  TPUDNN_LOG_E = 0,
  TPUDNN_LOG_1P = 1,
  TPUDNN_LOG_2 = 2,
  TPUDNN_LOG_10 = 10,
} tensor_log_type_t;

typedef enum {
  TPUDNN_POOLING_MAX = 0,
  TPUDNN_POOLING_MIN = 1,
  TPUDNN_POOLING_AVG = 2,
} tensor_pooling_mode_t;

typedef enum {
  TPUDNN_UPSAMPLING_NEAREST = 0,
  TPUDNN_UPSAMPLING_BILINEAR = 1,
} tensor_resize_mode_t;
typedef struct
{
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;
}tpudnnConv2dParam_t;

typedef enum
{
  TPUDNN_NO_FORMATED = 0,
  TPUDNN_CONV_W_INFER_FORMAT  = 1,
  TPUDNN_CONV_W_TRAIN_FORMAT  = 2,
  TPUDNN_CONV_DW_TRAIN_FORMAT = 3,
}
TpudnnFormatedType_t;

typedef struct
{
    void *addr;
    int dim;
    int shape[8];
    int stride[8];
    tpudnnDataType_t dtype;
    TpudnnFormatedType_t format_casted;
} tpudnnTensor_t;
typedef struct {
  int kh;
  int kw;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int output_h;
  int output_w;
  tensor_pooling_mode_t mode;
} TPUDNN_PoolingDescriptor_t;

static inline size_t tpudnnTensorDataSize(tpudnnDataType_t dtype)
{
    if (dtype == TPUDNN_DTYPE_INT8 ||
        dtype == TPUDNN_DTYPE_UINT8)
    {
      return 1;
    }
    else if (dtype == TPUDNN_DTYPE_INT16 ||
             dtype == TPUDNN_DTYPE_UINT16 ||
             dtype == TPUDNN_DTYPE_FP16 ||
             dtype == TPUDNN_DTYPE_BF16)
    {
        return 2;
    }
    else if (dtype == TPUDNN_DTYPE_FP32 ||
             dtype == TPUDNN_DTYPE_INT32 ||
             dtype == TPUDNN_DTYPE_UINT32)
    {
        return 4;
    }
    else if ( dtype == TPUDNN_DTYPE_INT64 )
    {
        return 8;
    }
    return -1;
}

static inline size_t tpudnnTensorBytes(const tpudnnTensor_t *tensor)
{
    size_t bytes = tpudnnTensorDataSize(tensor->dtype);
    for ( int i = 0; i < tensor->dim; ++i)
    {
        bytes *= tensor->shape[i];
    }
    return bytes;
}

static inline bool tpudnnIsTensorContiguous(const tpudnnTensor_t *tensor)
{
    int stride = 1;
    for (int i = tensor->dim - 1; i >= 0; --i)
    {
        if (tensor->shape[i] > 1 && tensor->stride[i] != stride)
        {
            return false;
        }
        else
        {
            stride *= tensor->shape[i];
        }
    }
    return true;
}

static inline bool tpudnnIsTensorTransposed ( const tpudnnTensor_t * tensor )
{
    if ( tensor->dim < 2 || tpudnnIsTensorContiguous ( tensor ) )
    {
        return false;
    }
    else
    {
        int stride = 1;
        for ( int i = tensor->dim - 1; i >= 0; --i )
        {
            if ( ( i == tensor->dim - 1 && tensor->stride[i] != tensor->shape[tensor->dim - 2] ) ||
                 ( i == tensor->dim - 2 && tensor->stride[i] != 1 ) ||
                 ( i < tensor->dim - 2 && tensor->stride[i] != stride ) )
            {
                return false;
            }
            else
            {
                stride *= tensor->shape[i];
            }
        }
    }
    return true;
}

static inline bool tpudnnIsSameShape ( const tpudnnTensor_t * tensor1, const tpudnnTensor_t * tensor2 )
{
    if ( tensor1->dim == tensor2->dim )
    {
        for ( int i = 0; i < tensor1->dim; ++i )
        {
            if ( tensor1->shape[i] != tensor2->shape[i] )
            {
                return false;
            }
        }
    }
    else
    {
        return false;
    }
    return true;
}

static inline tpudnnTensor_t tpudnnUndefinedTensor()
{
    tpudnnTensor_t tensor = {.addr = 0};

    return tensor;
}

tpudnnStatus_t tpudnnBinaryAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t other,
    float scalar,
    tpudnnTensor_t output,
    int binary_type);

tpudnnStatus_t tpudnnMatmulAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t left,
    tpudnnTensor_t right,
    tpudnnTensor_t bias,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnSliceScatterAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t src,
    tpudnnTensor_t indices,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnLogSoftmaxAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnSoftmaxAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnSoftmaxBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t output,
    int dim,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnLogAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    tensor_log_type_t log_type);

tpudnnStatus_t tpudnnSqueezeAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnWhereAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t cond,
    tpudnnTensor_t self,
    tpudnnTensor_t other,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnNorm2Async(
    tpudnnHandle_t handle,
    const tpudnnTensor_t input,
    int keepdim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnNegAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnArangeAsync (
    tpudnnHandle_t handle,
    int start,
    int end,
    int step,
    tpudnnTensor_t out);

tpudnnStatus_t tpudnnRepeatAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int* repeat_times,
    int repeat_dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnStridedCopyAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnConvertAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnNonzeroAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t self,
    tpudnnTensor_t out,
    tpudnnTensor_t num);

tpudnnStatus_t tpudnnClampAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    float min,
    float max,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnNativeGroupNormAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t gamma,
    tpudnnTensor_t beta,
    int group,
    int affine,
    float eps,
    tpudnnTensor_t output,
    tpudnnTensor_t mean,
    tpudnnTensor_t rstd);

tpudnnStatus_t tpudnnLogicalAndAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t other,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnLogicalNotAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnLogicalOrAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t other,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnBitwiseNotAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnCbrtAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnAddCMulAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t tensor1,
    tpudnnTensor_t tensor2,
    float scalar,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnAddCMulBcastAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t tensor1,
    tpudnnTensor_t tensor2,
    float scalar,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnAddCDivAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t tensor1,
    tpudnnTensor_t tensor2,
    float scalar,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnCrossEntropyLossAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t target,
    int reduction,
    int ignore_index,
    float label_smoothing,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnCrossEntropyLossBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t target,
    tpudnnTensor_t grad_output,
    int ignore_index,
    int batch,
    int reduction,
    float label_smoothing,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnMselossAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t self,
    tpudnnTensor_t target,
    tpudnnTensor_t out,
    int reduction);

tpudnnStatus_t tpudnnPoolingForwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    TPUDNN_PoolingDescriptor_t pooling_desc);

tpudnnStatus_t tpudnnReduceMaxOrMinAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int* reduction_dim,
    int reduction_dim_length,
    int keepdim,
    int mode,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReduceVarAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int* reduce_list,
    int reduce_dim,
    int correction,
    int keepdim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReduceVarAllAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int correction,
    bool keepdim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReduceAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int start_dim,
    int end_dim,
    int keepdim,
    int mode,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReduceProdAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int axis,
    int keepdim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnIndexSelectAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t indices,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnEmbeddingBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t indices,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnConcatAsync (
    tpudnnHandle_t handle ,
    const tpudnnTensor_t * inputs,
    int input_num,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnGatherAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t index,
    tpudnnTensor_t output,
    int axis);

tpudnnStatus_t tpudnnArgAsync(
    tpudnnHandle_t resource,
    tpudnnTensor_t input,
    int axis,
    int mode,
    tpudnnTensor_t values,
    tpudnnTensor_t indices,
    tpudnnTensor_t buffer);

tpudnnStatus_t tpudnnTopkAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int k,
    int axis,
    bool largest,
    bool sorted,
    tpudnnTensor_t value,
    tpudnnTensor_t index);

tpudnnStatus_t tpudnnConjAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnRealAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnBatchnorm2dAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t bias,
    float eps,
    tpudnnTensor_t running_mean,
    tpudnnTensor_t running_var,
    float momentum,
    tpudnnTensor_t output,
    tpudnnTensor_t saved_mean,
    tpudnnTensor_t saved_invstd);

tpudnnStatus_t tpudnnBatchnorm2dBackwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t saved_mean,
    tpudnnTensor_t saved_invstd,
    tpudnnTensor_t grad_input,
    tpudnnTensor_t grad_weight,
    tpudnnTensor_t grad_bias);

tpudnnStatus_t tpudnnLayernormAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t bias,
    int start_dim,
    float eps,
    tpudnnTensor_t output,
    tpudnnTensor_t mean,
    tpudnnTensor_t rstd);

tpudnnStatus_t tpudnnLayernormBackwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t mean,
    tpudnnTensor_t rstd,
    int start_dim,
    tpudnnTensor_t grad_input,
    tpudnnTensor_t grad_weight,
    tpudnnTensor_t grad_bias,
    int requires_grad_input);

tpudnnStatus_t tpudnnSignbitAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReLUBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnGELUAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnGELUBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnLeakyReLUAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    float negative_slope);

tpudnnStatus_t tpudnnHardtanhAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    float min_value,
    float max_value,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnActiveAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    tensor_active_type_t active_type);

tpudnnStatus_t tpudnnReorderConv2dWeightAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int mode,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnConv2dAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t bias,
    tpudnnConv2dParam_t param,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnConv2dBackwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnConv2dParam_t param,
    tpudnnTensor_t grad_input,
    tpudnnTensor_t grad_weight,
    tpudnnTensor_t grad_bias);

tpudnnStatus_t tpudnnUpsamplingAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    bool align_corners,
    tensor_resize_mode_t upsampling_type);

tpudnnStatus_t tpudnnUpsampleNearest2dBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t grad_input,
    int scale,
    TPUDNN_PoolingDescriptor_t pooling_desc);

tpudnnStatus_t tpudnnFlipAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int axis,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnTriangularizeAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t self,
    int is_upper,
    int diagonal,
    tpudnnTensor_t out);

tpudnnStatus_t tpudnnInfCheckAndUnscaleAsync(
    tpudnnHandle_t handle,
    std::vector<tpudnnTensor_t>& inputs,
    tpudnnTensor_t found_inf,
    float inv_scale);

tpudnnStatus_t tpudnnRmsNormForwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t scale,
    tpudnnTensor_t bias,
    tpudnnTensor_t output,
    int axis,
    float eps
);

tpudnnStatus_t tpudnnRmsNormBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t scale,
    tpudnnTensor_t rms,
    tpudnnTensor_t grad_input,
    tpudnnTensor_t grad_scale,
    tpudnnTensor_t grad_bias,
    int axis,
    double eps);

tpudnnStatus_t tpudnnC2CSend(
    tpudnnHandle_t handle,
    void *buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    int dst_rank,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CRecv(
    tpudnnHandle_t handle,
    void *buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    int src_rank,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CAllReduce(
    tpudnnHandle_t handle,
    void *send_buff,
    void *recv_buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    tpudnnReduceType_t reduce_method,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map,
    int loop);

tpudnnStatus_t tpudnnC2CReduce(
    tpudnnHandle_t handle,
    void *send_buff,
    void *recv_buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    tpudnnReduceType_t reduce_method,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CGather(
    tpudnnHandle_t handle,
    void *send_buff,
    uint64_t send_count,
    void *recv_buff,
    uint64_t recv_count,
    tpudnnDataType_t dtype,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CAllGather(
    tpudnnHandle_t handle,
    void *send_buff,
    uint64_t send_count,
    void *recv_buff,
    uint64_t recv_count,
    const char* uuid,
    tpudnnDataType_t dtype,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CBroadcast(
    tpudnnHandle_t handle,
    void *buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CScatter(
    tpudnnHandle_t handle,
    void *send_mem,
    tpudnnDataType_t send_type,
    void *recv_mem,
    uint64_t recv_count,
    tpudnnDataType_t recv_type,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CAllToAll(
    tpudnnHandle_t handle,
    void *send_mem,
    tpudnnDataType_t send_type,
    void *recv_mem,
    uint64_t recv_count,
    tpudnnDataType_t recv_type,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnMaskedFillAsync(
    tpudnnHandle_t handle,
    const tpudnnTensor_t input,
    const tpudnnTensor_t mask,
    float value,
    tpudnnTensor_t out);

tpudnnStatus_t tpudnnFillAsync(
    tpudnnHandle_t handle,
    const void * scalar_ptr,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnAdamBackwardMultiCoreAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t weight_out,
    tpudnnTensor_t m_out,
    tpudnnTensor_t v_out,
    tpudnnTensor_t vmax_out,
    tpudnnTensor_t grad_weight,
    tpudnnTensor_t weight_in,
    tpudnnTensor_t m_in,
    tpudnnTensor_t v_in,
    tpudnnTensor_t vmax_in,
    tpudnnTensor_t t,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    bool amsgrad,
    bool maximize);

tpudnnStatus_t tpudnnLoraMatmulForwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t loraA,
    tpudnnTensor_t loraB,
    tpudnnTensor_t weight,
    tpudnnTensor_t output,
    float scale);

tpudnnStatus_t tpudnnLlamaAttentionAsync ( 
    tpudnnHandle_t handle,
    tpudnnTensor_t OUT,
    tpudnnTensor_t Q,
    tpudnnTensor_t K,
    tpudnnTensor_t V,
    tpudnnTensor_t Kcache,
    tpudnnTensor_t Vcache,
    tpudnnTensor_t cos,
    tpudnnTensor_t sin,
    tpudnnTensor_t save_slots,
    tpudnnTensor_t fetch_slots,
    tpudnnTensor_t mask,
    tpudnnTensor_t Qbuffer,
    tpudnnTensor_t Kbuffer,
    tpudnnTensor_t Vbuffer,
    tpudnnTensor_t input_lengths_tensor,
    int*           input_lengths,
    int            num_input_lengths,
    int slots_size,
    int mask_size,
    int block_size,
    float C,
    int attention_mode // 2: prefile, 3: decode
    ); 

tpudnnStatus_t tpudnnLLamaMlpAsync ( 
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight0,
    tpudnnTensor_t weight1,
    tpudnnTensor_t weight2,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnLLamaA16MlpAsync ( 
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight0,
    tpudnnTensor_t zp0,
    tpudnnTensor_t scale0,
    tpudnnTensor_t weight1,
    tpudnnTensor_t zp1,
    tpudnnTensor_t scale1,
    tpudnnTensor_t weight2,
    tpudnnTensor_t zp2,
    tpudnnTensor_t scale2,
    int group_size,
    int weight_bits,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnGDMAD2DAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t src,
    tpudnnTensor_t dst,
    size_t size);
} // extern "C"
