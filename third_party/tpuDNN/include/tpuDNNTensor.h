#pragma once

#include <cstdint>
#include <cstddef>

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
  TPUDNN_LOG_E = 0,
  TPUDNN_LOG_1P = 1,
  TPUDNN_LOG_2 = 2,
  TPUDNN_LOG_10 = 10,
} tensor_log_type_t;

typedef struct
{
    void *addr;
    int dim;
    int shape[8];
    int stride[8];
    tpudnnDataType_t dtype;
} tpudnnTensor_t;

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
    tpudnnTensor_t tensor = { .addr = 0 };
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

tpudnnStatus_t tpudnnC2CAllReduce(
    tpudnnHandle_t handle,
    void *send_buff,
    void *recv_buff,
    int count,
    tpudnnDataType_t dtype,
    tpudnnReduceType_t reduce_method,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CReduce(
    tpudnnHandle_t handle,
    void *send_buff,
    void *recv_buff,
    int count,
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
    int send_count,
    void *recv_buff,
    int recv_count,
    tpudnnDataType_t dtype,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CAllGather(
    tpudnnHandle_t handle,
    void *send_buff,
    int send_count,
    void *recv_buff,
    int recv_count,
    const char* uuid,
    tpudnnDataType_t dtype,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CBroadcast(
    tpudnnHandle_t handle,
    void *buff,
    int count,
    tpudnnDataType_t dtype,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CScatter(
    tpudnnHandle_t handle,
    void *send_mem,
    int send_count,
    tpudnnDataType_t send_type,
    void *recv_mem,
    int recv_count,
    tpudnnDataType_t recv_type,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CAllToAll(
    tpudnnHandle_t handle,
    void *send_mem,
    int send_count,
    tpudnnDataType_t send_type,
    void *recv_mem,
    int recv_count,
    tpudnnDataType_t recv_type,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

} // extern "C"
