#pragma once

#include <cstdint>
#include <cstddef>

extern "C"
{

enum tpudnnDataType_t
{
    TPUDNN_DTYPE_UNKNOWN = 0,
    TPUDNN_DTYPE_INT8,
    TPUDNN_DTYPE_UINT8,
    TPUDNN_DTYPE_INT16,
    TPUDNN_DTYPE_UINT16,
    TPUDNN_DTYPE_FP16,
    TPUDNN_DTYPE_BF16,
    TPUDNN_DTYPE_INT32,
    TPUDNN_DTYPE_UINT32,
    TPUDNN_DTYPE_FP32,
    TPUDNN_DTYPE_INT64
};

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

tpudnnStatus_t tpudnnBinaryAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t other,
    float scalar,
    tpudnnTensor_t output,
    int binary_type);

} // extern "C"
