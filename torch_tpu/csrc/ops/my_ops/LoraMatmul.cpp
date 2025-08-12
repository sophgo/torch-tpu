#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"
namespace at{
    Tensor lora_matmul_forward(
        Tensor &input,
        Tensor &loraA,
        Tensor &loraB,
        Tensor &weight,
        Tensor &output,
        double_t scale)
    {
        TIMING_START;
        CHECK_TENSOR_IN_DEVICE(input);
        CHECK_TENSOR_IN_DEVICE(loraA);
        CHECK_TENSOR_IN_DEVICE(loraB);
        CHECK_TENSOR_IN_DEVICE(weight);
        CHECK_TENSOR_IN_DEVICE(output);
        
        auto stream = c10_tpu::getCurrentTPUStream();
        tpudnnStatus_t status = tpudnnLoraMatmulForwardAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, input),
            tpu::TPUGenerateTpudnnTensor(stream, loraA),
            tpu::TPUGenerateTpudnnTensor(stream, loraB),
            tpu::TPUGenerateTpudnnTensor(stream, weight),
            tpu::TPUGenerateTpudnnTensor(stream, output),
            scale
        );
        TORCH_CHECK(status ==  TPUDNN_STATUS_SUCCESS);
        TIMING_END;
        return output;
    }
}