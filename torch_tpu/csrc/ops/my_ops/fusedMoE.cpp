#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

namespace at
{
    Tensor fused_moe_grouped_topk(
        Tensor &topk_experts_res,
        Tensor &topk_weights_res_bf16,
        Tensor &left_bf16,
        Tensor &right_bf16,
        Tensor &topk_weights_res,
        Tensor &left,
        Tensor &right,
        Tensor &max,
        Tensor &matmul_res,
        Tensor &softmax_res)
    {
        CHECK_TENSOR_IN_DEVICE(topk_experts_res);
        CHECK_TENSOR_IN_DEVICE(topk_weights_res_bf16);
        CHECK_TENSOR_IN_DEVICE(left_bf16);
        CHECK_TENSOR_IN_DEVICE(right_bf16);
        CHECK_TENSOR_IN_DEVICE(topk_weights_res);
        CHECK_TENSOR_IN_DEVICE(left);
        CHECK_TENSOR_IN_DEVICE(right);
        CHECK_TENSOR_IN_DEVICE(max);
        CHECK_TENSOR_IN_DEVICE(matmul_res);
        CHECK_TENSOR_IN_DEVICE(softmax_res);

        TIMING_START;
        auto stream = c10_tpu::getCurrentTPUStream();
        tpudnnStatus_t status = tpudnnFusedMoEGroupedTopkMultiCoreAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, topk_experts_res),
            tpu::TPUGenerateTpudnnTensor(stream, topk_weights_res_bf16),
            tpu::TPUGenerateTpudnnTensor(stream, left_bf16),
            tpu::TPUGenerateTpudnnTensor(stream, right_bf16),
            tpu::TPUGenerateTpudnnTensor(stream, topk_weights_res),
            tpu::TPUGenerateTpudnnTensor(stream, left),
            tpu::TPUGenerateTpudnnTensor(stream, right),
            tpu::TPUGenerateTpudnnTensor(stream, max),
            tpu::TPUGenerateTpudnnTensor(stream, matmul_res),
            tpu::TPUGenerateTpudnnTensor(stream, softmax_res));

        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        TIMING_END(tpu::FUSED_MOE_GROUPED_TOPK);
        return topk_experts_res;
    }

    Tensor fused_moe_fused_experts(
        Tensor &output,
        Tensor &input,
        const c10::optional<Tensor> &output_sample,
        const c10::optional<Tensor> &input_sample,
        Tensor &gate_weights,
        Tensor &up_weights,
        Tensor &down_weights,
        Tensor &gate_scales,
        Tensor &up_scales,
        Tensor &down_scales,
        Tensor &select_experts,
        Tensor &routing_weights,
        const c10::optional<Tensor> &num_select_experts,
        const c10::optional<Tensor> &select_experts_middle,
        const c10::optional<Tensor> &routing_weights_middle,
        int64_t blocksize,
        int64_t num_experts,
        int64_t num_experts_per_tok,
        bool use_grouped_topk,
        int64_t num_expert_group,
        int64_t topk_group,
        const c10::optional<Tensor> &silu,
        const c10::optional<Tensor> &sigmoid,
        const c10::optional<Tensor> &m0,
        bool save_mid_res)
    {
        CHECK_TENSOR_IN_DEVICE(output);
        CHECK_TENSOR_IN_DEVICE(input);
        if (output_sample.has_value())
            CHECK_TENSOR_IN_DEVICE(output_sample.value());
        if (input_sample.has_value())
            CHECK_TENSOR_IN_DEVICE(input_sample.value());
        CHECK_TENSOR_IN_DEVICE(gate_weights);
        CHECK_TENSOR_IN_DEVICE(up_weights);
        CHECK_TENSOR_IN_DEVICE(down_weights);
        CHECK_TENSOR_IN_DEVICE(gate_scales);
        CHECK_TENSOR_IN_DEVICE(up_scales);
        CHECK_TENSOR_IN_DEVICE(down_scales);
        CHECK_TENSOR_IN_DEVICE(select_experts);
        TORCH_CHECK(select_experts.dtype() == torch::kInt32, "select_experts must be int32 dtype");
        CHECK_TENSOR_IN_DEVICE(routing_weights);
        if (num_select_experts.has_value()) {
            CHECK_TENSOR_IN_DEVICE(num_select_experts.value());
            TORCH_CHECK (num_select_experts.value().dtype() == torch::kInt64,
                        "num_select_experts must be int64 dtype");
        }
        if (select_experts_middle.has_value())
            CHECK_TENSOR_IN_DEVICE(select_experts_middle.value());
        if (routing_weights_middle.has_value())
            CHECK_TENSOR_IN_DEVICE(routing_weights_middle.value());
        if (silu.has_value())
            CHECK_TENSOR_IN_DEVICE(silu.value());
        if (sigmoid.has_value())
            CHECK_TENSOR_IN_DEVICE(sigmoid.value());
        if (m0.has_value())
            CHECK_TENSOR_IN_DEVICE(m0.value());

        TIMING_START;
        auto stream = c10_tpu::getCurrentTPUStream();
        tpudnnStatus_t status = tpudnnFusedMoEFusedExpertsMultiCoreAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, output),
            tpu::TPUGenerateTpudnnTensor(stream, input),
            output_sample.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, output_sample.value()) : tpudnnUndefinedTensor(),
            input_sample.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, input_sample.value()) : tpudnnUndefinedTensor(),
            tpu::TPUGenerateTpudnnTensor(stream, gate_weights),
            tpu::TPUGenerateTpudnnTensor(stream, up_weights),
            tpu::TPUGenerateTpudnnTensor(stream, down_weights),
            tpu::TPUGenerateTpudnnTensor(stream, gate_scales),
            tpu::TPUGenerateTpudnnTensor(stream, up_scales),
            tpu::TPUGenerateTpudnnTensor(stream, down_scales),
            tpu::TPUGenerateTpudnnTensor(stream, select_experts),
            tpu::TPUGenerateTpudnnTensor(stream, routing_weights),
            num_select_experts.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, num_select_experts.value()) : tpudnnUndefinedTensor(),
            select_experts_middle.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, select_experts_middle.value()) : tpudnnUndefinedTensor(),
            routing_weights_middle.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, routing_weights_middle.value()) : tpudnnUndefinedTensor(),
            blocksize,
            num_experts,
            num_experts_per_tok,
            use_grouped_topk,
            num_expert_group,
            topk_group,
            silu.has_value() ? tpu::TPUGenerateTpudnnTensor( stream, silu.value()) : tpudnnUndefinedTensor(),
            sigmoid.has_value() ? tpu::TPUGenerateTpudnnTensor( stream, sigmoid.value()) : tpudnnUndefinedTensor(),
            m0.has_value() ? tpu::TPUGenerateTpudnnTensor( stream, m0.value()) : tpudnnUndefinedTensor(),
            save_mid_res);

        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        TIMING_END(tpu::FUSED_MOE_FUSED_EXPERTS);
        return output;
    }
}
