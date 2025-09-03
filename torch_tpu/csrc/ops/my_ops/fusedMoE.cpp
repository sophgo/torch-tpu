#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

#ifdef USING_PPL
#include "fusedMoE.h"
static void fusedMoE_impl(uint64_t output_addr, uint64_t input_addr,
                          uint64_t gate_weight_addr, uint64_t up_weight_addr,
                          uint64_t down_weight_addr, uint64_t gate_scale_addr,
                          uint64_t up_scale_addr, uint64_t down_scale_addr,
                          uint64_t select_experts_addr,
                          uint64_t routing_weights_addr, uint64_t gather_index,
                          uint64_t scatter_index, uint64_t gather_buffer,
                          uint64_t scatter_buffer, int blocksize,
                          int num_experts, int num_experts_per_topk, int batch,
                          int input_w, int middle_w, at::ScalarType dtype,
                          at::ScalarType weight_dtype, int quantized) {
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
                    int tile_size, int index_batch) -> int {
    // if (src_type == at::kHalf) {
    //   if (dst_type == at::kFloat) {
    return moe_multi_core_bf16_fp8e4m3(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, gate_weight_addr, up_weight_addr,
        down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
        select_experts_addr, routing_weights_addr, gather_index, scatter_index,
        gather_buffer, scatter_buffer, blocksize, num_experts,
        num_experts_per_topk, batch, tile_size, index_batch, input_w, middle_w,
        quantized);
    //   }
    // }
    // return -1;
  };
  int32_t addrs[57];
  auto kernel_check_mem = [&](int tile_size, int index_batch) -> int {
    // if (src_type == at::kHalf) {
    //   if (dst_type == at::kFloat) {
    return moe_multi_core_bf16_fp8e4m3_check_mem(
        output_addr, input_addr, gate_weight_addr, up_weight_addr,
        down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
        select_experts_addr, routing_weights_addr, gather_index, scatter_index,
        gather_buffer, scatter_buffer, blocksize, num_experts,
        num_experts_per_topk, batch, tile_size, index_batch, input_w, middle_w,
        quantized, addrs);
    //   }
    // }
    // return -1;
  };

  auto stream = c10_tpu::getCurrentTPUStream();
  tpuKernelModule_t ppl_module = getPplModule();
  uint32_t tile_size = 256;
  uint32_t index_batch = 256;
  while (index_batch >= 1) {
    int ret = kernel_check_mem(1, index_batch);
    if (ret == 0) {
      break;
    } else {
      index_batch = index_batch / 2;
    }
  }
  while (tile_size >= 1) {
    int ret = kernel(stream, ppl_module, tile_size, index_batch);
    if (ret == 0) {
      return;
    } else {
      tile_size = tile_size / 2;
      continue;
    }
  }
}
#endif
namespace at {
Tensor fused_moe_grouped_topk(Tensor &topk_experts_res,
                              Tensor &topk_weights_res_bf16, Tensor &left_bf16,
                              Tensor &right_bf16, Tensor &topk_weights_res,
                              Tensor &left, Tensor &right, Tensor &max,
                              Tensor &matmul_res, Tensor &softmax_res) {
  TIMING_START;
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

  auto stream = c10_tpu::getCurrentTPUStream();
  tpudnnStatus_t status = tpudnnFusedMoEGroupedTopkMultiCoreAsync(
      stream, tpu::TPUGenerateTpudnnTensor(stream, topk_experts_res),
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
  TIMING_END;
  return topk_experts_res;
}

Tensor fused_moe_fused_experts(
    Tensor &output, Tensor &input, const c10::optional<Tensor> &output_sample,
    const c10::optional<Tensor> &input_sample, Tensor &gate_weights,
    Tensor &up_weights, Tensor &down_weights,
    const c10::optional<Tensor> &gate_scales,
    const c10::optional<Tensor> &up_scales,
    const c10::optional<Tensor> &down_scales, Tensor &select_experts,
    Tensor &routing_weights, const c10::optional<Tensor> &num_select_experts,
    const c10::optional<Tensor> &select_experts_middle,
    const c10::optional<Tensor> &routing_weights_middle, int64_t blocksize,
    int64_t num_experts, int64_t num_experts_per_tok, bool use_grouped_topk,
    int64_t num_expert_group, int64_t topk_group,
    const c10::optional<Tensor> &silu, const c10::optional<Tensor> &sigmoid,
    const c10::optional<Tensor> &m0, bool save_mid_res) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(output);
  CHECK_TENSOR_IN_DEVICE(input);
  if (output_sample.has_value()) CHECK_TENSOR_IN_DEVICE(output_sample.value());
  if (input_sample.has_value()) CHECK_TENSOR_IN_DEVICE(input_sample.value());
  CHECK_TENSOR_IN_DEVICE(gate_weights);
  CHECK_TENSOR_IN_DEVICE(up_weights);
  CHECK_TENSOR_IN_DEVICE(down_weights);
  if (gate_scales.has_value()) CHECK_TENSOR_IN_DEVICE(gate_scales.value());
  if (up_scales.has_value()) CHECK_TENSOR_IN_DEVICE(up_scales.value());
  if (down_scales.has_value()) CHECK_TENSOR_IN_DEVICE(down_scales.value());
  CHECK_TENSOR_IN_DEVICE(select_experts);
  TORCH_CHECK(select_experts.dtype() == torch::kInt32,
              "select_experts must be int32 dtype");
  CHECK_TENSOR_IN_DEVICE(routing_weights);
  if (num_select_experts.has_value()) {
    CHECK_TENSOR_IN_DEVICE(num_select_experts.value());
    TORCH_CHECK(num_select_experts.value().dtype() == torch::kInt64,
                "num_select_experts must be int64 dtype");
  }
  if (select_experts_middle.has_value())
    CHECK_TENSOR_IN_DEVICE(select_experts_middle.value());
  if (routing_weights_middle.has_value())
    CHECK_TENSOR_IN_DEVICE(routing_weights_middle.value());

  if (silu.has_value()) CHECK_TENSOR_IN_DEVICE(silu.value());
  if (sigmoid.has_value()) CHECK_TENSOR_IN_DEVICE(sigmoid.value());
  if (m0.has_value()) CHECK_TENSOR_IN_DEVICE(m0.value());

// #ifdef USING_PPL
//   if (usePPLKernels()) {
//     at::ScalarType dtype = input.scalar_type();
//     at::ScalarType weight_dtype = gate_weights.scalar_type();
//     int batch = input.size(0);
//     int input_w = input.size(1);
//     int middle_w = gate_weights.size(1);
//     fusedMoE_impl(
//         reinterpret_cast<uint64_t>(output.data_ptr()),
//         reinterpret_cast<uint64_t>(input.data_ptr()),
//         reinterpret_cast<uint64_t>(gate_weights.data_ptr()),
//         reinterpret_cast<uint64_t>(up_weights.data_ptr()),
//         reinterpret_cast<uint64_t>(down_weights.data_ptr()),
//         reinterpret_cast<uint64_t>(gate_scales.value().data_ptr()),
//         reinterpret_cast<uint64_t>(up_scales.value().data_ptr()),
//         reinterpret_cast<uint64_t>(down_scales.value().data_ptr()),
//         reinterpret_cast<uint64_t>(select_experts.data_ptr()),
//         reinterpret_cast<uint64_t>(routing_weights.data_ptr()),
//         reinterpret_cast<uint64_t>(select_experts_middle.value().data_ptr()),
//         reinterpret_cast<uint64_t>(routing_weights_middle.value().data_ptr()),
//         blocksize, num_experts, num_experts_per_tok, batch, input_w, middle_w,
//         dtype, weight_dtype, 1);
//   } else
// #endif
  // {
    auto stream = c10_tpu::getCurrentTPUStream();
    tpudnnStatus_t status = tpudnnFusedMoEFusedExpertsMultiCoreAsync(
        stream, tpu::TPUGenerateTpudnnTensor(stream, output),
        tpu::TPUGenerateTpudnnTensor(stream, input),
        output_sample.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream, output_sample.value())
            : tpudnnUndefinedTensor(),
        input_sample.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream, input_sample.value())
            : tpudnnUndefinedTensor(),
        tpu::TPUGenerateTpudnnTensor(stream, gate_weights),
        tpu::TPUGenerateTpudnnTensor(stream, up_weights),
        tpu::TPUGenerateTpudnnTensor(stream, down_weights),
        gate_scales.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream, gate_scales.value())
            : tpudnnUndefinedTensor(),
        up_scales.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream, up_scales.value())
            : tpudnnUndefinedTensor(),
        down_scales.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream, down_scales.value())
            : tpudnnUndefinedTensor(),
        tpu::TPUGenerateTpudnnTensor(stream, select_experts),
        tpu::TPUGenerateTpudnnTensor(stream, routing_weights),
        num_select_experts.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream, num_select_experts.value())
            : tpudnnUndefinedTensor(),
        select_experts_middle.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream,
                                           select_experts_middle.value())
            : tpudnnUndefinedTensor(),
        routing_weights_middle.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream,
                                           routing_weights_middle.value())
            : tpudnnUndefinedTensor(),
        blocksize, num_experts, num_experts_per_tok, use_grouped_topk,
        num_expert_group, topk_group,
        silu.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, silu.value())
                         : tpudnnUndefinedTensor(),
        sigmoid.has_value()
            ? tpu::TPUGenerateTpudnnTensor(stream, sigmoid.value())
            : tpudnnUndefinedTensor(),
        m0.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, m0.value())
                       : tpudnnUndefinedTensor(),
        save_mid_res);

    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  // }

  TIMING_END;
  return output;
}
Tensor fused_moe_fused_experts_v2(
    Tensor &output, Tensor &input, const c10::optional<Tensor> &output_sample,
    const c10::optional<Tensor> &input_sample, Tensor &gate_weights,
    Tensor &up_weights, Tensor &down_weights,
    const c10::optional<Tensor> &gate_scales,
    const c10::optional<Tensor> &up_scales,
    const c10::optional<Tensor> &down_scales, Tensor &select_experts,
    Tensor &routing_weights, const c10::optional<Tensor> &num_select_experts,
    const c10::optional<Tensor> &select_experts_middle,
    const c10::optional<Tensor> &routing_weights_middle,
    const c10::optional<Tensor> &gather_buffer,
    const c10::optional<Tensor> &scatter_buffer, int64_t blocksize,
    int64_t num_experts, int64_t num_experts_per_tok, bool use_grouped_topk,
    int64_t num_expert_group, int64_t topk_group,
    const c10::optional<Tensor> &silu, const c10::optional<Tensor> &sigmoid,
    const c10::optional<Tensor> &m0, bool save_mid_res) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(output);
  CHECK_TENSOR_IN_DEVICE(input);
  if (output_sample.has_value()) CHECK_TENSOR_IN_DEVICE(output_sample.value());
  if (input_sample.has_value()) CHECK_TENSOR_IN_DEVICE(input_sample.value());
  CHECK_TENSOR_IN_DEVICE(gate_weights);
  CHECK_TENSOR_IN_DEVICE(up_weights);
  CHECK_TENSOR_IN_DEVICE(down_weights);
  if (gate_scales.has_value()) CHECK_TENSOR_IN_DEVICE(gate_scales.value());
  if (up_scales.has_value()) CHECK_TENSOR_IN_DEVICE(up_scales.value());
  if (down_scales.has_value()) CHECK_TENSOR_IN_DEVICE(down_scales.value());
  CHECK_TENSOR_IN_DEVICE(select_experts);
  TORCH_CHECK(select_experts.dtype() == torch::kInt32,
              "select_experts must be int32 dtype");
  CHECK_TENSOR_IN_DEVICE(routing_weights);
  if (num_select_experts.has_value()) {
    CHECK_TENSOR_IN_DEVICE(num_select_experts.value());
    TORCH_CHECK(num_select_experts.value().dtype() == torch::kInt64,
                "num_select_experts must be int64 dtype");
  }
  if (select_experts_middle.has_value())
    CHECK_TENSOR_IN_DEVICE(select_experts_middle.value());
  if (routing_weights_middle.has_value())
    CHECK_TENSOR_IN_DEVICE(routing_weights_middle.value());

  if (gather_buffer.has_value()) CHECK_TENSOR_IN_DEVICE(gather_buffer.value());
  if (scatter_buffer.has_value())
    CHECK_TENSOR_IN_DEVICE(scatter_buffer.value());
  if (silu.has_value()) CHECK_TENSOR_IN_DEVICE(silu.value());
  if (sigmoid.has_value()) CHECK_TENSOR_IN_DEVICE(sigmoid.value());
  if (m0.has_value()) CHECK_TENSOR_IN_DEVICE(m0.value());

  #ifdef USING_PPL
  if (usePPLKernels()) {
    at::ScalarType dtype = input.scalar_type();
    at::ScalarType weight_dtype = gate_weights.scalar_type();
    int batch = input.size(0);
    int input_w = input.size(1);
    int middle_w = gate_weights.size(1);
    fusedMoE_impl(
        reinterpret_cast<uint64_t>(output.data_ptr()),
        reinterpret_cast<uint64_t>(input.data_ptr()),
        reinterpret_cast<uint64_t>(gate_weights.data_ptr()),
        reinterpret_cast<uint64_t>(up_weights.data_ptr()),
        reinterpret_cast<uint64_t>(down_weights.data_ptr()),
        reinterpret_cast<uint64_t>(gate_scales.value().data_ptr()),
        reinterpret_cast<uint64_t>(up_scales.value().data_ptr()),
        reinterpret_cast<uint64_t>(down_scales.value().data_ptr()),
        reinterpret_cast<uint64_t>(select_experts.data_ptr()),
        reinterpret_cast<uint64_t>(routing_weights.data_ptr()),
        reinterpret_cast<uint64_t>(select_experts_middle.value().data_ptr()),
        reinterpret_cast<uint64_t>(routing_weights_middle.value().data_ptr()),
        reinterpret_cast<uint64_t>(gather_buffer.value().data_ptr()),
        reinterpret_cast<uint64_t>(scatter_buffer.value().data_ptr()),
        blocksize, num_experts, num_experts_per_tok, batch, input_w, middle_w,
        dtype, weight_dtype, 1);
  } else  
  #endif
  {
    assert(0 && "fused_moe_fused_experts_v2 only support PPL");
  }

  // TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  return output;
  }
}  // namespace at
