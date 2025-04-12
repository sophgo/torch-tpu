
#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"
#include "ops.hpp"
#include "torch_tpu/csrc/aten/TPUNativeFunctions.h"
namespace at {
Tensor &dummy(Tensor &in) {
  CHECK_TENSOR_IN_DEVICE(in);
  TIMING_START;
#ifdef BACKEND_SG2260
  SAFE_CALL(sgdnnDummy(tpu::TPUGetDeviceResource(), true));
#endif
  TIMING_END(tpu::DUMMY);
  return in;
}
Tensor &dummy_no_kernel_launch(Tensor &in) {
  CHECK_TENSOR_IN_DEVICE(in);
  TIMING_START;
#ifdef BACKEND_SG2260
  SAFE_CALL(sgdnnDummy_WO_KERNEL_LAUNCH(tpu::TPUGetDeviceResource(), true));
#endif
  TIMING_END(tpu::DUMMY);
  return in;
}
} // namespace at

namespace at
{
	TORCH_LIBRARY(my_ops, m)
	{
		m.def("dummy", dummy);
		m.def("dummy_no_kernel_launch", dummy_no_kernel_launch);
		m.def("llava_mlp", llava_mlp);
		m.def("llava_attention", llava_attention);
		m.def("mla", mla);
		m.def("paged_latent_attention_fp8", paged_latent_attention_fp8);
		m.def("mlp_forward", mlp_forward);
		m.def("mlp_backward", mlp_backward);
		m.def("llama_mlp_forward", llama_mlp_forward);
		m.def("mlp_w8a16_dq_forward", mlp_w8a16_dq_forward);
		m.def("mm_w8a16_dq_forward", mm_w8a16_dq_forward);
		m.def("matmul_gptq_forward", matmul_gptq_forward);
		m.def("llama_mlp_gptq_forward", llama_mlp_gptq_forward);
		m.def("rmsnorm_forward", rmsnorm_forward);
		m.def("rmsnorm_backward", rmsnorm_backward);
		m.def("llama_attention", llama_attention);
		m.def("paged_attention", paged_attention);
		m.def("llama_attention_forward", llama_attention_forward);
		m.def("llama_attention_backward", llama_attention_backward);
		m.def("attn_forward", attn_forward);
		m.def("attn_backward", attn_backward);
		m.def("ln_mm_forward", ln_mm_forward);
		m.def("ln_mm_backward", ln_mm_backward);
		m.def("add_ln_mm_forward", add_ln_mm_forward);
		m.def("add_ln_mm_backward", add_ln_mm_backward);
		m.def("enable_pmu", enable_pmu);
		m.def("disable_pmu", disable_pmu);
		m.def("enable_profile", enable_profile);
		m.def("disable_profile", disable_profile);
		m.def("lora_matmul_forward",lora_matmul_forward);
		m.def("tgi_input_ids_update_decode_phase",TGI_input_ids_update_decode_phase);
		m.def("dynlib_execute",dynlib_execute);
		m.def("format_cast", FormatCast);
		
		m.def("fused_moe_grouped_topk", fused_moe_grouped_topk);
		m.def("fused_moe_fused_experts", fused_moe_fused_experts);
	}
}
