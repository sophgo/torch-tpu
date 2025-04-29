#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor latent_attention(
        Tensor &OUT,
        Tensor &Q,
        Tensor &KV,
        Tensor &PE,
        Tensor &WUQ,
        Tensor &WUKV,
        Tensor &KVcache,
        Tensor &PEcache,
        Tensor &cos,
        Tensor &sin,
        const c10::optional<Tensor> &mask, // decode: None
        const Tensor &input_lengths, 
        int64_t head,
        int64_t q_lora_rank,
        int64_t kv_lora_rank,
        int64_t qk_nope_head_dim,
        int64_t qk_rope_head_dim,
        int64_t v_head_dim,
        int64_t mask_size,
        int64_t max_cache_size,
        double C,
        int64_t attention_mode // prefille 0, decode 1
    ) {

		if (attention_mode == NORMAL_ATTENTION_DECODE || attention_mode == NORMAL_ATTENTION_PREFILL){
			TORCH_CHECK (tpu::TPUConvertDtype<SgdnnDataType_t>(input_lengths.dtype()) == SGDNN_DTYPE_INT32,
						"MLA input lenghts must be int32 dtype");
			TORCH_CHECK ( input_lengths.device().type() == DeviceType::CPU, 
						"MLA input lenghts must on CPU device" );
		}
		
		TIMING_START;
  		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnLatentAttentionAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, OUT),
            tpu::TPUGenerateTpudnnTensor(stream, Q),
            tpu::TPUGenerateTpudnnTensor(stream, KV),
            tpu::TPUGenerateTpudnnTensor(stream, PE),
            tpu::TPUGenerateTpudnnTensor(stream, WUQ),
            tpu::TPUGenerateTpudnnTensor(stream, WUKV),
            tpu::TPUGenerateTpudnnTensor(stream, KVcache),
            tpu::TPUGenerateTpudnnTensor(stream, PEcache),
            tpu::TPUGenerateTpudnnTensor(stream, cos),
            tpu::TPUGenerateTpudnnTensor(stream, sin),
            mask.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, mask.value()) : tpudnnUndefinedTensor(),
            (int*)input_lengths.data_ptr(),
            (int)(input_lengths.nbytes()/4),
            head,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            mask_size,
            max_cache_size,
            C,
            (AttentionMode_t)attention_mode);
        TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
        TIMING_END( tpu::MLA );
        return OUT;
    }

    Tensor paged_latent_attention(
        Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE,
        Tensor &WUQ, Tensor &WUKV, Tensor &KVcache,
        Tensor &PEcache, Tensor &cos, Tensor &sin,
        const c10::optional<Tensor> &fetch_slots,
        Tensor &save_slots,
        const c10::optional<Tensor> &mask, // decode: None
        const Tensor &input_lengths, int64_t head,
        int64_t q_lora_rank, int64_t kv_lora_rank,
        int64_t qk_nope_head_dim, int64_t qk_rope_head_dim,
        int64_t v_head_dim, int64_t mask_size,
        int64_t slots_size, int64_t paged_cache_block_size,
        double C, int64_t attention_mode // prefille 0, decode 1
    ) {

      if (attention_mode == PAGED_ATTENTION_DECODE ||
          attention_mode == PAGED_ATTENTION_PREFILL) {
        TORCH_CHECK(tpu::TPUConvertDtype<SgdnnDataType_t>(
                        input_lengths.dtype()) == SGDNN_DTYPE_INT32,
                    "MLA input lenghts must be int32 dtype");
        TORCH_CHECK(input_lengths.device().type() == DeviceType::CPU,
                    "MLA input lenghts must on CPU device");
      }

      TIMING_START;
      auto stream = c10_tpu::getCurrentTPUStream();
      auto status = tpudnnPagedLatentAttentionAsync(
          stream, tpu::TPUGenerateTpudnnTensor(stream, OUT),
          tpu::TPUGenerateTpudnnTensor(stream, Q),
          tpu::TPUGenerateTpudnnTensor(stream, KV),
          tpu::TPUGenerateTpudnnTensor(stream, PE),
          tpu::TPUGenerateTpudnnTensor(stream, WUQ),
          tpu::TPUGenerateTpudnnTensor(stream, WUKV),
          tpu::TPUGenerateTpudnnTensor(stream, KVcache),
          tpu::TPUGenerateTpudnnTensor(stream, PEcache),
          tpu::TPUGenerateTpudnnTensor(stream, cos),
          tpu::TPUGenerateTpudnnTensor(stream, sin),
          mask.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, mask.value())
                           : tpudnnUndefinedTensor(),
          fetch_slots.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, fetch_slots.value())
                           : tpudnnUndefinedTensor(),//fetch slots
          tpu::TPUGenerateTpudnnTensor(stream, save_slots),
          slots_size, paged_cache_block_size, (int *)input_lengths.data_ptr(),
          (int)(input_lengths.nbytes() / 4), head, q_lora_rank, kv_lora_rank,
          qk_nope_head_dim, qk_rope_head_dim, v_head_dim, mask_size,
          C, (AttentionMode_t)attention_mode);
      TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
      TIMING_END(tpu::MLA);
      return OUT;
    }

        Tensor paged_latent_attention_fp8(
            Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE, Tensor &WUQ,
            Tensor &WUKV, Tensor &KVcache, Tensor &PEcache, Tensor &cos,
            Tensor &sin, Tensor &WUQ_scale, Tensor &WUKV_scale,
            const c10::optional<Tensor> &fetch_slots,
            Tensor &save_slots,
            const c10::optional<Tensor> &mask, // decode: None
            const Tensor &seqlen, int64_t num_heads, int64_t q_lora_rank,
            int64_t kv_lora_rank, int64_t qk_nope_head_dim,
            int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
            int64_t quant_block_size, int64_t slots_size,
            int64_t paged_cache_block_size, double softmax_scale,
            int64_t attention_mode // prefille 0, decode 1
        ) {

          if (attention_mode == PAGED_ATTENTION_DECODE ||
              attention_mode == PAGED_ATTENTION_PREFILL) {
            TORCH_CHECK(
                seqlen.dtype() == torch::kInt32 &&
                    seqlen.device().type() == DeviceType::CPU,
                "MLA input seqlen must be int32 dtype && on CPU device");
          }

          TIMING_START;
          auto stream = c10_tpu::getCurrentTPUStream();
          auto status = tpudnnPagedLatentAttentionFp8Async(
              stream, tpu::TPUGenerateTpudnnTensor(stream, OUT),
              tpu::TPUGenerateTpudnnTensor(stream, Q),
              tpu::TPUGenerateTpudnnTensor(stream, KV),
              tpu::TPUGenerateTpudnnTensor(stream, PE),
              tpu::TPUGenerateTpudnnTensor(stream, WUQ),
              tpu::TPUGenerateTpudnnTensor(stream, WUKV),
              tpu::TPUGenerateTpudnnTensor(stream, KVcache),
              tpu::TPUGenerateTpudnnTensor(stream, PEcache),
              tpu::TPUGenerateTpudnnTensor(stream, cos),
              tpu::TPUGenerateTpudnnTensor(stream, sin),
              mask.has_value()
                  ? tpu::TPUGenerateTpudnnTensor(stream, mask.value())
                  : tpudnnUndefinedTensor(),
              tpu::TPUGenerateTpudnnTensor(stream, WUQ_scale),
              tpu::TPUGenerateTpudnnTensor(stream, WUKV_scale),
              fetch_slots.has_value()
                    ? tpu::TPUGenerateTpudnnTensor(stream, fetch_slots.value())
                    : tpudnnUndefinedTensor(),//fetch slots
              tpu::TPUGenerateTpudnnTensor(stream, save_slots), //save slots
              (const int *)seqlen.data_ptr(),
              (int)(seqlen.nbytes() / 4), num_heads, qk_nope_head_dim,
              qk_rope_head_dim, v_head_dim, q_lora_rank, kv_lora_rank,
              mask_size, quant_block_size, slots_size, paged_cache_block_size,
              softmax_scale, true, (AttentionMode_t)attention_mode);
          TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
          TIMING_END(tpu::MLA);
          return OUT;
        }

        Tensor latent_attention_fp8(
            Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE, Tensor &WUQ,
            Tensor &WUKV, Tensor &KVcache, Tensor &PEcache, Tensor &cos,
            Tensor &sin, Tensor &WUQ_scale, Tensor &WUKV_scale,
            const c10::optional<Tensor> &mask, // decode: None
            const Tensor &seqlen, int64_t num_heads, int64_t q_lora_rank,
            int64_t kv_lora_rank, int64_t qk_nope_head_dim,
            int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
            int64_t quant_block_size, int64_t max_cache_size,
            double softmax_scale,
            int64_t attention_mode // prefille 0, decode 1
        ) {

          if (attention_mode == NORMAL_ATTENTION_PREFILL ||
              attention_mode == NORMAL_ATTENTION_DECODE) {
            TORCH_CHECK(
                seqlen.dtype() == torch::kInt32 &&
                    seqlen.device().type() == DeviceType::CPU,
                "MLA input seqlen must be int32 dtype && on CPU device");
          }

          TIMING_START;
          auto stream = c10_tpu::getCurrentTPUStream();
          auto status = tpudnnLatentAttentionFp8Async(
              stream, tpu::TPUGenerateTpudnnTensor(stream, OUT),
              tpu::TPUGenerateTpudnnTensor(stream, Q),
              tpu::TPUGenerateTpudnnTensor(stream, KV),
              tpu::TPUGenerateTpudnnTensor(stream, PE),
              tpu::TPUGenerateTpudnnTensor(stream, WUQ),
              tpu::TPUGenerateTpudnnTensor(stream, WUKV),
              tpu::TPUGenerateTpudnnTensor(stream, KVcache),
              tpu::TPUGenerateTpudnnTensor(stream, PEcache),
              tpu::TPUGenerateTpudnnTensor(stream, cos),
              tpu::TPUGenerateTpudnnTensor(stream, sin),
              mask.has_value()
                  ? tpu::TPUGenerateTpudnnTensor(stream, mask.value())
                  : tpudnnUndefinedTensor(),
              tpu::TPUGenerateTpudnnTensor(stream, WUQ_scale),
              tpu::TPUGenerateTpudnnTensor(stream, WUKV_scale),
              (const int *)seqlen.data_ptr(),
              (int)(seqlen.nbytes() / 4), num_heads, qk_nope_head_dim,
              qk_rope_head_dim, v_head_dim, q_lora_rank, kv_lora_rank,
              mask_size, quant_block_size, max_cache_size,
              softmax_scale, true, (AttentionMode_t)attention_mode);
          TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
          TIMING_END(tpu::MLA);
          return OUT;
        }
}
