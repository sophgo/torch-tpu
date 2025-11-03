#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

#ifdef USING_PPL
#include "mla_decode.h"
#include "mla_prefill.h"

template <typename scalar_t>
static void latent_attention_fp8_impl(
    uint64_t Q_addr, uint64_t KV_addr, uint64_t PE_addr, uint64_t KVcache_addr,
    uint64_t PEcache_addr, uint64_t KVU_addr, uint64_t RoPE_cos_addr,
    uint64_t RoPE_sin_addr, uint64_t WUQ_addr, uint64_t WUKV_addr,
    uint64_t Mask_addr, uint64_t Y_addr, uint64_t WUQ_scale_addr,
    uint64_t WUKV_scale_addr, uint64_t block_table_addr,
    uint64_t save_slot_addr, int num_heads, int qk_nope_head_dim,
    int qk_rope_head_dim, int v_head_dim, int q_lora_rank, int kv_lora_rank,
    float softmax_scale, int mask_max, int quant_block_size,
    int max_paged_block_num, int paged_cache_block_size, int max_cache_size,
    int attention_mode, bool has_mask, int batch, int *seqlen)
{
    auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
        if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            if (attention_mode == PAGED_ATTENTION_DECODE ||
                attention_mode == NORMAL_ATTENTION_DECODE) {
                return mla_decode_bf16_fp8e4m3(
                    stream,
#ifndef BACKEND_SG2260
                    ppl_module,
#endif
                    Q_addr, KV_addr, PE_addr, KVcache_addr,
                    PEcache_addr, KVU_addr, RoPE_cos_addr, RoPE_sin_addr,
                    WUQ_addr, WUKV_addr, Mask_addr, Y_addr, WUQ_scale_addr,
                    WUKV_scale_addr, block_table_addr, save_slot_addr,
                    num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                    q_lora_rank, kv_lora_rank, softmax_scale, mask_max,
                    quant_block_size, max_paged_block_num,
                    paged_cache_block_size, max_cache_size, attention_mode,
                    has_mask, batch, seqlen);
            } else if (attention_mode == PAGED_ATTENTION_PREFILL ||
                       attention_mode == NORMAL_ATTENTION_PREFILL) {
                return mla_prefill_bf16_fp8e4m3(
                    stream,
#ifndef BACKEND_SG2260
                    ppl_module,
#endif
                    Q_addr, KV_addr, PE_addr, KVcache_addr,
                    PEcache_addr, KVU_addr, RoPE_cos_addr, RoPE_sin_addr,
                    WUQ_addr, WUKV_addr, Mask_addr, Y_addr, WUQ_scale_addr,
                    WUKV_scale_addr, block_table_addr, save_slot_addr,
                    num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                    q_lora_rank, kv_lora_rank, softmax_scale, mask_max,
                    quant_block_size, max_paged_block_num,
                    paged_cache_block_size, max_cache_size, attention_mode,
                    has_mask, batch, seqlen);
            }
        }
        return -1;
    };

    auto stream = c10_tpu::getCurrentTPUStream();
    tpuKernelModule_t ppl_module = getPplModule();

    int ret = kernel(stream, ppl_module);
    if (ret != 0) {
        TORCH_CHECK(false, "MLA kernel failed");
    }
    return;
}

template <typename scalar_t>
static void latent_attention_impl(
    uint64_t Q_addr, uint64_t KV_addr, uint64_t PE_addr, uint64_t KVcache_addr,
    uint64_t PEcache_addr, uint64_t KVU_addr, uint64_t RoPE_cos_addr,
    uint64_t RoPE_sin_addr, uint64_t WUQ_addr, uint64_t WUKV_addr,
    uint64_t Mask_addr, uint64_t Y_addr, uint64_t block_table_addr,
    uint64_t save_slot_addr, int num_heads, int qk_nope_head_dim,
    int qk_rope_head_dim, int v_head_dim, int q_lora_rank, int kv_lora_rank,
    float softmax_scale, int mask_max, int max_paged_block_num,
    int paged_cache_block_size, int max_cache_size, int attention_mode,
    bool has_mask, int batch, int *seqlen)
{
    auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
        if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            if (attention_mode == PAGED_ATTENTION_DECODE ||
                attention_mode == NORMAL_ATTENTION_DECODE) {
                return mla_decode_bf16(
                    stream,
#ifndef BACKEND_SG2260
                    ppl_module,
#endif
                    Q_addr, KV_addr, PE_addr, KVcache_addr,
                    PEcache_addr, KVU_addr, RoPE_cos_addr, RoPE_sin_addr,
                    WUQ_addr, WUKV_addr, Mask_addr, Y_addr, block_table_addr,
                    save_slot_addr, max_paged_block_num, paged_cache_block_size,
                    num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                    qk_rope_head_dim, v_head_dim, softmax_scale, has_mask,
                    batch, mask_max, max_cache_size, attention_mode, seqlen);
            } else if (attention_mode == PAGED_ATTENTION_PREFILL ||
                       attention_mode == NORMAL_ATTENTION_PREFILL) {
                return mla_prefill_bf16(
                    stream,
#ifndef BACKEND_SG2260
                    ppl_module,
#endif
                    Q_addr, KV_addr, PE_addr, KVcache_addr,
                    PEcache_addr, KVU_addr, RoPE_cos_addr, RoPE_sin_addr,
                    WUQ_addr, WUKV_addr, Mask_addr, Y_addr, block_table_addr,
                    save_slot_addr, max_paged_block_num, paged_cache_block_size,
                    num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                    qk_rope_head_dim, v_head_dim, softmax_scale, has_mask,
                    batch, mask_max, max_cache_size, attention_mode, seqlen);
            }
        }
        return -1;
    };

    auto stream = c10_tpu::getCurrentTPUStream();
    tpuKernelModule_t ppl_module = getPplModule();

    int ret = kernel(stream, ppl_module);
    if (ret != 0) {
        TORCH_CHECK(false, "MLA kernel failed");
    }
    return;
}

#endif

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
        const c10::optional<Tensor> &KVU,
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
		TIMING_START;
		if (attention_mode == NORMAL_ATTENTION_DECODE || attention_mode == NORMAL_ATTENTION_PREFILL){
			TORCH_CHECK (input_lengths.dtype() == torch::kInt32,
						"MLA input lenghts must be int32 dtype");
			TORCH_CHECK ( input_lengths.device().type() == DeviceType::CPU, 
						"MLA input lenghts must on CPU device" );
		}
#ifdef USING_PPL
        if (usePPLKernels())
        {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, Q.scalar_type(), "latent_attention", [&] {
            latent_attention_impl<scalar_t>(
                reinterpret_cast<uint64_t>(Q.data_ptr()),
                reinterpret_cast<uint64_t>(KV.data_ptr()),
                reinterpret_cast<uint64_t>(PE.data_ptr()),
                reinterpret_cast<uint64_t>(KVcache.data_ptr()),
                reinterpret_cast<uint64_t>(PEcache.data_ptr()),
                KVU.has_value() ? reinterpret_cast<uint64_t>(KVU->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(cos.data_ptr()),
                reinterpret_cast<uint64_t>(sin.data_ptr()),
                reinterpret_cast<uint64_t>(WUQ.data_ptr()),
                reinterpret_cast<uint64_t>(WUKV.data_ptr()),
                mask.has_value() ? reinterpret_cast<uint64_t>(mask->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(OUT.data_ptr()),
                0,
                0,
                head,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                q_lora_rank,
                kv_lora_rank,
                C,
                mask_size,
                0,
                0,
                max_cache_size,
                attention_mode,
                mask.has_value(),
                (int)(input_lengths.nbytes() / 4),
                (int *)input_lengths.data_ptr()
            );
        
    });
    } else
#endif
    {
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
            KVU.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, KVU.value()) : tpudnnUndefinedTensor(),
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
    }
        TIMING_END;
        return OUT;
    }

    Tensor paged_latent_attention(
        Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE,
        Tensor &WUQ, Tensor &WUKV, Tensor &KVcache,
        Tensor &PEcache, Tensor &cos, Tensor &sin,
        const c10::optional<Tensor> &block_table,
        Tensor &save_slots,
        const c10::optional<Tensor> &KVU,
        const c10::optional<Tensor> &mask, // decode: None
        const Tensor &input_lengths, const Tensor &cache_lengths, int64_t head,
        int64_t q_lora_rank, int64_t kv_lora_rank,
        int64_t qk_nope_head_dim, int64_t qk_rope_head_dim,
        int64_t v_head_dim, int64_t mask_size,
        int64_t max_paged_block_num, int64_t paged_cache_block_size,
        double C, int64_t attention_mode // prefille 0, decode 1
    ) {
      TIMING_START;
      if (attention_mode == PAGED_ATTENTION_DECODE ||
          attention_mode == PAGED_ATTENTION_PREFILL) {
        TORCH_CHECK(input_lengths.dtype() == torch::kInt32,
                    "MLA input lenghts must be int32 dtype");
        TORCH_CHECK(input_lengths.device().type() == DeviceType::CPU,
                    "MLA input lenghts must on CPU device");
        TORCH_CHECK(input_lengths.is_contiguous(),
                    "MLA input lengths must be contiguous tensor");
      }
#ifdef USING_PPL
    if (usePPLKernels())
    {
        int max_cache_size = 0;
        int batch = (int)(input_lengths.nbytes() / 4);
        std::vector<uint32_t> seq(2 * batch, 0);
        memcpy(seq.data(), input_lengths.data_ptr(), batch * sizeof(int));
        memcpy(seq.data() + batch * sizeof(int), cache_lengths.data_ptr(), batch * sizeof(int));
            AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, Q.scalar_type(), "paged_latent_attention", [&] {
            latent_attention_impl<scalar_t>(
                reinterpret_cast<uint64_t>(Q.data_ptr()),
                reinterpret_cast<uint64_t>(KV.data_ptr()),
                reinterpret_cast<uint64_t>(PE.data_ptr()),
                reinterpret_cast<uint64_t>(KVcache.data_ptr()),
                reinterpret_cast<uint64_t>(PEcache.data_ptr()),
                KVU.has_value() ? reinterpret_cast<uint64_t>(KVU->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(cos.data_ptr()),
                reinterpret_cast<uint64_t>(sin.data_ptr()),
                reinterpret_cast<uint64_t>(WUQ.data_ptr()),
                reinterpret_cast<uint64_t>(WUKV.data_ptr()),
                mask.has_value() ? reinterpret_cast<uint64_t>(mask->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(OUT.data_ptr()),
                block_table.has_value() ? reinterpret_cast<uint64_t>(block_table->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(save_slots.data_ptr()),
                head,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                q_lora_rank,
                kv_lora_rank,
                C,
                mask_size,
                max_paged_block_num,
                paged_cache_block_size,
                max_cache_size,
                attention_mode,
                mask.has_value(),
                (int)(input_lengths.nbytes() / 4),
                (int *)seq.data()
            );
        
    });
    } else
#endif
    {
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
          KVU.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, KVU.value()) : tpudnnUndefinedTensor(),
          mask.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, mask.value())
                           : tpudnnUndefinedTensor(),
          block_table.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, block_table.value())
                           : tpudnnUndefinedTensor(),//fetch slots
          tpu::TPUGenerateTpudnnTensor(stream, save_slots),
          max_paged_block_num, paged_cache_block_size,
          (int *)input_lengths.data_ptr(), (int *)cache_lengths.data_ptr(),
          (int)(input_lengths.nbytes() / 4), head, q_lora_rank, kv_lora_rank,
          qk_nope_head_dim, qk_rope_head_dim, v_head_dim, mask_size,
          C, (AttentionMode_t)attention_mode);
      TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }
      TIMING_END;
      return OUT;
    }

    Tensor paged_latent_attention_fp8(
        Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE, Tensor &WUQ,
        Tensor &WUKV, Tensor &KVcache, Tensor &PEcache, Tensor &cos,
        Tensor &sin, Tensor &WUQ_scale, Tensor &WUKV_scale,
        const c10::optional<Tensor> &block_table,
        Tensor &save_slots,
        const c10::optional<Tensor> &KVU,
        const c10::optional<Tensor> &mask, // decode: None
        const Tensor &seqlen, const Tensor &cache_seqlen,
        const c10::optional<Tensor> &topk_indices,
        int64_t num_heads, int64_t q_lora_rank,
        int64_t kv_lora_rank, int64_t qk_nope_head_dim,
        int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
        int64_t quant_block_size, int64_t max_paged_block_num,
        int64_t paged_cache_block_size, int64_t topk_size,
        double softmax_scale, int64_t attention_mode // prefille 0, decode 1
    ) {
        TIMING_START;
        if (attention_mode == PAGED_ATTENTION_DECODE ||
            attention_mode == PAGED_ATTENTION_PREFILL) {
            TORCH_CHECK(seqlen.dtype() == torch::kInt32);
            TORCH_CHECK(seqlen.is_contiguous(),
            "MLA seqlen must be contiguous tensor");
        }
#ifdef USING_PPL
    if (usePPLKernels())
    {

        int max_cache_size = 0;
        int batch = (int)(seqlen.nbytes() / 4);
        std::vector<uint32_t> seq(2 * batch, 0);
        memcpy(seq.data(), seqlen.data_ptr(), batch * sizeof(uint32_t));
        memcpy(seq.data() + batch * sizeof(uint32_t), cache_seqlen.data_ptr(), batch * sizeof(uint32_t));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, Q.scalar_type(), "paged_latent_attention_fp8", [&] {
            latent_attention_fp8_impl<scalar_t>(
                reinterpret_cast<uint64_t>(Q.data_ptr()),
                reinterpret_cast<uint64_t>(KV.data_ptr()),
                reinterpret_cast<uint64_t>(PE.data_ptr()),
                reinterpret_cast<uint64_t>(KVcache.data_ptr()),
                reinterpret_cast<uint64_t>(PEcache.data_ptr()),
                KVU.has_value() ? reinterpret_cast<uint64_t>(KVU->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(cos.data_ptr()),
                reinterpret_cast<uint64_t>(sin.data_ptr()),
                reinterpret_cast<uint64_t>(WUQ.data_ptr()),
                reinterpret_cast<uint64_t>(WUKV.data_ptr()),
                mask.has_value() ? reinterpret_cast<uint64_t>(mask->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(OUT.data_ptr()),
                reinterpret_cast<uint64_t>(WUQ_scale.data_ptr()),
                reinterpret_cast<uint64_t>(WUKV_scale.data_ptr()),
                block_table.has_value() ? reinterpret_cast<uint64_t>(block_table->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(save_slots.data_ptr()),
                num_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                q_lora_rank,
                kv_lora_rank,
                softmax_scale,
                mask_size,
                quant_block_size,
                max_paged_block_num,
                paged_cache_block_size,
                max_cache_size,
                attention_mode,
                mask.has_value(),
                (int)(seqlen.nbytes() / 4),
                (int *)seq.data()
            );
        
    });
    } else
#endif
    {
        bool cpu_lengths = seqlen.device().type() == DeviceType::CPU;
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
            KVU.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, KVU.value()) : tpudnnUndefinedTensor(),
            mask.has_value()
                ? tpu::TPUGenerateTpudnnTensor(stream, mask.value())
                : tpudnnUndefinedTensor(),
            tpu::TPUGenerateTpudnnTensor(stream, WUQ_scale),
            tpu::TPUGenerateTpudnnTensor(stream, WUKV_scale),
            cpu_lengths ? tpudnnUndefinedTensor() : tpu::TPUGenerateTpudnnTensor(stream, seqlen),
            cpu_lengths ? tpudnnUndefinedTensor() : tpu::TPUGenerateTpudnnTensor(stream, cache_seqlen),
            block_table.has_value()
                ? tpu::TPUGenerateTpudnnTensor(stream, block_table.value())
                : tpudnnUndefinedTensor(),//block_table
            tpu::TPUGenerateTpudnnTensor(stream, save_slots), //save slots
            topk_indices.has_value()
                ? tpu::TPUGenerateTpudnnTensor(stream, topk_indices.value())
                : tpudnnUndefinedTensor(),//topk_indices
            cpu_lengths ? (const int *)seqlen.data_ptr() : nullptr,
            cpu_lengths ? (const int *)cache_seqlen.data_ptr() : nullptr,
            (int)(seqlen.nbytes() / 4), num_heads, qk_nope_head_dim,
            qk_rope_head_dim, v_head_dim, q_lora_rank, kv_lora_rank,
            mask_size, quant_block_size, max_paged_block_num, paged_cache_block_size,
            topk_size, softmax_scale, true, (AttentionMode_t)attention_mode);
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }
        TIMING_END;
        return OUT;
    }

    Tensor latent_attention_fp8(
        Tensor &OUT, Tensor &Q, Tensor &KV, Tensor &PE, Tensor &WUQ,
        Tensor &WUKV, Tensor &KVcache, Tensor &PEcache, Tensor &cos,
        Tensor &sin, Tensor &WUQ_scale, Tensor &WUKV_scale,
        const c10::optional<Tensor> &KVU,
        const c10::optional<Tensor> &mask, // decode: None
        const Tensor &seqlen, int64_t num_heads, int64_t q_lora_rank,
        int64_t kv_lora_rank, int64_t qk_nope_head_dim,
        int64_t qk_rope_head_dim, int64_t v_head_dim, int64_t mask_size,
        int64_t quant_block_size, int64_t max_cache_size,
        double softmax_scale,
        int64_t attention_mode // prefille 0, decode 1
    ) {
        TIMING_START;
        if (attention_mode == NORMAL_ATTENTION_PREFILL ||
            attention_mode == NORMAL_ATTENTION_DECODE) {
        TORCH_CHECK(
            seqlen.dtype() == torch::kInt32 &&
                seqlen.device().type() == DeviceType::CPU,
            "MLA input seqlen must be int32 dtype && on CPU device");
        }
#ifdef USING_PPL
    if (usePPLKernels())
    {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, Q.scalar_type(), "latent_attention_fp8", [&] {
            latent_attention_fp8_impl<scalar_t>(
                reinterpret_cast<uint64_t>(Q.data_ptr()),
                reinterpret_cast<uint64_t>(KV.data_ptr()),
                reinterpret_cast<uint64_t>(PE.data_ptr()),
                reinterpret_cast<uint64_t>(KVcache.data_ptr()),
                reinterpret_cast<uint64_t>(PEcache.data_ptr()),
                KVU.has_value() ? reinterpret_cast<uint64_t>(KVU->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(cos.data_ptr()),
                reinterpret_cast<uint64_t>(sin.data_ptr()),
                reinterpret_cast<uint64_t>(WUQ.data_ptr()),
                reinterpret_cast<uint64_t>(WUKV.data_ptr()),
                mask.has_value() ? reinterpret_cast<uint64_t>(mask->data_ptr()) : 0,
                reinterpret_cast<uint64_t>(OUT.data_ptr()),
                reinterpret_cast<uint64_t>(WUQ_scale.data_ptr()),
                reinterpret_cast<uint64_t>(WUKV_scale.data_ptr()),
                0,
                0,
                num_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                q_lora_rank,
                kv_lora_rank,
                softmax_scale,
                mask_size,
                quant_block_size,
                0,
                0,
                max_cache_size,
                attention_mode,
                false,
                (int)(seqlen.nbytes() / 4),
                (int *)seqlen.data_ptr()
            );
        
    });
    } else
#endif
    {
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
            KVU.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, KVU.value()) : tpudnnUndefinedTensor(),
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
        TIMING_END;
    }
        return OUT;
    }
}
