#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"
#include <ATen/TensorIndexing.h>
using namespace torch::indexing;

namespace at
{
	Tensor paged_attention(
		Tensor &OUT, // (tokens, heads, heads_size)
		Tensor &Q, // (tokens, heads, heads_size)
		Tensor &K, // (tokens, heads, heads_size)
		Tensor &V, // (tokens, heads, heads_size)
		Tensor &Kcache, // (blocks, block_size, heads, heads_size)
		Tensor &Vcache, // (blocks, block_size, heads, heads_size)
		const c10::optional<Tensor> &cos, // (tokens, 1, 128)
		const c10::optional<Tensor> &sin, // (tokens, 1, 128)
		const Tensor &input_lengths, // [10, 11, 8]
		const c10::optional<Tensor> &cache_lengths, // [5, 7, 5]
		const Tensor &save_slots, //  prefill : [[0, 1], [2, 0], [3, 0]]  decode : [[17], [35], [50]]
		const c10::optional<Tensor> &block_tables, // prefill : null, decode: [[0, 1], [2, 0], [3, 0]]
		const c10::optional<Tensor> &mask, // prefill: (max_length, max_length), decode: None
        int64_t rope_head_size,
		int64_t	slots_size, // prefill: save_slots.size(1) decode: block_tables.size(1)
		int64_t	fetch_size,
		int64_t mask_size, // mask_size
		int64_t block_size, // tokens num of one block
		double C, // softmax_scale
		int64_t attention_mode // prefille 2, decode 3
		)
	{
		TIMING_START;
		if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous()){
			// LOG(WARNING) << "llama_attention not contiguous, use QKV packed.";
		}
		if (!Kcache.is_contiguous() || !Vcache.is_contiguous()){
			LOG(WARNING) << "llama_attention not contiguous, change Kcache, Vcache to contiguous.";
			Kcache = Kcache.contiguous();
			Vcache = Vcache.contiguous();
		}

		CHECK_TENSOR_IN_DEVICE(OUT);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(Q);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(K);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(V);
		CHECK_TENSOR_IN_DEVICE(Kcache);
		CHECK_TENSOR_IN_DEVICE(Vcache);
		CHECK_TENSOR_IN_DEVICE(save_slots);
		if (cos.has_value())
			CHECK_TENSOR_IN_DEVICE(cos.value());
		if (sin.has_value())
			CHECK_TENSOR_IN_DEVICE(sin.value());
		if (block_tables.has_value()) {
			CHECK_TENSOR_IN_DEVICE(block_tables.value());
			TORCH_CHECK (block_tables.value().dtype() == torch::kInt32,
						"LLammaAttention block_tables must be int32 dtype");
		}
		if (mask.has_value())
			CHECK_TENSOR_IN_DEVICE(mask.value());
		if (attention_mode == 3 || attention_mode == 2){
			TORCH_CHECK (input_lengths.dtype() == torch::kInt32,
						"LLammaAttention input lenghts must be int32 dtype");
		}
        bool cpu_lengths = input_lengths.device().type() == DeviceType::CPU;
        if (cpu_lengths)
        {
		    auto kvcache_shape = Kcache.sizes();
		    int block_num = kvcache_shape[0];
		    int64_t block_need_num = 0;
		    int num_input_lengths = input_lengths.nbytes() / 4;
		    int* input_lengths_ptr = (int*)input_lengths.data_ptr();
		    for (int i = 0; i < num_input_lengths; ++i) {
		    	block_need_num += (input_lengths_ptr[i] + block_size - 1) / block_size;
		    }
		    TORCH_CHECK ( block_num >= block_need_num, "LLamatAttention KVCache block_num must be larger than ", block_need_num);
        }

  		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnPagedAttentionAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor(stream, OUT),
			tpu::TPUGenerateTpudnnTensor(stream, Q),
			tpu::TPUGenerateTpudnnTensor(stream, K),
			tpu::TPUGenerateTpudnnTensor(stream, V),
			tpu::TPUGenerateTpudnnTensor(stream, Kcache),
			tpu::TPUGenerateTpudnnTensor(stream, Vcache),
			cos.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, cos.value()) : tpudnnUndefinedTensor(),
			sin.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, sin.value()) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor(stream, save_slots),
			block_tables.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, block_tables.value()) : tpudnnUndefinedTensor(),
			mask.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, mask.value()) : tpudnnUndefinedTensor(),
			tpudnnUndefinedTensor(), tpudnnUndefinedTensor(), tpudnnUndefinedTensor(),
			cpu_lengths ? tpudnnUndefinedTensor() : tpu::TPUGenerateTpudnnTensor(stream, input_lengths),
			!cpu_lengths && cache_lengths.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, cache_lengths.value()) : tpudnnUndefinedTensor(),
            rope_head_size,
			cpu_lengths ? (int*)input_lengths.data_ptr() : nullptr,
			cpu_lengths && cache_lengths.has_value()  ? (int*)cache_lengths.value().data_ptr() : nullptr,
      	    (int)(input_lengths.nbytes()/4),
			slots_size,
			fetch_size,
			mask_size,
			block_size,
			C,
			attention_mode);
		TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
		TIMING_END;
		return OUT;
	}

	Tensor llama_attention(
		Tensor &OUT, // (tokens, heads, heads_size)
		Tensor &Q, // (tokens, heads, heads_size)
		Tensor &K, // (tokens, heads, heads_size)
		Tensor &V, // (tokens, heads, heads_size)
		Tensor &Kcache, // (blocks, block_size, heads, heads_size)
		Tensor &Vcache, // (blocks, block_size, heads, heads_size)
		const c10::optional<Tensor> &cos, // (tokens, 1, 128)
		const c10::optional<Tensor> &sin, // (tokens, 1, 128)
		const Tensor &input_lengths, // [10, 11, 8]
		const Tensor &save_slots, //  prefill : [[0, 16], [32, 0], [48, 0]]  decode : [[17], [35], [50]]
		const c10::optional<Tensor> &block_tables, // prefill : null, decode: [[0, 16], [32, 0], [48, 0]]
		const c10::optional<Tensor> &mask, // prefill: (max_length, max_length), decode: None
		int64_t	slots_size, // prefill: save_slots.size(1) decode: block_tables.size(1)
		int64_t mask_size, // mask_size
		int64_t block_size, // tokens num of one block
		double C, // softmax_scale
		int64_t attention_mode // prefille 2, decode 3
		)
	{
        return paged_attention(
            OUT, Q, K, V, Kcache, Vcache, cos, sin,
            input_lengths, c10::nullopt, save_slots, block_tables, mask,
            V.size(-1), slots_size, slots_size, mask_size, block_size, C, attention_mode);
	}

	Tensor hybrid_attention(
		Tensor &OUT, // (tokens, heads, heads_size)
		Tensor &mode_tensor, // index of itmes when its input_lengths > 1; [0, 1, 2](all prefill) or [2, 3](prefill + decode) or [](decode)
		Tensor &Q, // (tokens, heads, heads_size)
		Tensor &K, // (tokens, heads, heads_size)
		Tensor &V, // (tokens, heads, heads_size)
		Tensor &Kcache, // (blocks, block_size, heads, heads_size)
		Tensor &Vcache, // (blocks, block_size, heads, heads_size)
		const c10::optional<Tensor> &cos, // (tokens, 1, 128)
		const c10::optional<Tensor> &sin, // (tokens, 1, 128)
		const Tensor &input_lengths, // [1, 16, 32]
		const Tensor &cache_lengths, // [16, 7, 5]
		const Tensor &prompt_lengths, // [17, 23, 37]
		const Tensor &slots,
		const Tensor &block_tables,
		const c10::optional<Tensor> &mask, // prefill: (max_length, max_length), decode: None
		int64_t	slots_size, // block_tables.size(1)
		int64_t mask_size, // mask_size
		int64_t block_size, // tokens num of one block(16)
		double C // softmax_scale
		)
	{
		TORCH_CHECK (prompt_lengths.dtype() == torch::kInt32,
					"HybridAttention prompt lenghts must be int32 dtype");
		TORCH_CHECK ( prompt_lengths.device().type() == DeviceType::CPU,
					"HybridAttention prompt lenghts must on CPU device" );
		// 3 conditions according to mode_tensor:1.all prefill;2.prefill + decode;3.all decode
		if (mode_tensor.size(0) != 0) {
			int64_t first_large_index = mode_tensor[0].item<int64_t>();
            if (first_large_index == 0){
			//  all prefill
				return paged_attention(
							OUT, Q, K, V, Kcache, Vcache, cos, sin,
							input_lengths, cache_lengths, slots, block_tables, mask,
							V.size(-1), slots_size, slots_size, mask_size, block_size, C, 2);
			}else{
			// prefill + decode
				// decode
				paged_attention(
						OUT, Q, K, V, Kcache, Vcache, cos, sin,
						prompt_lengths, c10::nullopt, slots, block_tables, mask,
						V.size(-1), slots_size, slots_size, mask_size, block_size, C, 3);
				// prefill
				// calculate decode_length to split prefill and decode
				int64_t decode_length = input_lengths.index({Slice(0, first_large_index)}).sum().item<int64_t>();

				// slice tensor for prefill
				auto OUT_slice = OUT.index({Slice(decode_length, None), Slice()});
				auto Q_slice = Q.index({Slice(decode_length, None), Slice()});
				auto K_slice = K.index({Slice(decode_length, None), Slice()});
				auto V_slice = V.index({Slice(decode_length, None), Slice()});
				auto cos_slice = cos.has_value()
					? c10::make_optional(cos.value().index({Slice(decode_length, None), Slice()}))
					: c10::nullopt;
				auto sin_slice = sin.has_value()
					? c10::make_optional(sin.value().index({Slice(decode_length, None), Slice()}))
					: c10::nullopt;
				auto input_lengths_slice = input_lengths.index({Slice(first_large_index, None)});
				auto cache_lengths_slice = cache_lengths.index({Slice(first_large_index, None)});
				auto slots_slice = slots.index({Slice(decode_length, None)});
				auto block_tables_slice = block_tables.index({Slice(first_large_index, None), Slice()});

				return paged_attention(
					OUT_slice, Q_slice, K_slice, V_slice, Kcache, Vcache, cos_slice, sin_slice,
					input_lengths_slice, cache_lengths_slice, slots_slice, block_tables_slice, mask,
					V.size(-1), slots_size, slots_size, mask_size, block_size, C, 2
				);
				return OUT;
			}
		}else{
			// all decode
			return paged_attention(
				OUT, Q, K, V, Kcache, Vcache, cos, sin,
				prompt_lengths, c10::nullopt, slots, block_tables, mask,
				V.size(-1), slots_size, slots_size, mask_size, block_size, C, 3);
		}
	}

	Tensor llama_attention_forward(
		Tensor &OUT,
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		const c10::optional<Tensor> &cos,
		const c10::optional<Tensor> &sin,
		const c10::optional<Tensor> &mask,
		const c10::optional<Tensor> &softmax_lse,
		const Tensor &input_lengths,
		int64_t mask_size, // mask_size
		double C,//softmax_scale
		double dropout_rate,
		int64_t batch)
	{
		TIMING_START;
		//if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous()){
			// LOG(WARNING) << "llama_attention not contiguous, use QKV packed.";
		//}

		CHECK_TENSOR_IN_DEVICE(OUT);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(Q);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(K);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(V);

		if (cos.has_value())
			CHECK_TENSOR_IN_DEVICE(cos.value());
		if (sin.has_value())
			CHECK_TENSOR_IN_DEVICE(sin.value());
		if (mask.has_value())
			CHECK_TENSOR_IN_DEVICE(mask.value());

		auto stream =  c10_tpu::getCurrentTPUStream();
		tpudnnStatus_t status = tpudnnLlamaAttentionForwardAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor(stream, OUT),
			tpu::TPUGenerateTpudnnTensor(stream, Q),
			tpu::TPUGenerateTpudnnTensor(stream, K),
			tpu::TPUGenerateTpudnnTensor(stream, V),
			cos.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, cos.value()) : tpudnnUndefinedTensor(),
			sin.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, sin.value()) : tpudnnUndefinedTensor(),
			mask.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, mask.value()) : tpudnnUndefinedTensor(),
			softmax_lse.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, softmax_lse.value()) : tpudnnUndefinedTensor(),
			(int*)input_lengths.data_ptr(),
      	    (int)(input_lengths.nbytes()/4),
			mask_size,
			C,
			dropout_rate,
			batch);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
		TIMING_END;
		return OUT;

	}

	std::tuple<Tensor, Tensor, Tensor> llama_attention_backward(
		Tensor &Q,
		Tensor &K,
		Tensor &V,
		Tensor &O,
		Tensor &dO,
		Tensor &l,
		Tensor &dQ,
		Tensor &dK,
		Tensor &dV,
		const c10::optional<Tensor> &cos,
		const c10::optional<Tensor> &sin,
		const c10::optional<Tensor> &mask,
		const Tensor &input_lengths,
		int64_t mask_max, // mask_size
		double C // softmax_scale
		)
	{
		CHECK_TENSOR_IN_DEVICE(Q);
		CHECK_TENSOR_IN_DEVICE(K);
		CHECK_TENSOR_IN_DEVICE(V);
		CHECK_TENSOR_IN_DEVICE(O);
		CHECK_TENSOR_IN_DEVICE(dO);
		CHECK_TENSOR_IN_DEVICE(l);
		CHECK_TENSOR_IN_DEVICE(dQ);
		CHECK_TENSOR_IN_DEVICE(dK);
		CHECK_TENSOR_IN_DEVICE(dV);
		CHECK_TENSOR_IN_DEVICE(input_lengths);
		if (cos.has_value())
			CHECK_TENSOR_IN_DEVICE(cos.value());
		if (sin.has_value())
			CHECK_TENSOR_IN_DEVICE(sin.value());
		if (mask.has_value())
			CHECK_TENSOR_IN_DEVICE(mask.value());

		TIMING_START;
		auto stream = c10_tpu::getCurrentTPUStream();
		tpudnnStatus_t status = tpudnnLlamaAttentionBackwardAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor(stream,Q),
			tpu::TPUGenerateTpudnnTensor(stream,K),
			tpu::TPUGenerateTpudnnTensor(stream,V),
			tpu::TPUGenerateTpudnnTensor(stream,O),
			tpu::TPUGenerateTpudnnTensor(stream,dO),
			tpu::TPUGenerateTpudnnTensor(stream,l),
			tpu::TPUGenerateTpudnnTensor(stream,dQ),
			tpu::TPUGenerateTpudnnTensor(stream,dK),
			tpu::TPUGenerateTpudnnTensor(stream,dV),
			cos.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, cos.value()) : tpudnnUndefinedTensor(),
			sin.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, sin.value()) : tpudnnUndefinedTensor(),
			mask.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, mask.value()) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor(stream, input_lengths),
			mask_max,
			C);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
		TIMING_END; 
		return std::tuple<Tensor, Tensor, Tensor>(dQ, dK, dV);
	}

Tensor paged_attention_v3(
    Tensor &output,
    const Tensor &query,
    const Tensor &key,
    const Tensor &value,
    const Tensor &kcache,
    const Tensor &vcache,
    const c10::optional<Tensor> &pos_cos,
    const c10::optional<Tensor> &pos_sin,
    const Tensor &input_lengths,
    const Tensor &cache_lengths,
    const Tensor &save_slots,
    const Tensor &block_tables,
    const c10::optional<Tensor> &mask,
    double softmax_scale,

    // The following parameters can be calculated
    // if seqlen tensors are on CPU. But now they
    // may be on TPU so we pass them in but make 
    // them optional.
    int64_t max_seqlen = 0,
    int64_t stage = 0) // prefill 2, decode 3
{
  if (stage == 3 || (input_lengths == 1).all().item().toBool()) {
    // decode only
    return paged_attention(
        output,
        const_cast<Tensor &>(query), 
        const_cast<Tensor &>(key),
        const_cast<Tensor &>(value),
        const_cast<Tensor &>(kcache),
        const_cast<Tensor &>(vcache),
        pos_cos,
        pos_sin,
        input_lengths,
        cache_lengths,
        save_slots,
        block_tables,
        mask,
        query.size(-1),
        block_tables.size(1), // use block size as slots size
        0, // useless for decode
        0, // useless for decode
        kcache.size(1),
        softmax_scale,
        3);
  } else if (stage == 2 || (input_lengths > 1).all().item().toBool()) {
    // prefill only
    if (max_seqlen <= 0)
    {
      const auto seq_lengths = input_lengths + cache_lengths;
      max_seqlen = seq_lengths.max().item().toInt();
    }
    return paged_attention(
        output,
        const_cast<Tensor &>(query),
        const_cast<Tensor &>(key),
        const_cast<Tensor &>(value),
        const_cast<Tensor &>(kcache),
        const_cast<Tensor &>(vcache),
        pos_cos,
        pos_sin,
        input_lengths,
        cache_lengths,
        save_slots,
        block_tables,
        mask,
        query.size(-1),
        0, // useless for prefill
        block_tables.size(1),
        mask.has_value() ? mask.value().size(0) : max_seqlen,
        kcache.size(1),
        softmax_scale,
        2);
  } else {
    // prefill + decode
    auto prefill_indices = torch::where(input_lengths > 1)[0];
    for (int64_t i = 1; i < prefill_indices.numel(); i++) {
        TORCH_CHECK(prefill_indices[i].item().toInt() == prefill_indices[i - 1].item().toInt() + 1,
                    "prefill_indices must be continuous");
    }
    int64_t start_prefill_id = prefill_indices[0].item().toInt();
    auto seperate_tensors = [&](int64_t start_id, c10::optional<int64_t> end_id,
                                bool is_prefill) {
      auto output_slice = output.slice(0, start_id, end_id);
      auto query_slice = query.slice(0, start_id, end_id);
      auto key_slice = key.slice(0, start_id, end_id);
      auto value_slice = value.slice(0, start_id, end_id);
      auto cos_slice =
          pos_cos.has_value()
              ? c10::make_optional(pos_cos.value().slice(0, start_id, end_id))
              : c10::nullopt;
      auto sin_slice =
          pos_sin.has_value()
              ? c10::make_optional(pos_sin.value().slice(0, start_id, end_id))
              : c10::nullopt;

      auto input_lengths_slice = input_lengths.slice(0, start_id, end_id);
      auto cache_lengths_slice = cache_lengths.slice(0, start_id, end_id);
      auto save_slots_slice = save_slots.slice(0, start_id, end_id);
      auto block_tables_slice = block_tables.slice(0, start_id, end_id);

      if (max_seqlen <= 0)
      {
        auto seq_lengths_slice = input_lengths_slice + cache_lengths_slice;
        max_seqlen = seq_lengths_slice.max().item().toInt();
      }
      return paged_attention(
          output_slice,
          query_slice,
          key_slice,
          value_slice,
          const_cast<Tensor &>(kcache),
          const_cast<Tensor &>(vcache),
          cos_slice,
          sin_slice,
          input_lengths_slice,
          cache_lengths_slice,
          save_slots_slice,
          block_tables_slice,
          mask,
          query_slice.size(-1),
          block_tables_slice.size(1),
          block_tables_slice.size(1),
          mask.has_value() ? mask.value().size(0) : max_seqlen,
          kcache.size(1),
          softmax_scale,
          is_prefill ? 2 : 3);
    };

    seperate_tensors(0, start_prefill_id, false);
    return seperate_tensors(start_prefill_id, None, true);
  }
  return output;
}
Tensor paged_attention_v2(
    Tensor &output,
    const Tensor &query,
    const Tensor &key,
    const Tensor &value,
    const Tensor &kcache,
    const Tensor &vcache,
    const c10::optional<Tensor> &pos_cos,
    const c10::optional<Tensor> &pos_sin,
    const Tensor &input_lengths,
    const Tensor &cache_lengths,
    const Tensor &save_slots,
    const Tensor &block_tables,
    const c10::optional<Tensor> &mask,
    double softmax_scale)
{
    return paged_attention_v3(
        output, query, key, value, kcache, vcache,
        pos_cos, pos_sin, input_lengths, cache_lengths,
        save_slots, block_tables, mask, softmax_scale);
}

}
