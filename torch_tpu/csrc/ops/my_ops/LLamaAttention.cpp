#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
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
		const c10::optional<Tensor> &fetch_slots, // prefill : null, decode: [[0, 16], [32, 0], [48, 0]]
		const c10::optional<Tensor> &mask, // prefill: (max_length, max_length), decode: None
		int64_t	slots_size, // prefill: save_slots.size(1) decode: fetch_slots.size(1)
		int64_t mask_size, // mask_size
		int64_t block_size, // tokens num of one block
		double C, // softmax_scale
		int64_t attention_mode // prefille 2, decode 3
		)
	{
		if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous()){
#ifndef USE_QKV_PACKED
			LOG(WARNING) << "llama_attention not contiguous, change Q, K, V to contiguous.";
			Q = Q.contiguous();
			K = K.contiguous();
			V = V.contiguous();
#else

			LOG(WARNING) << "llama_attention not contiguous, use QKV packed.";
#endif

		}
		if (!Kcache.is_contiguous() || !Vcache.is_contiguous()){
			LOG(WARNING) << "llama_attention not contiguous, change Kcache, Vcache to contiguous.";
			Kcache = Kcache.contiguous();
			Vcache = Vcache.contiguous();
		}

		CHECK_TENSOR_IN_DEVICE(OUT);
#ifndef USE_QKV_PACKED
		CHECK_TENSOR_IN_DEVICE(Q);
		CHECK_TENSOR_IN_DEVICE(K);
		CHECK_TENSOR_IN_DEVICE(V);
#else
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(Q);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(K);
		CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(V);
#endif
		CHECK_TENSOR_IN_DEVICE(Kcache);
		CHECK_TENSOR_IN_DEVICE(Vcache);
		CHECK_TENSOR_IN_DEVICE(input_lengths);
		CHECK_TENSOR_IN_DEVICE(save_slots);
		if (cos.has_value())
			CHECK_TENSOR_IN_DEVICE(cos.value());
		if (sin.has_value())
			CHECK_TENSOR_IN_DEVICE(sin.value());
		if (fetch_slots.has_value())
			CHECK_TENSOR_IN_DEVICE(fetch_slots.value());
		if (mask.has_value())
			CHECK_TENSOR_IN_DEVICE(mask.value());
		int Ntotal = 0;
		for (int i=0; i<input_lengths.size(0); ++i){
			Ntotal = Ntotal + input_lengths[i].item().toInt();
		}

#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
		tpu_status_t status = sgdnnLlamaAttention(
			tpu::TPUGetDeviceResource(),
			tpu::TPUGenerateSgdnnTensor(OUT),
			tpu::TPUGenerateSgdnnTensor(Q),
			tpu::TPUGenerateSgdnnTensor(K),
			tpu::TPUGenerateSgdnnTensor(V),
			tpu::TPUGenerateSgdnnTensor(Kcache),
			tpu::TPUGenerateSgdnnTensor(Vcache),
			cos.has_value() ? tpu::TPUGenerateSgdnnTensor(cos.value()) : sgdnnUndefinedTensor(),
			sin.has_value() ? tpu::TPUGenerateSgdnnTensor(sin.value()) : sgdnnUndefinedTensor(),
			tpu::TPUGenerateSgdnnTensor(input_lengths),
			tpu::TPUGenerateSgdnnTensor(save_slots),
			fetch_slots.has_value() ? tpu::TPUGenerateSgdnnTensor(fetch_slots.value()) : sgdnnUndefinedTensor(),
			mask.has_value() ? tpu::TPUGenerateSgdnnTensor(mask.value()) : sgdnnUndefinedTensor(),
			slots_size,
			mask_size,
			block_size,
			C,
			attention_mode,
			Ntotal);
		TORCH_CHECK(status == SG_SUCCESS);

#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::LLAMA_ATTENTION, timer.ElapsedUS());
#endif
		return OUT;
	}

}

