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
		if (fetch_slots.has_value())
			CHECK_TENSOR_IN_DEVICE(fetch_slots.value());
		if (mask.has_value())
			CHECK_TENSOR_IN_DEVICE(mask.value());
		if (attention_mode == 3 || attention_mode == 2){
			TORCH_CHECK (tpu::TPUConvertDtype<SgdnnDataType_t>(input_lengths.dtype()) == SGDNN_DTYPE_INT32,
						"LLammaAttention input lenghts must be int32 dtype");
			TORCH_CHECK ( input_lengths.device().type() == DeviceType::CPU, 
						"LLammaAttention input lenghts must on CPU device" );
		}

		TIMING_START;
		Tensor Qbuffer, Kbuffer, Vbuffer;
		if (attention_mode == 2)
		{
			TensorOptions options = TensorOptions ( Q.device() ).dtype ( Q.dtype() );
			Qbuffer = empty({Q.sizes()}, options);
			Kbuffer = empty({K.sizes()}, options);
			Vbuffer = empty({V.sizes()}, options);
		}
  		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnLlamaAttentionAsync(
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
			fetch_slots.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, fetch_slots.value()) : tpudnnUndefinedTensor(),
			mask.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, mask.value()) : tpudnnUndefinedTensor(),
			attention_mode == 2 ? tpu::TPUGenerateTpudnnTensor(stream, Qbuffer) : tpudnnUndefinedTensor(),
			attention_mode == 2 ? tpu::TPUGenerateTpudnnTensor(stream, Kbuffer) : tpudnnUndefinedTensor(),
			attention_mode == 2 ? tpu::TPUGenerateTpudnnTensor(stream, Vbuffer) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor(stream, input_lengths),
			(int*)input_lengths.data_ptr(),
      	    (int)(input_lengths.nbytes()/4),
			slots_size,
			mask_size,
			block_size,
			C,
			attention_mode);
		TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
		TIMING_END( tpu::LLAMA_ATTENTION );
		return OUT;
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
		int64_t mask_size, // mask_size
		double C,
		double dropout_rate,
		int64_t batch)
	{
		if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous()){
			// LOG(WARNING) << "llama_attention not contiguous, use QKV packed.";
		}

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

#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif

		tpu_status_t status = sgdnnLlamaAttentionForward(
			tpu::TPUGetDeviceResource(),
			tpu::TPUGenerateSgdnnTensor(OUT),
			tpu::TPUGenerateSgdnnTensor(Q),
			tpu::TPUGenerateSgdnnTensor(K),
			tpu::TPUGenerateSgdnnTensor(V),
			cos.has_value() ? tpu::TPUGenerateSgdnnTensor(cos.value()) : sgdnnUndefinedTensor(),
			sin.has_value() ? tpu::TPUGenerateSgdnnTensor(sin.value()) : sgdnnUndefinedTensor(),
			mask.has_value() ? tpu::TPUGenerateSgdnnTensor(mask.value()) : sgdnnUndefinedTensor(),
			softmax_lse.has_value() ? tpu::TPUGenerateSgdnnTensor(softmax_lse.value()) : sgdnnUndefinedTensor(),
			mask_size,
			C,
			dropout_rate,
			batch);
		TORCH_CHECK(status == SG_SUCCESS);

#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::LLAMA_ATTENTION, timer.ElapsedUS());
#endif
		return OUT;

	}

}
