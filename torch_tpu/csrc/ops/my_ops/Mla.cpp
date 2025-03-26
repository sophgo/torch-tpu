#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor mla(
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
		double C,
		int64_t attention_mode // prefille 0, decode 1
		)
	{

		if (attention_mode == 0 || attention_mode == 1){
			TORCH_CHECK (tpu::TPUConvertDtype<SgdnnDataType_t>(input_lengths.dtype()) == SGDNN_DTYPE_INT32,
						"MLA input lenghts must be int32 dtype");
			TORCH_CHECK ( input_lengths.device().type() == DeviceType::CPU, 
						"MLA input lenghts must on CPU device" );
		}
		
		TIMING_START;
  		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnMLAAsync(
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
			C,
			attention_mode);
		TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
		TIMING_END( tpu::MLA );
		return OUT;
	}
}