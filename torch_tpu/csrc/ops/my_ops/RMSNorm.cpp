#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

namespace at
{
	Tensor rmsnorm_forward(
		Tensor &input,
		const c10::optional<Tensor> &scale,
		const c10::optional<Tensor> &bias,
		Tensor &output,
		int64_t axis,
		double_t eps)
	{
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(output);
		if(scale.has_value()){
			CHECK_TENSOR_IN_DEVICE(scale.value());
		}
		if(bias.has_value()){
			CHECK_TENSOR_IN_DEVICE(bias.value());
		}
		
		TIMING_START;
		auto stream = c10_tpu::getCurrentTPUStream();
		tpudnnStatus_t status = tpudnnRmsNormForwardAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor(stream, input),
			scale.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, scale.value()) : tpudnnUndefinedTensor(),
			bias.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, bias.value()) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor(stream, output),
			axis,
			eps);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
		TIMING_END(tpu::RMSNORM_FORWARD);
		return output;
	}

	Tensor rmsnorm_backward(
		Tensor &grad_output,
		Tensor &input,
		const c10::optional<Tensor> &scale,
		const c10::optional<Tensor> &bias,
		Tensor &rms,
		const c10::optional<Tensor> &grad_input,
		const c10::optional<Tensor> &grad_scale,
		const c10::optional<Tensor> &grad_bias,
		int64_t axis,
		double_t eps)
	{
		CHECK_TENSOR_IN_DEVICE(grad_output);
		CHECK_TENSOR_IN_DEVICE(input);
		if (scale.has_value())
		{
			CHECK_TENSOR_IN_DEVICE(scale.value());
		}
		if (bias.has_value())
		{
			CHECK_TENSOR_IN_DEVICE(bias.value());
		}
		CHECK_TENSOR_IN_DEVICE(rms);
		if (grad_input.has_value())
		{
			CHECK_TENSOR_IN_DEVICE(grad_input.value());
		}
		if (grad_scale.has_value())
		{
			CHECK_TENSOR_IN_DEVICE(grad_scale.value());
		}
		if (grad_bias.has_value())
		{
			CHECK_TENSOR_IN_DEVICE(grad_bias.value());
		}	
		TIMING_START;
  		auto stream = c10_tpu::getCurrentTPUStream();
		tpudnnStatus_t status = tpudnnRmsNormBackwardAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor(stream, grad_output),
			tpu::TPUGenerateTpudnnTensor(stream, input),
			scale.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, scale.value()) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor(stream, rms),
			grad_input.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, grad_input.value()) : tpudnnUndefinedTensor(),
			grad_scale.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, grad_scale.value()) : tpudnnUndefinedTensor(),
			grad_bias.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, grad_bias.value()) : tpudnnUndefinedTensor(),
			axis,
			eps);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
		TIMING_END(tpu::RMSNORM_BACKWARD);
		return rms;
	}

}
