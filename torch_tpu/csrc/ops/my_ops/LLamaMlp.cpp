#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	Tensor llama_mlp_forward(
		Tensor &input,
		Tensor &weight0,
		Tensor &weight1,
		Tensor &weight2,
		const c10::optional<Tensor> &bias0,
		const c10::optional<Tensor> &bias1,
		const c10::optional<Tensor> &bias2,
		const c10::optional<Tensor> &fc1,
		const c10::optional<Tensor> &m0,
		Tensor &output,
		bool save_mid_res)
	{
		TIMING_START;
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(weight0);
		CHECK_TENSOR_IN_DEVICE(weight1);
		CHECK_TENSOR_IN_DEVICE(weight2);
		if (bias0.has_value())
			CHECK_TENSOR_IN_DEVICE(bias0.value());
		if (bias1.has_value())
			CHECK_TENSOR_IN_DEVICE(bias1.value());
		if (bias2.has_value())
			CHECK_TENSOR_IN_DEVICE(bias2.value());
		if (fc1.has_value())
			CHECK_TENSOR_IN_DEVICE(fc1.value());
		if (m0.has_value())
			CHECK_TENSOR_IN_DEVICE(m0.value());
		CHECK_TENSOR_IN_DEVICE(output);

#if defined BACKEND_SG2260
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnLLamaMlpAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, input),
			tpu::TPUGenerateTpudnnTensor( stream, weight0),
			tpu::TPUGenerateTpudnnTensor( stream, weight1),
			tpu::TPUGenerateTpudnnTensor( stream, weight2),
			bias0.has_value()? tpu::TPUGenerateTpudnnTensor( stream, bias0.value()) : tpudnnUndefinedTensor(),
			bias1.has_value()? tpu::TPUGenerateTpudnnTensor( stream, bias1.value()) : tpudnnUndefinedTensor(),
			bias2.has_value()? tpu::TPUGenerateTpudnnTensor( stream, bias2.value()) : tpudnnUndefinedTensor(),
			fc1.has_value() ? tpu::TPUGenerateTpudnnTensor( stream, fc1.value()) : tpudnnUndefinedTensor(),
			m0.has_value() ? tpu::TPUGenerateTpudnnTensor( stream, m0.value()) : tpudnnUndefinedTensor(),
			tpu::TPUGenerateTpudnnTensor( stream, output),
			save_mid_res);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END;
		return output;
  }

std::tuple<Tensor, Tensor, Tensor, Tensor> mlp_backward(
    Tensor &input,
    Tensor &weight0,
    Tensor &weight1,
    Tensor &weight2,
    Tensor &w0x,
    Tensor &w1x,
    Tensor &output,
    Tensor &grad_tmp,
    Tensor &grad_input,
    Tensor &grad_weight0,
    Tensor &grad_weight1,
    Tensor &grad_weight2)
    {
    TIMING_START;
    CHECK_TENSOR_IN_DEVICE(input);
    CHECK_TENSOR_IN_DEVICE(weight0);
    CHECK_TENSOR_IN_DEVICE(weight1);
    CHECK_TENSOR_IN_DEVICE(weight2);
    CHECK_TENSOR_IN_DEVICE(w0x);
    CHECK_TENSOR_IN_DEVICE(w1x);
    CHECK_TENSOR_IN_DEVICE(output);
    CHECK_TENSOR_IN_DEVICE(grad_tmp);
    CHECK_TENSOR_IN_DEVICE(grad_input);
    CHECK_TENSOR_IN_DEVICE(grad_weight0);
    CHECK_TENSOR_IN_DEVICE(grad_weight1);
    CHECK_TENSOR_IN_DEVICE(grad_weight2);

    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnMlpbackward(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, input),
      tpu::TPUGenerateTpudnnTensor(stream, weight0),
      tpu::TPUGenerateTpudnnTensor(stream, weight1),
      tpu::TPUGenerateTpudnnTensor(stream, weight2),
      tpu::TPUGenerateTpudnnTensor(stream, w0x),
      tpu::TPUGenerateTpudnnTensor(stream, w1x),
      tpu::TPUGenerateTpudnnTensor(stream, output),
      tpu::TPUGenerateTpudnnTensor(stream, grad_tmp),
      tpu::TPUGenerateTpudnnTensor(stream, grad_input),
      tpu::TPUGenerateTpudnnTensor(stream, grad_weight0),
      tpu::TPUGenerateTpudnnTensor(stream, grad_weight1),
      tpu::TPUGenerateTpudnnTensor(stream, grad_weight2));

    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END;
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(grad_input, grad_weight0, grad_weight1, grad_weight2);
}

Tensor llama_mlp_gptq_forward(
		Tensor &input,
		Tensor &weight0,
		Tensor &zp0,
		Tensor &scale0,
		Tensor &weight1,
		Tensor &zp1,
		Tensor &scale1,
		Tensor &weight2,
		Tensor &zp2,
		Tensor &scale2,
		int64_t group_size,
		int64_t weight_bits,
		Tensor &output)
	{
		TIMING_START;

		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(weight0);
		CHECK_TENSOR_IN_DEVICE(zp0);
		CHECK_TENSOR_IN_DEVICE(scale0);
		CHECK_TENSOR_IN_DEVICE(weight1);
		CHECK_TENSOR_IN_DEVICE(zp1);
		CHECK_TENSOR_IN_DEVICE(scale1);
		CHECK_TENSOR_IN_DEVICE(weight2);
		CHECK_TENSOR_IN_DEVICE(zp2);
		CHECK_TENSOR_IN_DEVICE(scale2);
		CHECK_TENSOR_IN_DEVICE(output);

#if defined BACKEND_SG2260
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnLLamaA16MlpAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, input),
			tpu::TPUGenerateTpudnnTensor( stream, weight0),
			tpu::TPUGenerateTpudnnTensor( stream, zp0),
			tpu::TPUGenerateTpudnnTensor( stream, scale0),
			tpu::TPUGenerateTpudnnTensor( stream, weight1),
			tpu::TPUGenerateTpudnnTensor( stream, zp1),
			tpu::TPUGenerateTpudnnTensor( stream, scale1),
			tpu::TPUGenerateTpudnnTensor( stream, weight2),
			tpu::TPUGenerateTpudnnTensor( stream, zp2),
			tpu::TPUGenerateTpudnnTensor( stream, scale2),
			group_size,
			weight_bits,
			tpu::TPUGenerateTpudnnTensor( stream,output));
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END;
		return output;
	}

}
