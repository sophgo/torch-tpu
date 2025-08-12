#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
	void dynlib_execute(
		const std::string &so_url,
		const std::string &func_name,
		const std::vector<Tensor> &tensors,
		const std::vector<int64_t> &tensors_index,
		const std::vector<double> &fp_scalars,
		const std::vector<int64_t> &fp_scalars_index,
		const std::vector<int64_t> &fixed_scalars,
        const std::vector<int64_t> &fixed_scalars_index
		)
	{
		TIMING_START;
		TORCH_CHECK(tensors.size() == tensors_index.size());
		TORCH_CHECK(fp_scalars.size() == fp_scalars_index.size());
		TORCH_CHECK(fixed_scalars.size() == fixed_scalars_index.size());
		for (size_t i = 0; i < tensors.size(); i++) {
			CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(tensors[i]);
		}
#if defined BACKEND_SG2260
		auto stream = c10_tpu::getCurrentTPUStream();
		std::vector<tpudnnTensor_t> dnnTensors;
		for (size_t i = 0; i < tensors.size(); i++) {
			dnnTensors.emplace_back(tpu::TPUGenerateTpudnnTensor(stream, tensors[i]));
		}
		auto status = tpudnnDynLibExecuteAsync(
											stream,
											so_url.c_str(),
											func_name.c_str(),
											dnnTensors,
											tensors_index,
											fp_scalars,
											fp_scalars_index,
											fixed_scalars,
											fixed_scalars_index);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END;
		return;
	}
}
