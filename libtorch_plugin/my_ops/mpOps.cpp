#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"
#include "ops.hpp"

namespace at
{
	TORCH_LIBRARY(my_ops, m)
	{
		m.def("mlp_forward", mlp_forward);
		m.def("mlp_backward", mlp_backward);
		m.def("attn_forward", attn_forward);
		m.def("attn_backward", attn_backward);
		m.def("ln_mm_forward", ln_mm_forward);
	}
}