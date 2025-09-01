#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"
namespace at
{
void _amp_foreach_non_finite_check_and_unscale_tpu(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale) {
    TIMING_START;
    CHECK_TENSOR_IN_DEVICE ( found_inf );
    CHECK_TENSOR_IN_DEVICE ( inv_scale );
    TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
    TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
    TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor.");
    auto stream = c10_tpu::getCurrentTPUStream();
    auto buffer = empty({8}, inv_scale.options());
    std::vector<tpudnnTensor_t> inputs;
    for (const auto & s : self){
        CHECK_TENSOR_IN_DEVICE ( s );
        inputs.push_back( tpu::TPUGenerateTpudnnTensor (stream,s) ); }
    auto status = tpudnnInfCheckAndUnscaleAsync(
    stream,
    inputs,
    tpu::TPUGenerateTpudnnTensor (stream,found_inf),
    tpu::TPUGenerateTpudnnTensor (stream,buffer),
    tpu::TPUGenerateTpudnnTensor (stream,inv_scale));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS, "_amp_foreach_non_finite_check_and_unscale_ failed.");\
    TIMING_END;
    SHOW_TENSOR_OP( found_inf, inv_scale);
}



TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "_amp_foreach_non_finite_check_and_unscale_",  _amp_foreach_non_finite_check_and_unscale_tpu);
}
} //namespace at