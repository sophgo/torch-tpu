#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"
namespace at 
{
void _amp_foreach_non_finite_check_and_unscale_tpu(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale) {
    CHECK_TENSOR_IN_DEVICE ( found_inf );
    CHECK_TENSOR_IN_DEVICE ( inv_scale );
    TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
    TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
    TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor.");

    Tensor inv_scale_cpu = inv_scale.cpu();
    std::vector<SgdnnTensor_t> inputs;
    for (const auto & s : self){ 
        CHECK_TENSOR_IN_DEVICE ( s );
        inputs.push_back( tpu::TPUGenerateSgdnnTensor(s) ); }
    TIMING_START;
    bm_status_t status = sgdnnInfCheckAndUnscale(
        tpu::TPUGetDeviceHandle(),
        inputs,
        tpu::TPUGenerateSgdnnTensor(found_inf),
        *inv_scale_cpu.data_ptr<float>());
    TORCH_CHECK(status == BM_SUCCESS, "_amp_foreach_non_finite_check_and_unscale_ failed.");
    TIMING_END(tpu::InfCheckAndUnscale);
    SHOW_TENSOR_OP( found_inf, inv_scale);
}



TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "_amp_foreach_non_finite_check_and_unscale_",  _amp_foreach_non_finite_check_and_unscale_tpu);
}
} //namespace at