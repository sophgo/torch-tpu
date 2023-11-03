#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"
namespace at{

Tensor & _amp_update_scale_tpu(Tensor & self, Tensor & growth_tracker, const Tensor & found_inf,
    double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval)
{
    CHECK_TENSOR_IN_DEVICE ( self );
    CHECK_TENSOR_IN_DEVICE ( growth_tracker );
    CHECK_TENSOR_IN_DEVICE ( found_inf );

    TORCH_CHECK(growth_tracker.numel() == 1, "growth_tracker must be a 1-element tensor.");
    TORCH_CHECK(self.numel() == 1, "current_scale must be a 1-element tensor.");
    TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
    TORCH_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int, "growth_tracker must be an int tensor.");
    TORCH_CHECK(self.scalar_type() == at::ScalarType::Float, "current_scale must be a float tensor.");
    TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

    auto self_cpu = self.cpu(); // scale
    auto growth_tracker_cpu = growth_tracker.cpu();
    if (found_inf.cpu().data_ptr<float>()){ // found inf
        *(self_cpu.data_ptr<float>()) = *(self_cpu.data_ptr<float>()) * scale_backoff_factor;
        *(growth_tracker_cpu.data_ptr<int>()) = 0;
        tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.data_ptr(), self.nbytes() );
        tpu::TPUCopyHostToDevice ( growth_tracker.data_ptr(), growth_tracker_cpu.data_ptr(), growth_tracker.nbytes() );
    } else {
        // Entering this branch means we just carried out a successful step,
        // so growth_tracker is incremented before comparing to growth_interval.
        auto successful = *(growth_tracker_cpu.data_ptr<int>()) + 1;
        if (successful == growth_interval){
            *(self_cpu.data_ptr<float>()) = *(self_cpu.data_ptr<float>()) * scale_growth_factor;
            *(growth_tracker_cpu.data_ptr<int>()) = 0;
            tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.data_ptr(), self.nbytes() );
            tpu::TPUCopyHostToDevice ( growth_tracker.data_ptr(), growth_tracker_cpu.data_ptr(), growth_tracker.nbytes() );
        }else{
            *(growth_tracker_cpu.data_ptr<int>()) = successful;
            tpu::TPUCopyHostToDevice ( growth_tracker.data_ptr(), growth_tracker_cpu.data_ptr(), growth_tracker.nbytes() );
        }
    }
    return self;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "_amp_update_scale_",  _amp_update_scale_tpu);
}

}; //namespace at