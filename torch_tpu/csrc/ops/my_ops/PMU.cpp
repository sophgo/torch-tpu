#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
    void enable_pmu() {
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
        tpu_status_t status = sgdnnPMU(
            tpu::TPUGetDeviceResource(),
            1
        );
        TORCH_CHECK(status == SG_SUCCESS);
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::ENABLE_PMU, timer.ElapsedUS());
#endif
        return;
    }

    void disable_pmu() {
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
        tpu_status_t status = sgdnnPMU(
            tpu::TPUGetDeviceResource(),
            0
        );
        TORCH_CHECK(status == SG_SUCCESS);
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::DISABLE_PMU, timer.ElapsedUS());
#endif
        return;
    }
}