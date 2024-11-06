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
    void enable_profile(
        c10::optional<int64_t> max_record_num,
        c10::optional<bool> enable_mcu) {
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
#if defined BACKEND_SG2260
    int32_t num = 40960;
    bool en_mcu = true;
    if (max_record_num.has_value())
        num = max_record_num.value();
    if (enable_mcu.has_value())
        en_mcu = enable_mcu.value();
    auto stream = c10_tpu::getCurrentTPUStream();
    tpudnnEnableProfile(stream, num, en_mcu);
#else
    TORCH_CHECK(false);
#endif
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::ENABLE_PMU, timer.ElapsedUS());
#endif
        return;
    }

    void disable_profile() {
#ifdef TPU_OP_TIMING
		auto timer = tpu::Timer().Start();
#endif
#if defined BACKEND_SG2260
        auto stream = c10_tpu::getCurrentTPUStream();   
        tpudnnDisableProfile(stream);
#else
    TORCH_CHECK(false);
#endif
#ifdef TPU_OP_TIMING
		tpu::OpTimer::Instance().AddTime(tpu::DISABLE_PMU, timer.ElapsedUS());
#endif
        return;
    }
}