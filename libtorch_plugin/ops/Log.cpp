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

    Tensor &logx_out_tpu(const Tensor &self, Tensor &out, int log_type)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(out);
#if 0
 
  auto self_cpu = log ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
        if (self.dim() == 0)
        {
            auto self_cpu = log(self.cpu());
            tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(), self.nbytes());
        }
        else if (IS_TPU_TENSOR(self))
        {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnLog(
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensor(self),
                tpu::TPUGenerateSgdnnTensor(out),
                log_type);
            TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime(tpu::LOG_FORWARD, timer.ElapsedUS());
#endif
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        return out;
    }

    Tensor logx_tpu(const Tensor &self, int log_type)
    {
        auto out = empty(self.sizes(), self.options());
        return logx_out_tpu(self, out, log_type);
    }

    Tensor &log_out_tpu(const Tensor &self, Tensor &out)
    {
        return logx_out_tpu(self, out, 0);
    }

    Tensor &log1p_out_tpu(const Tensor &self, Tensor &out)
    {
        return logx_out_tpu(self, out, 1);
    }

    Tensor &log2_out_tpu(const Tensor &self, Tensor &out)
    {
        return logx_out_tpu(self, out, 2);
    }

    Tensor &log10_out_tpu(const Tensor &self, Tensor &out)
    {
        return logx_out_tpu(self, out, 10);
    }

    Tensor log_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options());
        return log_out_tpu(self, out);
    }
    Tensor log1p_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options());
        return log1p_out_tpu(self, out);
    }
    Tensor log2_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options());
        return log2_out_tpu(self, out);
    }
    Tensor log10_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options());
        return log10_out_tpu(self, out);
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("log.out", log_out_tpu);
        m.impl("log", log_tpu);
        m.impl("log1p.out", log1p_out_tpu);
        m.impl("log1p", log1p_tpu);
        m.impl("log2.out", log2_out_tpu);
        m.impl("log2", log2_tpu);
        m.impl("log10.out", log10_out_tpu);
        m.impl("log10", log10_tpu);
    }

} // namespace at
