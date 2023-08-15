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

    Tensor &asinh_out_tpu(const Tensor &self, Tensor &out)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(out);
#if 0
 
  auto self_cpu = asinh ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
        if (self.dim() == 0)
        {
            auto self_cpu = asinh(self.cpu());
            tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(), self.nbytes());
        }
        else if (IS_TPU_TENSOR(self))
        {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnASinH(
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensor(self),
                tpu::TPUGenerateSgdnnTensor(out));
            TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime(tpu::ASINH_FORWARD, timer.ElapsedUS());
#endif
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        return out;
    }

    Tensor &acosh_out_tpu(const Tensor &self, Tensor &out)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(out);
#if 0
 
  auto self_cpu = acosh ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
        if (self.dim() == 0)
        {
            auto self_cpu = acosh(self.cpu());
            tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(), self.nbytes());
        }
        else if (IS_TPU_TENSOR(self))
        {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnACosH(
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensor(self),
                tpu::TPUGenerateSgdnnTensor(out));
            TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime(tpu::ACOSH_FORWARD, timer.ElapsedUS());
#endif
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        return out;
    }

    Tensor &atanh_out_tpu(const Tensor &self, Tensor &out)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(out);
#if 0
 
  auto self_cpu = atanh ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
        if (self.dim() == 0)
        {
            auto self_cpu = atanh(self.cpu());
            tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(), self.nbytes());
        }
        else if (IS_TPU_TENSOR(self))
        {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnATanH(
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensor(self),
                tpu::TPUGenerateSgdnnTensor(out));
            TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime(tpu::ATANH_FORWARD, timer.ElapsedUS());
#endif
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        return out;
    }

    Tensor atanh_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options());
        return atanh_out_tpu(self, out);
    }

    Tensor acosh_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options());
        return acosh_out_tpu(self, out);
    }

    Tensor asinh_tpu(const Tensor &self)
    {
        auto out = empty(self.sizes(), self.options());
        return asinh_out_tpu(self, out);
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("atanh.out", atanh_out_tpu);
        m.impl("atanh", atanh_tpu);
        m.impl("acosh.out", acosh_out_tpu);
        m.impl("acosh", acosh_tpu);
        m.impl("asinh.out", asinh_out_tpu);
        m.impl("asinh", asinh_tpu);
    }

} // namespace at
