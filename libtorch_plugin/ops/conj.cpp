#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <iostream>
#include "common/config.h"

namespace at
{

    Tensor & conj_out_tpu (const Tensor &self, Tensor &out)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(out);
#if 0
 
  auto self_cpu = neg ( self.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
#else
        if (self.dim() == 0)
        {
            auto self_cpu = conj(self.cpu());   //直接在cpu上执行real，这里有问题
            tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(), out.nbytes());
        }
        else if (IS_TPU_TENSOR(self))
        {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnConj(
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensorforComplex64(self),
                tpu::TPUGenerateSgdnnTensorforComplex64(out));
            TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime(tpu::ADD, timer.ElapsedUS());
#endif
        }
        else
        {
            
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        return out;
    }

    Tensor conj_tpu(const Tensor &self)
    {
        //std::cout<<"enter conj_tpu\n";
        
        auto out = empty(self.sizes(), self.options());
        return conj_out_tpu(self, out);
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("_conj", conj_tpu);
    }

} // namespace at

    