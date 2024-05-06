#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

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
            CPU_IMPL_WARNING();
            TIMING_START;
            auto self_cpu = conj(self.cpu());   //直接在cpu上执行real，这里有问题
            tpu::TPUCopyHostToDevice(out.data_ptr(), self_cpu.contiguous().data_ptr(), out.nbytes());
            TIMING_END(tpu::CPU_LAYER);
        }
        else if (IS_TPU_TENSOR(self))
        {
            TIMING_START;
            #if defined BACKEND_1684X
            auto status = sgdnnConj(
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensorforComplex64(self),
                tpu::TPUGenerateSgdnnTensorforComplex64(out));
            TORCH_CHECK(status == BM_SUCCESS);
            #elif defined BACKEND_SG2260
            auto status = sgdnnConj(
                c10_tpu::getCurrentTPUStream(),
                tpu::TPUGenerateSgdnnTensorforComplex64(self),
                tpu::TPUGenerateSgdnnTensorforComplex64(out));
            TORCH_CHECK(status == tpuRtSuccess);
            #endif
            TIMING_END(tpu::CONJ);
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        SHOW_TENSOR_OP(self, out);
        return out;
    }

    Tensor conj_tpu(const Tensor &self)
    {        
        auto out = empty(self.sizes(), self.options());
        return conj_out_tpu(self, out);
    }

    TORCH_LIBRARY_IMPL(aten, TPU, m)
    {
        m.impl("_conj", conj_tpu);
    }

} // namespace at

    