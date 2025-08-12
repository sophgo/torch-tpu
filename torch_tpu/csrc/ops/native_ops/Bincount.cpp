#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <vector>
#include "TPUTorchUtils.h"


#include "common/config.h"
namespace at {

Tensor & bincount_out_tpu(const Tensor & self, const c10::optional<Tensor> & weights, int64_t minlength, Tensor & out)
{
    TIMING_START;
    CPU_IMPL_WARNING();
    Tensor weights_cpu;
    if ( weights.has_value() ) { weights_cpu = weights.value().cpu(); }
    auto out_cpu = bincount(self.cpu(), c10::optional<Tensor>(weights_cpu), minlength);
    out = out_cpu.to(out.device()).to(out.dtype());
    TIMING_END;
    return out;
}

Tensor bincount_tpu(const Tensor & self, const c10::optional<Tensor> & weights, int64_t minlength)
{
    TensorOptions options = TensorOptions(self.device()).dtype(ScalarType::Int);
    if (weights.has_value()) { options = options.dtype(ScalarType::Float); } 
    LOG(WARNING) << "bincount will not check max-value of input and use minlength directly !!!!!! \n /\
                     Please make sure the minlenth to avoid overflow";
    TORCH_CHECK( minlength > 0, "get minlength == 0, maybe you not set minlength arg");
    auto out = empty({minlength}, options);
    return bincount_out_tpu(self, weights, minlength, out);    
}
// test case
// https://pytorch.org/docs/2.1/generated/torch.bincount.html#torch.bincount
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "bincount.out",  bincount_out_tpu);
 m.impl ( "bincount",      bincount_tpu);
}

}