#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

namespace at{
Tensor where_self_tpu(const Tensor & condition, const Tensor & self, const Tensor & other) {

    auto out = where(condition.cpu(), self.cpu(), other.cpu());
    return TENSOR_TO_TPU(out);

}


TORCH_LIBRARY_IMPL(aten, TPU, m)
{
    m.impl ("where.self", where_self_tpu);
}

} // namespace at