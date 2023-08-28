#include <torch/torch.h>
#include <torch/library.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>

#include "common/config.h"
using namespace at;

namespace c10d {
namespace ops {

c10::intrusive_ptr<Work> reduce_tpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
    auto tensor_vec = tensors.vec();
    return process_group->reduce(
        tensor_vec,
        ReduceOptions{*reduce_op.get(), root_rank, root_tensor, std::chrono::milliseconds(timeout)});
}

TORCH_LIBRARY_IMPL(c10d, TPU, m) {
    m.impl("reduce_", reduce_tpu_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
    m.impl("reduce_", reduce_tpu_);
}

} // namespace ops
} // namespace c10d