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

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_tpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
    auto tensor_vec = tensors.vec();
    auto work = process_group->allreduce(
        tensor_vec,
        AllreduceOptions{*reduce_op.get(), std::chrono::milliseconds(timeout)});
    // Return input tensors as output tensors to make inplace allreduce look like
    // a functional API, so that make_fx can correctly build the dependencies in
    // the graph later.
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
        std::move(tensor_vec), work);
}

TORCH_LIBRARY_IMPL(c10d, TPU, m) {
    m.impl("allreduce_", allreduce_tpu_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
    m.impl("allreduce_", allreduce_tpu_);
}

} // namespace ops
} // namespace c10d