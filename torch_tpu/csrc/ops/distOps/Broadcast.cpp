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

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> broadcast_tpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->broadcast(
              tensor_vec,
              BroadcastOptions{
                  root_rank, root_tensor, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

TORCH_LIBRARY_IMPL(c10d, TPU, m) {
    m.impl("broadcast_", broadcast_tpu_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
    m.impl("broadcast_", broadcast_tpu_);
}

} // namespace ops
} // namespace c10d