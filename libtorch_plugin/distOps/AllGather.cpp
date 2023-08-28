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

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_tpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{std::chrono::milliseconds(timeout)});

  // Copy output tensors (not storage) so that this can be used in a functional
  // manner
  return std::
      tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>(
          output_tensors, work);
}

TORCH_LIBRARY_IMPL(c10d, TPU, m) {
    m.impl("allgather_", allgather_tpu_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
    m.impl("allgather_", allgather_tpu_);
}

} // namespace ops
} // namespace c10d