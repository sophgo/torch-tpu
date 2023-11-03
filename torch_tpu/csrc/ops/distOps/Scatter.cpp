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

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_tpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

TORCH_LIBRARY_IMPL(c10d, TPU, m) {
    m.impl("scatter_", scatter_tpu_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
    m.impl("scatter_", scatter_tpu_);
}

} // namespace ops
} // namespace c10d