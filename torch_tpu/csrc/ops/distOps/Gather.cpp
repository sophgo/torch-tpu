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

c10::intrusive_ptr<Work> gather_tpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}

TORCH_LIBRARY_IMPL(c10d, TPU, m) {
    m.impl("gather_", gather_tpu_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
    m.impl("gather_", gather_tpu_);
}

} // namespace ops
} // namespace c10d