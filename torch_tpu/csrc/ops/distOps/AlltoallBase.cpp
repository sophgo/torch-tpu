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

c10::intrusive_ptr<Work> alltoall_base_tpu_(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    int64_t timeout) {
  return process_group->alltoall_base(
          output,
          input,
          output_split_sizes,
          input_split_sizes,
          AllToAllOptions{std::chrono::milliseconds(timeout)});
}

TORCH_LIBRARY_IMPL(c10d, TPU, m) {
    m.impl("alltoall_base_", alltoall_base_tpu_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
    m.impl("alltoall_base_", alltoall_base_tpu_);
}

} // namespace ops
} // namespace c10d