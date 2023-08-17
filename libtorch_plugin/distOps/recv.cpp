
#include <torch/torch.h>
#include <torch/library.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include "common/config.h"

namespace c10d {
namespace ops {

c10::intrusive_ptr<Work> recv_tpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->recv(
      tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}


TORCH_LIBRARY_IMPL(c10d, TPU, m) {
    m.impl("recv_", recv_tpu_);
}

} // namespace ops
} // namespace c10d
