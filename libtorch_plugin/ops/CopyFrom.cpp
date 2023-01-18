#include <torch/library.h>
#include <ATen/core/TensorBase.h>

#include <iostream>

namespace at
{
Tensor _copy_from_tpu ( const Tensor & self, const Tensor & dst,
                        bool non_blocking )
{
  std::cout << "This is _copy_from_tpu" << std::endl;
  std::cout << "self device = " << self.device() << std::endl;
  std::cout << "dst device = " << dst.device() << std::endl;
  std::cout << "self is contiguous = " << self.is_contiguous() << std::endl;
  std::cout << "dst is contiguous = " << dst.is_contiguous() << std::endl;
  auto tensor = dst;
  return tensor;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "_copy_from", _copy_from_tpu );
}
} // namespace at
