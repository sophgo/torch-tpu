#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

namespace at
{
Tensor & normal_tpu(Tensor & self, double mean, double std, c10::optional<at::Generator> generator)
{
  CHECK_TENSOR_IN_DEVICE ( self );
#if 1
  CPU_IMPL_WARNING();
  TIMING_START;
  std::vector<int64_t> sizes_vec ( self.dim() );
  for ( int i = 0; i < self.dim(); i++ ) { sizes_vec[i] = self.size( i ); }
  
  IntArrayRef sizes ( sizes_vec.data(), sizes_vec.size() );
  TensorOptions TenOption = self.options().device("cpu").dtype(torch::kFloat);

  auto out_cpu = normal(mean, std, sizes, c10::nullopt, TenOption);
  self = out_cpu.to(self.device()).to(self.dtype());
  TIMING_END(tpu::CPU_LAYER);
#else
  //TODO
#endif
  SHOW_TENSOR_OP(self);
  return self;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "normal_", normal_tpu );
}
} // namespace at
