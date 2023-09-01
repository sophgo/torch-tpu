#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include "common/config.h"

namespace at
{
Tensor & normal_tpu(Tensor & self, double mean, double std, c10::optional<at::Generator> generator)
{
  CHECK_TENSOR_IN_DEVICE ( self );
#if 1
  LOG ( WARNING ) << "normal use cpu impl";
  std::vector<int64_t> sizes_vec ( self.dim() );
  for ( int i = 0; i < self.dim(); i++ ) { sizes_vec[i] = self.size( i ); }
  
  IntArrayRef sizes ( sizes_vec.data(), sizes_vec.size() );
  TensorOptions TenOption = self.options().device("cpu");

  auto out_cpu = normal(mean, std, sizes, generator, TenOption);
  tpu::TPUCopyHostToDevice ( self.data_ptr(), out_cpu.contiguous().data_ptr(), self.nbytes() );
#else
  //TODO
#endif
  return self;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "normal_", normal_tpu );
}
} // namespace at
