#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

#define TPU_MAX_CONCAT_NUM 10

namespace at
{
Tensor & cat_out_tpu ( const ITensorListRef & tensors, int64_t dim, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( out );
  if ( tensors.size() > TPU_MAX_CONCAT_NUM )
  {
    std::vector<Tensor> tensors_cpu;
    for ( auto tensor : tensors )
    {
      CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( tensor );
      tensors_cpu.push_back ( tensor.cpu() );
      ITensorListRef tensors_lis_cpu ( tensors_cpu.data(), tensors_cpu.size() );
      auto out_cpu = cat ( tensors_lis_cpu, dim );
      tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
    }
  }
  else
  {
    std::vector<TensorDescriptor_t> inputDescs;
    std::vector<const void *> inputs;
    std::vector<Tensor> contiguous_tensors;
    for ( auto tensor : tensors )
    {
      CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( tensor );
      contiguous_tensors.push_back ( tensor.contiguous() );
      inputDescs.push_back ( tpu::TPUGenerateTensorDesc ( contiguous_tensors.back() ) );
      inputs.push_back ( ADDR_IN_DEVICE ( contiguous_tensors.back() ) );
    }
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    auto status = sgdnn_concat (
                  tpu::TPUGetDeviceHandle(),
                  inputDescs.data(),
                  inputs.data(),
                  inputs.size(),
                  tpu::TPUGenerateTensorDesc ( out ),
                  ADDR_IN_DEVICE ( out ),
                  dim );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::CONCAT, timer.ElapsedUS() );
#endif
  }
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "cat.out", cat_out_tpu );
}
} // namespace at
