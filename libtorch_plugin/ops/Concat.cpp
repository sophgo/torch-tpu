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
  // if input.dtype != output.dtype , use cpu to convert.
  int flag = 0;
  for (auto tensor : tensors) {
    if (tensor.cpu().scalar_type() != out.cpu().scalar_type()) {
      flag = 1;
      break;
    }
  }
  if(flag) LOG(WARNING) << "concat use cpu impl";
  if ( tensors.size() > TPU_MAX_CONCAT_NUM || flag )
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
    std::vector<SgdnnTensor_t> inputs;
    std::vector<Tensor> contiguous_tensors;
    for ( auto tensor : tensors )
    {
      CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( tensor );
      contiguous_tensors.push_back ( tensor.contiguous() );
      inputs.push_back ( tpu:: TPUGenerateSgdnnTensor ( contiguous_tensors.back() ) );
    }
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    auto status = sgdnnConcat ( tpu::TPUGetDeviceHandle(),
                                inputs.data(),
                                inputs.size(),
                                dim,
                                tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
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
