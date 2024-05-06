#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
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
  if ( tensors.size() > TPU_MAX_CONCAT_NUM || flag )
  {
    CPU_IMPL_WARNING();
    TIMING_START;
    std::vector<Tensor> tensors_cpu;
    for ( auto tensor : tensors )
    {
      CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( tensor );
      tensors_cpu.push_back ( tensor.cpu() );
      ITensorListRef tensors_lis_cpu ( tensors_cpu.data(), tensors_cpu.size() );
      auto out_cpu = cat ( tensors_lis_cpu, dim );
      tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
    }
    TIMING_END(tpu::CPU_LAYER);
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

    TIMING_START;
    #if defined BACKEND_1684X
    auto status = sgdnnConcat ( tpu::TPUGetDeviceHandle(),
                                inputs.data(),
                                inputs.size(),
                                dim,
                                tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
    #elif defined BACKEND_SG2260
    auto status = sgdnnConcat ( c10_tpu::getCurrentTPUStream(),
                                inputs.data(),
                                inputs.size(),
                                dim,
                                tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == tpuRtSuccess );
    #endif
    TIMING_END ( tpu::CONCAT );
  }
  SHOW_TENSOR_OP(out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "cat.out", cat_out_tpu );
}
} // namespace at
