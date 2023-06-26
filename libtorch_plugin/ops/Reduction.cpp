#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

namespace at
{
Tensor & mean_out_tpu ( const Tensor & self, OptionalIntArrayRef dim_opt, bool keepdim, c10::optional<ScalarType> dtype_opt, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = mean ( self.cpu(), dim_opt, keepdim, dtype_opt );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto reduce_dim = dim_opt.value_or ( IntArrayRef {} );
  std::vector<int> reduction_dim_vec;
  if ( reduce_dim.size() > 0 )
  {
    for ( auto it : reduce_dim )
    {
      reduction_dim_vec.push_back ( it < 0 ? it + self.dim() : it );
    }
    std::sort ( reduction_dim_vec.begin(), reduction_dim_vec.end() );
  }
  else
  {
    for ( auto i = 0; i < self.dim(); ++i )
    {
      reduction_dim_vec.push_back ( i );
    }
  }
  for ( auto i = 0; i < reduction_dim_vec.size() - 1; ++i )
  {
    TORCH_CHECK ( reduction_dim_vec[i] + 1 == reduction_dim_vec[i + 1], "Reduction only supports contiguous reduction dimension now" );
  }
  bm_status_t status = sgdnn_reduce (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       tpu::TPUGenerateTensorDesc ( out ),
                       ADDR_IN_DEVICE ( out ),
                       reduction_dim_vec[0],
                       reduction_dim_vec.back() + 1,
                       keepdim,
                       Reduction_Mean );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::REDUCE_MEAN, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "mean.out", mean_out_tpu );
}

Tensor & sum_IntList_out_tpu ( const Tensor & self, OptionalIntArrayRef dim_opt, bool keepdim, c10::optional<ScalarType> dtype_opt, Tensor & out )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = sum ( self.cpu(), dim_opt, keepdim, dtype_opt );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  auto reduce_dim = dim_opt.value_or ( IntArrayRef {} );
  std::vector<int> reduction_dim_vec;
  if ( reduce_dim.size() > 0 )
  {
    for ( auto it : reduce_dim )
    {
      reduction_dim_vec.push_back ( it < 0 ? it + self.dim() : it );
    }
    std::sort ( reduction_dim_vec.begin(), reduction_dim_vec.end() );
  }
  else
  {
    for ( auto i = 0; i < self.dim(); ++i )
    {
      reduction_dim_vec.push_back ( i );
    }
  }
  for ( auto i = 0; i < reduction_dim_vec.size() - 1; ++i )
  {
    TORCH_CHECK ( reduction_dim_vec[i] + 1 == reduction_dim_vec[i + 1], "Reduction only supports contiguous reduction dimension now" );
  }
  bm_status_t status = sgdnn_reduce (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateTensorDesc ( self ),
                       ADDR_IN_DEVICE ( self ),
                       tpu::TPUGenerateTensorDesc ( out ),
                       ADDR_IN_DEVICE ( out ),
                       reduction_dim_vec[0],
                       reduction_dim_vec.back() + 1,
                       keepdim,
                       Reduction_Sum );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::REDUCE_SUM, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "sum.IntList_out", sum_IntList_out_tpu );
}
} // namespace at
