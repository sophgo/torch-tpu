#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

namespace at {
Tensor where_self_tpu ( const Tensor & condition, const Tensor & self, const Tensor & other ) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( condition );
  auto condition_ = condition.contiguous();
  if ( self.dim() > 0 )
  {
    CHECK_TENSOR_IN_DEVICE ( self );
  }
  if ( other.dim() > 0 )
  {
    CHECK_TENSOR_IN_DEVICE ( other );
  }
  TORCH_CHECK ( condition_.dim() > 0 );
  std::vector<int64_t> sizes_vec ( condition_.dim() );
  for ( auto i = 0; i < condition_.dim(); ++i )
  {
    sizes_vec[i] = condition_.size ( i );
  }
  if ( self.dim() > 0 )
  {
    TORCH_CHECK ( condition_.dim() == self.dim() );
    for ( auto i = 0; i < self.dim(); ++i )
    {
      sizes_vec [i] = std::max ( sizes_vec[i], self.size ( i ) );
    }
  }
  if ( other.dim() > 0 )
  {
    TORCH_CHECK ( condition_.dim() == other.dim() );
    for ( auto i = 0; i < other.dim(); ++i )
    {
      sizes_vec [i] = std::max ( sizes_vec[i], other.size ( i ) );
    }
  }
  TensorOptions options = TensorOptions ( condition_.device() ).dtype ( self.dtype() );
  IntArrayRef sizes ( sizes_vec.data(), sizes_vec.size() );
  auto out = torch::empty ( sizes, options );
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status = sgdnnWhere (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor ( condition_ ),
                       tpu::TPUGenerateSgdnnTensor ( self ),
                       tpu::TPUGenerateSgdnnTensor ( other ),
                       tpu::TPUGenerateSgdnnTensor ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::WHERE, timer.ElapsedUS() );
#endif
  return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "where.self", where_self_tpu );
}

} // namespace at
