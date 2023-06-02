#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

namespace at {
Tensor where_self_tpu ( const Tensor & condition, const Tensor & self, const Tensor & other ) {
  CHECK_TENSOR_IN_DEVICE ( condition );
  if ( self.dim() > 0 )
  {
    CHECK_TENSOR_IN_DEVICE ( self );
  }
  if ( other.dim() > 0 )
  {
    CHECK_TENSOR_IN_DEVICE ( other );
  }
  TORCH_CHECK ( condition.dim() > 0 );
  std::vector<int64_t> sizes_vec ( condition.dim() );
  for ( auto i = 0; i < condition.dim(); ++i )
  {
    sizes_vec[i] = condition.size ( i );
  }
  if ( self.dim() > 0 )
  {
    TORCH_CHECK ( condition.dim() == self.dim() );
    for ( auto i = 0; i < self.dim(); ++i )
    {
      sizes_vec [i] = std::max ( sizes_vec[i], self.size ( i ) );
    }
  }
  if ( other.dim() > 0 )
  {
    TORCH_CHECK ( condition.dim() == other.dim() );
    for ( auto i = 0; i < other.dim(); ++i )
    {
      sizes_vec [i] = std::max ( sizes_vec[i], other.size ( i ) );
    }
  }
  TensorOptions options = TensorOptions ( condition.device() ).dtype ( self.dtype() );
  IntArrayRef sizes ( sizes_vec.data(), sizes_vec.size() );
  auto out = torch::empty ( sizes, options );
  bm_status_t status = sgdnn_where (
                       tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateTensorDesc ( condition ),
                       ADDR_IN_DEVICE ( condition ),
                       tpu::TPUGenerateTensorDesc ( self ),
                       self.dim() == 0 ? self.cpu().data_ptr() : ADDR_IN_DEVICE ( self ),
                       tpu::TPUGenerateTensorDesc ( other ),
                       other.dim() == 0 ? other.cpu().data_ptr() : ADDR_IN_DEVICE ( other ),
                       tpu::TPUGenerateTensorDesc ( out ),
                       ADDR_IN_DEVICE ( out ) );
  TORCH_CHECK ( status == BM_SUCCESS );
  return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "where.self", where_self_tpu );
}

} // namespace at
