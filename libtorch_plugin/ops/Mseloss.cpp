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
Tensor mse_loss_tpu(const at::Tensor & self, const at::Tensor & target, int64_t reduction)
{
    if ( self.numel() == 0 && reduction == 1) { return torch::tensor( NAN, self.options() ); }
    CHECK_TENSOR_IN_DEVICE ( self );
    CHECK_TENSOR_IN_DEVICE ( target );
#if 0
    auto ans = sub (self.cpu(), target.cpu());
    ans = mul( ans, ans );
    // reduction == 0: none
    if ( reduction == 0 ){
        return ans;
    }
    // reduction == 1: mean
    else if ( reduction == 1 ){
        auto ans_mean = torch::sum( ans );
        ans_mean = div( ans_mean, ans.numel());
        return ans_mean;
    }
    // reduction == 0: sum
    else if ( reduction == 2 ){
        ans = torch::sum( ans );
        return ans;
    }
#else
    Tensor out;
    if ( reduction == 0 ){
        out = torch::empty( self.sizes(), self.options() );
    }
    else if ( reduction != 0 ){
        out = torch::tensor( 0., self.options() );
    }
    TIMING_START;
    bm_status_t status = sgdnnMseloss( tpu::TPUGetDeviceHandle(),
                                        tpu::TPUGenerateSgdnnTensor ( self ),
                                        tpu::TPUGenerateSgdnnTensor ( target ),
                                        tpu::TPUGenerateSgdnnTensor ( out ),
                                        reduction );
    TORCH_CHECK ( status == BM_SUCCESS );
    TIMING_END(tpu::MSE_LOSS);
    return out;
#endif
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "mse_loss",  mse_loss_tpu);
}

/*
* x ->      "-"        -> z1     ->  "pow"       -> z2     -> "reduce"     -> L 
* Y -> 
* ==backward
* dC/dx <- "x dz1/dx " <- dC/dz1 <- "x dz2/dz1 " <- dC/dz2 <- "x (dL/dz2)" <- dC/dL
*/
Tensor mse_loss_backward_tpu( const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction ){
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( target );
  Tensor grad_in;
#if 0
  LOG( WARNING ) << " mes_loss_backward use cpu impl";
  auto grad_in_cpu = mse_loss_backward(grad_output.cpu(), self.cpu(), target.cpu(), reduction );
  grad_in = grad_in_cpu.to(self.device());
#else
  TIMING_START;
  // TODO: use a kernel func 
  auto dz2_dz1 = 2 * (self - target);
  if (reduction == 0 ) // none
  {
    grad_in = grad_output * dz2_dz1;
  }
  else if ( reduction == 1) //mean
  {    
    grad_in = (grad_output / self.numel()) * dz2_dz1;
  }
  else if ( reduction == 2) //sum 
  {
    grad_in = grad_output * dz2_dz1;
  }
  else{
    TORCH_CHECK( false );
  }
  TIMING_END(tpu::MSE_LOSS_BACKWARD);
#endif
  return grad_in;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "mse_loss_backward" , mse_loss_backward_tpu );
}
}
