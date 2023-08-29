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
    bm_status_t status = sgdnnMseloss( tpu::TPUGetDeviceHandle(),
                                        tpu::TPUGenerateSgdnnTensor ( self ),
                                        tpu::TPUGenerateSgdnnTensor ( target ),
                                        tpu::TPUGenerateSgdnnTensor ( out ),
                                        reduction );
    TORCH_CHECK ( status == BM_SUCCESS );
    return out;
#endif
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "mse_loss",  mse_loss_tpu);
}
}
