#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ConvUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <limits.h>

namespace at
{

Tensor & relu__tpu ( Tensor & self )
{
  CHECK_TENSOR_IN_DEVICE ( self );
#ifdef TPU_LIBTORCH_OP_COMPARE
  auto self_cpu = self.to ( torch::Device ( "cpu" ) );
  auto & output_exp = relu_ ( self_cpu );
#endif
  auto handle = tpu::TPUGetDeviceHandle();
  bm_status_t status = BM_SUCCESS;
  float alpha = 1.f;
  float beta = 0.f;
  auto self_desc = tpu::TPUGenerateTensorDesc ( self );
  ActivationDescriptor_t activation_desc =
  {
    .mode = Activation_Relu,
    .NanOpt = Not_Propagate_Nan,
    .coef = std::numeric_limits<double>::max()
  };
  status = sgdnn_activation_forward_cudnn (
           handle,
           activation_desc,
           &alpha,
           self_desc,
           ADDR_IN_DEVICE ( self ),
           &beta,
           self_desc,
           ADDR_IN_DEVICE ( self ) );
#ifdef TPU_LIBTORCH_OP_COMPARE
  std::cout << "Comparing inplace relu:"
            << " self shape = " << self.sizes()
            << " self dtype = " << self.dtype()
            << std::endl;
  auto output_got = self.to ( torch::Device ( "cpu" ) );
  tpu::TPUCompareResult ( output_got, output_exp );
#endif
  return self;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "relu_", relu__tpu );
}

} // namespace at
