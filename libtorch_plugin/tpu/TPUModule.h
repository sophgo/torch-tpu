#pragma once

#include <torch/torch.h>
#include <torch/script.h>

namespace tpu
{

typedef enum
{
  CONVOLUTION_BACKWARD_ACCURACY_FP32 = 0x0,
  CONVOLUTION_BACKWARD_ACCURACY_FP16,
  CONVOLUTION_BACKWARD_ACCURACY_AMP
}
ConvolutionBackwardAccuracy_t;

ConvolutionBackwardAccuracy_t GetConvolutionBackwardAccuracy();
void SetConvolutionBackwardAccuracy ( ConvolutionBackwardAccuracy_t Accuracy );

void MoveModuleToTPUDevice ( torch::nn::Module & Module );

void MoveModuleToCPUDevice ( torch::nn::Module & Module );

class TorchscriptModule : public torch::nn::Module
{
public:

  TorchscriptModule ( const std::string & Path );

  TorchscriptModule ( const torch::jit::Module & Module );

  torch::Tensor forward ( const torch::Tensor & Input );

private:

  void Register();

  torch::jit::Module Module_;
};

} // namespace tpu
