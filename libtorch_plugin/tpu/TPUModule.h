#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>

namespace tpu
{

typedef enum
{
  ALGORITHM_ACCURACY_FP32 = 0x0,
  ALGORITHM_ACCURACY_FP16,
}
AlgorithmAccuracy_t;

AlgorithmAccuracy_t GetConvolutionBackwardAccuracy();
void SetConvolutionBackwardAccuracy ( AlgorithmAccuracy_t Accuracy );

AlgorithmAccuracy_t GetConvolutionForwardAccuracy();
void SetConvolutionForwardAccuracy ( AlgorithmAccuracy_t Accuracy );

void MoveModuleToTPUDevice ( torch::nn::Module & Module );

void MoveModuleToCPUDevice ( torch::nn::Module & Module );

class TorchscriptModule : public torch::nn::Module
{
public:

  TorchscriptModule ( const std::string & Path );

  TorchscriptModule ( const torch::jit::Module & Module );

  torch::Tensor forward ( const torch::Tensor & Input );

  std::vector<torch::Tensor> forward ( const std::vector<torch::Tensor> & Inputs );

private:

  void Register();

  torch::jit::Module Module_;
};

std::vector<at::Tensor> GetNamedParameters ( const torch::nn::Module & Module );

} // namespace tpu
