#pragma once

#include <torch/torch.h>
#include <torch/script.h>

namespace tpu
{

void MoveModuleToTPUDevice ( torch::nn::Module & Module );

void DumpParameterValues ( torch::nn::Module & Module, const std::string & Name );

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
