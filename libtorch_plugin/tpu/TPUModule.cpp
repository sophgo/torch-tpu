#include <TPUModule.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>

namespace tpu
{

void MoveModuleToTPUDevice ( torch::nn::Module & Module )
{
  auto Device = tpu::TPUGetCurrentDevice();
  for ( auto & Mod : Module.named_children() )
  {
    MoveModuleToTPUDevice ( *Mod.value() );
  }
  for ( auto & Par : Module.named_parameters ( false ) )
  {
    auto tmp = Par->to ( Device );
    Par->unsafeGetTensorImpl()->_change_backend_component_keys ( Device );
    Par->unsafeGetTensorImpl()->set_storage_keep_dtype ( tmp.storage() );
  }
  for ( auto & Buf : Module.named_buffers ( false ) )
  {
    if ( Buf.key() == "running_mean" || Buf.key() == "running_var" )
    {
      auto tmp = Buf->to ( Device );
      Buf->unsafeGetTensorImpl()->_change_backend_component_keys ( Device );
      Buf->unsafeGetTensorImpl()->set_storage_keep_dtype ( tmp.storage() );
    }
  }
}

void MoveModuleToCPUDevice ( torch::nn::Module & Module )
{
  auto Device = torch::Device ( "cpu" );
  for ( auto & Mod : Module.named_children() )
  {
    MoveModuleToCPUDevice ( *Mod.value() );
  }
  for ( auto & Par : Module.named_parameters ( false ) )
  {
    auto tmp = Par->to ( Device );
    Par->unsafeGetTensorImpl()->_change_backend_component_keys ( Device );
    Par->unsafeGetTensorImpl()->set_storage_keep_dtype ( tmp.storage() );
  }
  for ( auto & Buf : Module.named_buffers ( false ) )
  {
    if ( Buf.key() == "running_mean" || Buf.key() == "running_var" )
    {
      auto tmp = Buf->to ( Device );
      Buf->unsafeGetTensorImpl()->_change_backend_component_keys ( Device );
      Buf->unsafeGetTensorImpl()->set_storage_keep_dtype ( tmp.storage() );
    }
  }
}

TorchscriptModule::TorchscriptModule ( const std::string & Path )
{
  Module_ = torch::jit::load ( Path );
  Register();
}

TorchscriptModule::TorchscriptModule ( const torch::jit::Module & Module )
{
  Module_ = Module;
  Register();
}

torch::Tensor TorchscriptModule::forward ( const torch::Tensor & Input )
{
  std::vector<torch::jit::IValue> Inputs;
  Inputs.push_back ( Input );
  return Module_.forward ( Inputs ).toTensor();
}

std::vector<torch::Tensor> TorchscriptModule::forward ( const std::vector<torch::Tensor> & Inputs )
{
  std::vector<torch::jit::IValue> Inputs_;
  for ( auto i = 0; i < Inputs.size(); ++i )
  {
    Inputs_.push_back ( Inputs[i] );
  }
  auto Outputs_ = Module_.forward ( Inputs_ );
  std::vector<torch::Tensor> Outputs;
  if ( Outputs_.isTensor() )
  {
    Outputs = { Outputs_.toTensor() };
  }
  else if ( Outputs_.isTuple() )
  {
    auto OutputsTuple = Outputs_.toTuple()->elements();
    for ( auto it : OutputsTuple )
    {
      Outputs.push_back ( it.toTensor() );
    }
  }
  return Outputs;
}

void TorchscriptModule::Register()
{
  for ( const auto & Mod : Module_.named_children() )
  {
    register_module ( Mod.name, std::make_shared<TorchscriptModule> ( Mod.value ) );
  }
  for ( const auto & Par : Module_.named_parameters ( false ) )
  {
    register_parameter ( Par.name, Par.value );
  }
  for ( const auto & Buf : Module_.named_buffers ( false ) )
  {
    register_buffer ( Buf.name, Buf.value );
  }
}

static AlgorithmAccuracy_t kConvolutionBackwardAccuracy = ALGORITHM_ACCURACY_FP32;

AlgorithmAccuracy_t GetConvolutionBackwardAccuracy()
{
  return kConvolutionBackwardAccuracy;
}

void SetConvolutionBackwardAccuracy ( AlgorithmAccuracy_t Accuracy )
{
  kConvolutionBackwardAccuracy = Accuracy;
}

static AlgorithmAccuracy_t kConvolutionForwardAccuracy = ALGORITHM_ACCURACY_FP32;

AlgorithmAccuracy_t GetConvolutionForwardAccuracy()
{
  return kConvolutionForwardAccuracy;
}

void SetConvolutionForwardAccuracy ( AlgorithmAccuracy_t Accuracy )
{
  kConvolutionForwardAccuracy = Accuracy;
}

static inline void GetNamedParameters_ (
const torch::nn::Module & Module, std::vector<at::Tensor> & Params )
{
  for ( auto & Par : Module.named_parameters ( false ) )
  {
    Params.push_back ( Par.value() );
  }
  for ( auto & Mod : Module.named_children() )
  {
    GetNamedParameters_ ( *Mod.value(), Params );
  }
}

std::vector<at::Tensor> GetNamedParameters ( const torch::nn::Module & Module )
{
  std::vector<at::Tensor> Params;
  GetNamedParameters_ ( Module, Params );
  return Params;
}

} // namespace tpu
