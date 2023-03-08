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

static ConvolutionBackwardAccuracy_t kConvolutionBackwardAccuracy = CONVOLUTION_BACKWARD_ACCURACY_FP32;

ConvolutionBackwardAccuracy_t GetConvolutionBackwardAccuracy()
{
  return kConvolutionBackwardAccuracy;
}

void SetConvolutionBackwardAccuracy ( ConvolutionBackwardAccuracy_t Accuracy )
{
  kConvolutionBackwardAccuracy = Accuracy;
}

} // namespace tpu
