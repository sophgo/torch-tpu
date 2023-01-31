#include <TPUModule.h>

namespace tpu
{

void MoveModuleToTPUDevice ( torch::nn::Module & Module )
{
  static const torch::Device Device ( "privateuseone" );
  for ( auto & Mod : Module.named_children() )
  {
    MoveModuleToTPUDevice ( *Mod.value() );
  }
  for ( auto & Par : Module.named_parameters ( false ) )
  {
    Par.value().to ( Device );
  }
  for ( auto & Buf : Module.named_buffers ( false ) )
  {
    Buf.value().to ( Device );
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
    register_module ( Mod.name,
                      std::make_shared<TorchscriptModule> ( Mod.value ) );
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

} // namespace tpu
