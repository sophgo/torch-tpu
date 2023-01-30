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

} // namespace tpu
