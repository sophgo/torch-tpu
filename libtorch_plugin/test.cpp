#include <torch/torch.h>
#include <torch/script.h>

struct Net : torch::nn::Module
{
  torch::jit::Module module_;
  Net ( const torch::jit::Module &module )
  {
    module_ = module;
    for ( const auto &mod : module_.named_children() )
    {
      register_module ( mod.name, std::make_shared<Net> ( mod.value ) );
    }
    for ( const auto &par : module_.named_parameters ( false ) )
    {
      register_parameter ( par.name, par.value );
    }
  }
  torch::Tensor forward ( const torch::Tensor &Input )
  {
    std::vector<torch::jit::IValue> Inputs;
    Inputs.push_back ( Input );
    return module_.forward ( Inputs ).toTensor();
  }
};

int main()
{
  // Create Resnet50 net
  std::string ModelPath = "./Resnet50_Own.pt";
  auto Resnet50 = std::make_shared<Net> ( torch::jit::load ( ModelPath ) );
  torch::Device Device ( "privateuseone" );
  torch::Tensor Input = torch::randn ( { 1, 3, 224, 224 } );
  Resnet50->to ( Device );
  return 0;
}
