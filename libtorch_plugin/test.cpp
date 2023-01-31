#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>

int main()
{
  std::string ModelPath = "./Resnet50_Own.pt";
  auto Resnet50 = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  torch::Device Device ( "privateuseone" );
  auto A = torch::randn ( { 1, 3, 224, 224 } ).to ( Device );
  tpu::MoveModuleToTPUDevice ( *Resnet50 );
  Resnet50->forward ( A );
  return 0;
}
