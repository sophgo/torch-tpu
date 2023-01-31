#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>

int main()
{
  torch::Device Device ( "privateuseone" );
#if 0
  std::string ModelPath = "./Resnet50_Own.pt";
  auto Resnet50 = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  auto A = torch::randn ( { 1, 3, 224, 224 } ).to ( Device );
  tpu::MoveModuleToTPUDevice ( *Resnet50 );
  Resnet50->forward ( A );
#endif
  auto A = torch::randn ( {1, 1, 10, 10} );
  for ( int i = 0; i < A.numel(); ++i )
  {
    A.data_ptr<float>() [i] = i;
  }
  for ( int i = 0; i < A.numel(); ++i )
  {
    std::cout << A.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
  A = A.to ( Device );
  auto B = A.add ( A );
  auto BB = B.to ( torch::Device ( "cpu" ) );
  for ( int i = 0; i < BB.numel(); ++i )
  {
    std::cout << BB.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
