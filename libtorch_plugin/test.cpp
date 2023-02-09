#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>

int main()
{
  torch::Device Device ( "privateuseone:0" );
  //torch::Device Device ( "cpu" );
#if 0
  std::string ModelPath = "./Resnet50_Own.pt";
  auto Resnet50 = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  auto A = torch::randn ( { 1, 3, 224, 224 } ).to ( Device );
  tpu::MoveModuleToTPUDevice ( *Resnet50 );
  Resnet50->forward ( A );
#else
  auto T = torch::empty ( {2, 3, 4, 5}, Device );
  auto A = torch::randn ( {2, 3, 4, 5} );
//  for ( int i = 0; i < A.numel(); ++i )
//  {
//    A.data_ptr<float>() [i] = i;
//  }
#if 0
  A = A.to ( torch::kFloat16 );
  for ( int i = 0; i < A.numel(); ++i )
  {
    std::cout << A.data_ptr<at::Half>() [i] << " ";
  }
  std::cout << std::endl;
#endif
  A = A.to ( Device ).set_requires_grad ( true );
  auto B = A + A;
  auto E = torch::ones ( {2, 3, 4, 5} ).to ( Device );
  B.backward ( E );
  auto BB = A.grad().to ( torch::Device ( "cpu" ) );
  for ( int i = 0; i < BB.numel(); ++i )
  {
    std::cout << BB.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
  auto Input = torch::randn ( {1, 3, 224, 224} ).to ( Device );
  torch::nn::Conv2d Conv2d ( torch::nn::Conv2dOptions ( 3, 32, {3, 3} ) );
  tpu::MoveModuleToTPUDevice ( *Conv2d );
  auto Output = Conv2d->forward ( Input );
#endif
  return 0;
}
