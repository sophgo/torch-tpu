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
  auto T = torch::empty ( {1, 1, 10, 10}, Device );
  auto A = torch::randn ( {1, 1, 10, 10} );
  for ( int i = 0; i < A.numel(); ++i )
  {
    A.data_ptr<float>() [i] = i;
  }
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
  auto E = torch::ones ( {1, 1, 10, 10} ).to ( Device );
  B.backward ( E );
  auto BB = A.grad().to ( torch::Device ( "cpu" ) );
  for ( int i = 0; i < BB.numel(); ++i )
  {
    std::cout << BB.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
#endif
  return 0;
}
