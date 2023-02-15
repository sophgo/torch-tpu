#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>
#include <TPUTorchUtils.h>

int main()
{
  auto Device = tpu::TPUGetCurrentDevice();
  //torch::Device Device ( "cpu" );
#if 1
  std::string ModelPath = "../Resnet50_Own.pt";
  auto Resnet50 = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  auto A = torch::ones ( { 1, 3, 224, 224 } ).to ( Device );
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
  auto InputCPU = torch::ones ( {64, 3, 3, 3} );
  auto Input = InputCPU.to ( Device );
  Input.set_requires_grad ( true );
  torch::nn::Conv2d Conv2d ( torch::nn::Conv2dOptions ( 3, 32, {2, 2} ) );
  tpu::MoveModuleToTPUDevice ( *Conv2d );
  auto Output = Conv2d->forward ( Input );
  auto E = torch::ones ( Output.sizes() ).to ( Device );
  std::cout << Output.grad_fn() << std::endl;
  std::cout << Output.grad_fn()->name() << std::endl;
  //Output.set_requires_grad ( true );
  Output.backward ( E );
  auto OO = Input.grad();
#endif
  return 0;
}
