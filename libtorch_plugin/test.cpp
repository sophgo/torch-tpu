#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>
#include <TPUTorchUtils.h>

int main()
{
  int batch = 1;
  auto Device = tpu::TPUGetCurrentDevice();
  //torch::Device Device ( "cpu" );
#if 1
  std::string ModelPath = "../Resnet50_Own.pt";
  auto Resnet50CPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  auto Resnet50TPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  tpu::MoveModuleToTPUDevice ( *Resnet50TPU );
  auto InputCPU = torch::ones ( { batch, 3, 224, 224 } );
  auto InputTPU = InputCPU.to ( Device );
  int Loops = 1;
  torch::Tensor OutputCPU, OutputTPU;
  tpu::Timer timer;
  timer.Start();
  for ( int i = 0; i < Loops; ++i )
  {
    OutputCPU = Resnet50CPU->forward ( InputCPU );
  }
  unsigned long elapsed_us_per_loop = ( double ) timer.ElapsedUS() / Loops;
  std::cout << "CPU Elapsed time = " << elapsed_us_per_loop << "us" << std::endl;
  timer.Start();
  for ( int i = 0; i < Loops; ++i )
  {
    OutputTPU = Resnet50TPU->forward ( InputTPU );
  }
  elapsed_us_per_loop = ( double ) timer.ElapsedUS() / Loops;
  std::cout << "TPU Elapsed time = " << elapsed_us_per_loop << "us" << std::endl;
  auto OutputGot = OutputTPU.to ( torch::Device ( "cpu" ) );
  auto OutputExp = OutputCPU;
  torch::Tensor LabelsTPU;
  auto Loss = torch::nll_loss ( OutputTPU, LabelsTPU );
  //tpu::TPUCompareResult ( OutputGot, OutputExp );
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
