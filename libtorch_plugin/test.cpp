#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>
#include <TPUTorchUtils.h>

int main()
{
  int Batch = 1;
  int Loops = 1;
  auto TPU = tpu::TPUGetCurrentDevice();
  auto CPU = torch::Device ( "cpu" );
#if 1
  std::string ModelPath = "../Resnet50_Own.pt";
  auto Resnet50CPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  auto Resnet50TPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  Resnet50CPU->train();
  Resnet50TPU->train();
  tpu::MoveModuleToTPUDevice ( *Resnet50TPU );
  torch::optim::SGD optimizerCPU ( Resnet50CPU->parameters(), /*lr=*/0.01 );
  torch::optim::SGD optimizerTPU ( Resnet50TPU->parameters(), /*lr=*/0.01 );
  optimizerCPU.zero_grad();
  optimizerTPU.zero_grad();
  auto InputCPU = torch::ones ( { Batch, 3, 224, 224 } );
  auto InputTPU = torch::ones ( { Batch, 3, 224, 224 } ).to ( TPU );
  InputCPU.set_requires_grad ( true );
  InputTPU.set_requires_grad ( true );
  torch::Tensor OutputCPU, OutputTPU;
  unsigned long elapsed_us_per_loop = 0;
  tpu::Timer timer;
  timer.Start();
  tpu::OpTimer::Instance().Clear();
  for ( int i = 0; i < Loops; ++i )
  {
    OutputCPU = Resnet50CPU->forward ( InputCPU );
  }
  elapsed_us_per_loop = ( double ) timer.ElapsedUS() / Loops;
  std::cout << "Resnet50(Batch = " << Batch << ") Forward CPU Elapsed time = " << elapsed_us_per_loop << "us" << std::endl;
  timer.Start();
  for ( int i = 0; i < Loops; ++i )
  {
    OutputTPU = Resnet50TPU->forward ( InputTPU );
  }
  elapsed_us_per_loop = ( double ) timer.ElapsedUS() / Loops;
  std::cout << "Resnet50(Batch = " << Batch << ") Forward TPU Elapsed time = " << elapsed_us_per_loop << "us" << std::endl;
  auto OutputGot = OutputTPU.to ( CPU );
  auto OutputExp = OutputCPU;
  tpu::TPUCompareResult ( OutputGot, OutputExp );
  //torch::Tensor LabelsTPU;
  //auto Loss = torch::nll_loss ( OutputTPU, LabelsTPU );
  auto BackwardInputCPU = torch::ones ( OutputTPU.sizes() );
  auto BackwardInputTPU = torch::ones ( OutputTPU.sizes() ).to ( TPU );
  timer.Start();
  for ( int i = 0; i < Loops; ++i )
  {
    OutputCPU.backward ( BackwardInputCPU );
  }
  elapsed_us_per_loop = ( double ) timer.ElapsedUS() / Loops;
  std::cout << "Resnet50(Batch = " << Batch << ") Backward CPU Elapsed time = " << elapsed_us_per_loop << "us" << std::endl;
  timer.Start();
  for ( int i = 0; i < Loops; ++i )
  {
    OutputTPU.backward ( BackwardInputTPU );
  }
  elapsed_us_per_loop = ( double ) timer.ElapsedUS() / Loops;
  std::cout << "Resnet50(Batch = " << Batch << ") Backward TPU Elapsed time = " << elapsed_us_per_loop << "us" << std::endl;
  tpu::OpTimer::Instance().Dump();
  auto GradInputGot = InputTPU.grad().to ( CPU );
  auto GradInputExp = InputCPU.grad();
  tpu::TPUCompareResult ( GradInputGot, GradInputExp );
#else
#if 0
#endif
  auto InputCPU = torch::ones ( {1, 3, 3, 3} );
  auto InputTPU = InputCPU.to ( TPU );
  InputCPU.set_requires_grad ( true );
  InputTPU.set_requires_grad ( true );
  torch::nn::Conv2d Conv2dTPU ( torch::nn::Conv2dOptions ( 3, 32, {2, 2} ) );
  //auto OutputCPU = Conv2dCPU->forward ( InputCPU );
  //auto Conv2dTPU ( Conv2dCPU );
  tpu::MoveModuleToTPUDevice ( *Conv2dTPU );
  auto OutputTPU = Conv2dTPU->forward ( InputTPU );
  auto LossCPU = torch::ones ( OutputTPU.sizes() );
  auto LossTPU = LossCPU.to ( TPU );
  //Output.set_requires_grad ( true );
  //OutputCPU.backward ( GradInputCPU );
  std::cout << "OutputTPU requires grad = " << OutputTPU.requires_grad() << std::endl;
  OutputTPU.backward ( LossTPU );
  auto GradInputGot = InputTPU.grad().to ( CPU );
  for ( auto i = 0; i < GradInputGot.numel(); ++i )
  {
    std::cout << GradInputGot.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
#endif
  return 0;
}
