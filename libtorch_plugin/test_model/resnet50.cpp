#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>
#include <TPUTorchUtils.h>

int main()
{
  int Batch = 1;
  tpu::SetConvolutionForwardAccuracy ( tpu::ALGORITHM_ACCURACY_FP32 );
  tpu::SetConvolutionBackwardAccuracy ( tpu::ALGORITHM_ACCURACY_FP32 );
  auto TPU = tpu::TPUGetCurrentDevice();
  auto CPU = torch::Device ( "cpu" );
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
  tpu::Timer timer;
  timer.Start();
  tpu::OpTimer::Instance().Clear();
  OutputCPU = Resnet50CPU->forward ( InputCPU );
  std::cout << "Resnet50(Batch = " << Batch << ") Forward CPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  timer.Start();
  OutputTPU = Resnet50TPU->forward ( InputTPU );
  std::cout << "Resnet50(Batch = " << Batch << ") Forward TPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  auto OutputGot = OutputTPU.to ( CPU );
  auto OutputExp = OutputCPU;
  tpu::TPUCompareResult ( OutputGot, OutputExp );
  auto BackwardInputCPU = torch::ones ( OutputTPU.sizes() );
  auto BackwardInputTPU = torch::ones ( OutputTPU.sizes() ).to ( TPU );
  timer.Start();
  OutputCPU.backward ( BackwardInputCPU );
  std::cout << "Resnet50(Batch = " << Batch << ") Backward CPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  timer.Start();
  OutputTPU.backward ( BackwardInputTPU );
  std::cout << "Resnet50(Batch = " << Batch << ") Backward TPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  tpu::OpTimer::Instance().Dump();
  auto GradInputGot = InputTPU.grad().to ( CPU );
  auto GradInputExp = InputCPU.grad();
  tpu::TPUCompareResult ( GradInputGot, GradInputExp );
  return 0;
}
