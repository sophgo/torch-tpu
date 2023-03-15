#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>
#include <TPUTorchUtils.h>

int main()
{
  int Batch = 1;
  auto TPU = tpu::TPUGetCurrentDevice();
  auto CPU = torch::Device ( "cpu" );
  std::string ModelPath = "../bert_base_traced-2.11.0.pt";
  auto BertCPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  auto BertTPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  BertCPU->train();
  BertTPU->train();
  tpu::MoveModuleToTPUDevice ( *BertTPU );
  auto Input0CPU = torch::randint ( 0, 28990, { Batch, 384 }, torch::kInt );
  auto Input1CPU = torch::randint ( 0, 2, { Batch, 384 }, torch::kInt );
  auto Input2CPU = torch::randint ( 0, 2, { Batch, 384 }, torch::kInt );
  auto Input0TPU = Input0CPU.to ( TPU );
  auto Input1TPU = Input1CPU.to ( TPU );
  auto Input2TPU = Input2CPU.to ( TPU );
  //auto OutputsCPU = BertCPU->forward ( { Input0CPU, Input1CPU, Input2CPU } );
  auto OutputsTPU = BertTPU->forward ( { Input0TPU, Input1TPU, Input2TPU } );
#if 0
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
#endif
  return 0;
}
