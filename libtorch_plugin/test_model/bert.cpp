#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>
#include <TPUTorchUtils.h>

int main()
{
  int Batch = 16;
  auto TPU = tpu::TPUGetCurrentDevice();
  auto CPU = torch::Device ( "cpu" );
  std::string ModelPath = "../bert_base_traced-2.11.0.pt";
  auto BertCPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  auto BertTPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  BertCPU->train();
  BertTPU->train();
  tpu::MoveModuleToTPUDevice ( *BertTPU );
  torch::manual_seed ( 0 );
  auto Input0CPU = torch::randint ( 0, 28990, { Batch, 384 }, torch::kInt );
  auto Input1CPU = torch::randint ( 0, 2, { Batch, 384 }, torch::kInt );
  auto Input2CPU = torch::randint ( 0, 2, { Batch, 384 }, torch::kInt );
  auto Input0TPU = Input0CPU.to ( TPU );
  auto Input1TPU = Input1CPU.to ( TPU );
  auto Input2TPU = Input2CPU.to ( TPU );
  tpu::Timer timer;
  timer.Start();
  auto OutputsCPU = BertCPU->forward ( { Input0CPU, Input1CPU, Input2CPU } );
  std::cout << "BertBase(Batch = " << Batch << ") Forward CPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  tpu::OpTimer::Instance().Clear();
  timer.Start();
  auto OutputsTPU = BertTPU->forward ( { Input0TPU, Input1TPU, Input2TPU } );
  std::cout << "BertBase(Batch = " << Batch << ") Forward TPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  tpu::OpTimer::Instance().Dump();;
  auto EXP0 = OutputsCPU[0].contiguous();
  auto EXP1 = OutputsCPU[1].contiguous();
  auto GOT0 = TENSOR_TO_CPU ( OutputsTPU[0] );
  auto GOT1 = TENSOR_TO_CPU ( OutputsTPU[1] );
  tpu::TPUCompareResult ( GOT0, EXP0 );
  tpu::TPUCompareResult ( GOT1, EXP1 );
  auto BackwardInput0CPU = torch::ones ( OutputsTPU[0].sizes() );
  auto BackwardInput0TPU = torch::ones ( OutputsTPU[0].sizes() ).to ( TPU );
  tpu::OpTimer::Instance().Clear();
  timer.Start();
  OutputsCPU[0].backward ( BackwardInput0CPU );
  std::cout << "BertBase(Batch = " << Batch << ") Backward CPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  timer.Start();
  OutputsTPU[0].backward ( BackwardInput0TPU );
  std::cout << "BertBase(Batch = " << Batch << ") Backward TPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  //auto GradInputGot = InputTPU.grad().to ( CPU );
  //auto GradInputExp = InputCPU.grad();
//  tpu::TPUCompareResult ( GradInputGot, GradInputExp );
  tpu::OpTimer::Instance().Dump();
  return 0;
}
