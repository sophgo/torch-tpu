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
  tpu::OpTimer::Instance().Clear();
  tpu::Timer timer;
  timer.Start();
  auto OutputsTPU = BertTPU->forward ( { Input0TPU, Input1TPU, Input2TPU } );
  std::cout << "BertBase(Batch = " << Batch << ") Forward TPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  timer.Start();
  auto OutputsCPU = BertCPU->forward ( { Input0CPU, Input1CPU, Input2CPU } );
  std::cout << "BertBase(Batch = " << Batch << ") Forward CPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  auto EXP0 = OutputsCPU[0].contiguous();
  auto EXP1 = OutputsCPU[1].contiguous();
  auto GOT0 = TENSOR_TO_CPU ( OutputsTPU[0] );
  auto GOT1 = TENSOR_TO_CPU ( OutputsTPU[1] );
#if 0
  for ( auto i = 0; i < 10; ++i )
  {
    std::cout << EXP0.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
  for ( auto i = 0; i < 10; ++i )
  {
    std::cout << GOT0.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
  for ( auto i = 0; i < 10; ++i )
  {
    std::cout << EXP1.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
  for ( auto i = 0; i < 10; ++i )
  {
    std::cout << GOT1.data_ptr<float>() [i] << " ";
  }
  std::cout << std::endl;
#endif
  tpu::TPUCompareResult ( GOT0, EXP0 );
  tpu::TPUCompareResult ( GOT1, EXP1 );
#if 0
  OutputCPU = Resnet50CPU->forward ( InputCPU );
  OutputTPU = Resnet50TPU->forward ( InputTPU );
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
  auto GradInputGot = InputTPU.grad().to ( CPU );
  auto GradInputExp = InputCPU.grad();
  tpu::TPUCompareResult ( GradInputGot, GradInputExp );
#endif
  tpu::OpTimer::Instance().Dump();
  return 0;
}
