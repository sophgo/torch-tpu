#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>
#include <TPUTorchUtils.h>

int main()
{
  int Batch = 32;
  int Seq_len = 256;
  const int Hidden_size = 768;
  tpu::SetMatrixMultiplyForwardAccuracy ( tpu::ALGORITHM_ACCURACY_FP32 );
  tpu::SetMatrixMultiplyBackwardAccuracy ( tpu::ALGORITHM_ACCURACY_FP32 );
  auto TPU = tpu::TPUGetCurrentDevice();
  auto CPU = torch::Device ( "cpu" );
  std::string ModelPath = "../mygpt2block.pt";
  auto GptCPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  auto GptTPU = std::make_shared<tpu::TorchscriptModule> ( ModelPath );
  GptCPU->train();
  GptTPU->train();
  tpu::MoveModuleToTPUDevice ( *GptTPU );
  // https://pytorch.org/cppdocs/api/structtorch_1_1optim_1_1_adam_options.html#exhale-struct-structtorch-1-1optim-1-1-adam-options
  torch::optim::Adam OptimizerCPU { GptCPU->parameters(), torch::optim::AdamOptions () };
  torch::optim::Adam optimizerTPU { GptTPU->parameters(), torch::optim::AdamOptions () };
  OptimizerCPU.zero_grad();
  optimizerTPU.zero_grad();
  torch::manual_seed ( 0 );
  tpu::Timer timer;
  // Forward
  auto Input0CPU = torch::randn ( { Batch, Seq_len, Hidden_size } );
  auto Input1CPU = torch::tril ( torch::ones ( { Seq_len, Seq_len } ) );
  auto Input0TPU = Input0CPU.to ( TPU );
  auto Input1TPU = Input1CPU.to ( TPU );
  timer.Start();
  auto OutputsCPU = GptCPU->forward ( { Input0CPU, Input1CPU } );
  std::cout << "Gpt2Block Forward CPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  tpu::OpTimer::Instance().Clear();
  timer.Start();
  auto OutputsTPU = GptTPU->forward ( { Input0TPU, Input1TPU } );
  std::cout << "Gpt2Block Forward TPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  tpu::OpTimer::Instance().Dump();
  auto EXP = OutputsCPU[0].contiguous();
  auto GOT = TENSOR_TO_CPU ( OutputsTPU[0] );
  tpu::TPUCompareResult ( GOT, EXP );
  // Backward
  auto BackwardInput0CPU = torch::ones ( OutputsCPU[0].sizes() );
  auto BackwardInput0TPU = torch::ones ( OutputsTPU[0].sizes() ).to ( TPU );
  tpu::OpTimer::Instance().Clear();
  timer.Start();
  OutputsCPU[0].backward ( BackwardInput0CPU );
  std::cout << "Gpt2Block Backward CPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  timer.Start();
  OutputsTPU[0].backward ( BackwardInput0TPU );
  std::cout << "Gpt2Block Backward TPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  tpu::OpTimer::Instance().Dump();
  // // Optimizer Step
  // tpu::OpTimer::Instance().Clear();
  // timer.Start();
  // OptimizerCPU.step();
  // std::cout << "Gpt2Block Optimizer Step CPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  // timer.Start();
  // optimizerTPU.step();
  // std::cout << "Gpt2Block Optimizer Step TPU Elapsed time = " << timer.ElapsedUS() << "us" << std::endl;
  std::vector<at::Tensor> ParamCPU = tpu::GetNamedParameters ( *GptCPU );
  std::vector<at::Tensor> ParamTPU = tpu::GetNamedParameters ( *GptTPU );
  TORCH_CHECK ( ParamCPU.size() == ParamTPU.size() );
  std::cout << "Gpt2Block (Batch = " << Batch << ", Seq_len = "<< Seq_len << ", Param_size = " << ParamCPU.size() << ")" << std::endl;
  tpu::TPUCompareResult ( ParamTPU[0].grad().cpu(), ParamCPU[0].grad() );
  return 0;
}