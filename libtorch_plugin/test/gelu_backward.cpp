#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef grad_output_shape )
{
  auto input_cpu = torch::randn ( grad_output_shape ) - 0.5;
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto grad_output_cpu = torch::randn ( grad_output_shape ) - 0.5;
  auto grad_output_tpu = grad_output_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto output_cpu = torch::gelu_backward ( grad_output_cpu, input_cpu);
  auto output_tpu = torch::gelu_backward ( grad_output_tpu, input_tpu);
  auto output_exp = output_cpu;
  auto output_got = output_tpu.to ( torch::Device ( "cpu" ) );
  std::cout << "Comparing gelu backward:"
            << " grad_output shape = " << grad_output_tpu.sizes()
            << " grad_output dtype = " << grad_output_tpu.dtype()
            << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp );
}

int main ()
{
  const int batch = 4;
  test ( { batch,   64, 112, 112 } ); // 0
  test ( { batch,   64,  56,  56 } ); // 1
  test ( { batch,  256,  56,  56 } ); // 2
  test ( { batch,  128,  56,  56 } ); // 3
  test ( { batch,  128,  28,  28 } ); // 4
  test ( { batch,  512,  28,  28 } ); // 5
  test ( { batch,  256,  28,  28 } ); // 6
  test ( { batch,  256,  14,  14 } ); // 7
  test ( { batch, 1024,  14,  14 } ); // 8
  test ( { batch,  512,  14,  14 } ); // 9
  test ( { batch,  512,   7,   7 } ); // 10
  test ( { batch, 2048,   7,   7 } ); // 11
  test ( {     1,   32,   1,3072 } ); // 12
  return 0;
}
