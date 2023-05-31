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
  int max_len = 1000000;
  int loop = 10;
  for (int i = 0; i < loop; i++)
  {
    srand(time(NULL));
    std::cout << "\ntest gelu backward case: " << i << std::endl;
    int len = rand() % max_len + 1;
    test ( { len } );
  }
  return 0;
}
