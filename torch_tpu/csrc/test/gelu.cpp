#include <torch/torch.h>
#include <TPUTorchUtils.h>
#include <unistd.h>

static inline void test ( at::IntArrayRef input_shape , bool use_half)
{
  torch::manual_seed ( 0 );
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  if ( use_half )
  {
    input_tpu = input_tpu.to ( torch::kHalf );
  }
  auto output_cpu = torch::gelu ( input_cpu );
  auto output_tpu = torch::gelu ( input_tpu );
  auto output_exp = output_cpu;
  if ( use_half )
  {
    output_tpu = output_tpu.to ( torch::kFloat );
  }
  auto output_got = output_tpu.to ( torch::Device ( "cpu" ) );
  std::cout << "Comparing gelu:"
            << " input shape = " << input_tpu.sizes()
            << " input dtype = " << input_tpu.dtype()
            << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp, 1e-2 );
}

int main ()
{
  int max_len = 1000000;
  int loop = 10;
  for (int i = 0; i < loop; i++)
  {
    srand(time(NULL));
    sleep(1.0);
    std::cout << "\ntest gelu case: " << i << std::endl;
    int len = rand() % max_len + 1;
    test ( { len } , true );
  }
  return 0;
}
