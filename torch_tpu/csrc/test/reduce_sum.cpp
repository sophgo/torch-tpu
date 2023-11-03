#include <torch/torch.h>
#include <TPUTorchUtils.h>
#include <unistd.h>

static inline void test ( at::IntArrayRef input_shape, at::IntArrayRef dim_opt, bool keepdim, bool use_half )
{
  auto input_cpu = torch::randn ( input_shape );
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  if ( use_half )
  {
    input_tpu = input_tpu.to( torch::kHalf );
  }
  auto out_cpu = sum ( input_cpu, dim_opt, keepdim );
  auto out_tpu = sum ( input_tpu, dim_opt, keepdim );
  auto output_exp = out_cpu;
  auto output_got = out_tpu;
  if ( use_half )
  {
    output_got = out_tpu.to( torch::kFloat ).to ( torch::Device ( "cpu" ) );
  }
  else
  {
    output_got = out_tpu.to ( torch::Device ( "cpu" ) );
  }
  std::cout << "Comparing reduce sum:"
            << "\ninput shape = " << input_tpu.sizes()
            << "\ninput dtype = " << input_tpu.dtype()
            << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp, 1e-2 );
}

int main ()
{
  int max_len = 1000;
  int loop = 5;
  for (int i = 0; i < loop; i++)
  {
    srand(time(NULL));
    sleep(1.0);
    std::cout << "\ntest reduce case: " << i << std::endl;
    int c = rand() % max_len + 1;
    int w = rand() % max_len + 1;
    test ( { c, w }, { 0 }, true, true );
  }
  return 0;
}