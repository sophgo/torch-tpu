#include <torch/torch.h>
#include <TPUTorchUtils.h>
#include <unistd.h>

static inline void test ( at::IntArrayRef input_shape, at::IntArrayRef dim_opt, bool keepdim )
{
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto out_cpu = sum ( input_cpu, dim_opt, keepdim );
  auto out_tpu = sum ( input_tpu, dim_opt, keepdim );
  auto output_exp = out_cpu;
  auto output_got = out_tpu.to ( torch::Device ( "cpu" ) );
  std::cout << "Comparing reduce sum "
            << " input shape = " << input_tpu.sizes()
            << " input dtype = " << input_tpu.dtype()
            << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp );
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
    test ( { c, w }, { 0 }, true );
  }
  return 0;
}