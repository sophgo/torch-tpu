#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef input_shape , bool use_half)
{
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
  tpu::TPUCompareResult ( output_got, output_exp );
}

int main ()
{
  const int batch = 64;
  test ( { batch,   64, 112, 112 } , true ); // 0
  test ( { batch,   64,  56,  56 } , true ); // 1
  test ( { batch,  256,  56,  56 } , true ); // 2
  test ( { batch,  128,  56,  56 } , true ); // 3
  test ( { batch,  128,  28,  28 } , true ); // 4
  test ( { batch,  512,  28,  28 } , true ); // 5
  test ( { batch,  256,  28,  28 } , true ); // 6
  test ( { batch,  256,  14,  14 } , true ); // 7
  test ( { batch, 1024,  14,  14 } , true ); // 8
  test ( { batch,  512,  14,  14 } , true ); // 9
  test ( { batch,  512,   7,   7 } , true ); // 10
  test ( { batch, 1024,   1, 768 } , true ); // 11
  return 0;
}
