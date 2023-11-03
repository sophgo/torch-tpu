#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef mat1_shape, at::IntArrayRef mat2_shape )
{
  auto mat1_cpu = torch::randn ( mat1_shape ) - 0.5;
  auto mat2_cpu = torch::randn ( mat2_shape ) - 0.5;
  auto mat1_tpu = mat1_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto mat2_tpu = mat2_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto output_cpu = torch::bmm ( mat1_cpu, mat2_cpu );
  auto output_tpu = torch::bmm ( mat1_tpu, mat2_tpu );
  auto output_exp = output_cpu;
  auto output_got = output_tpu.to ( torch::Device ( "cpu" ) );
  std::cout << "Comparing bmm:"
            << " mat1 shape = " << mat1_tpu.sizes()
            << " mat1 dtype = " << mat1_tpu.dtype()
            << " mat2 shape = " << mat2_tpu.sizes()
            << " mat2 dtype = " << mat2_tpu.dtype()
            << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp );
}

int main ()
{
  const int batch = 12;
  test ( { batch, 384, 384 }, { batch, 384, 64 } ); // 0
  return 0;
}
