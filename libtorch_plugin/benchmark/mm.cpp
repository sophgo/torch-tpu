#include <torch/torch.h>
#include <TPUTorchUtils.h>

static int loops = 0;

static inline void benchmark ( at::IntArrayRef mat1_shape,
                               at::IntArrayRef mat2_shape )
{
  auto mat1_cpu = torch::randn ( mat1_shape ) - 0.5;
  auto mat2_cpu = torch::randn ( mat2_shape ) - 0.5;
  auto mat1_tpu = mat1_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto mat2_tpu = mat2_cpu.to ( tpu::TPUGetCurrentDevice() );
  at::Tensor output_tpu;
  tpu::Timer timer;
  timer.Start();
  for ( int i = 0; i < loops; ++i )
  {
    output_tpu = torch::mm ( mat1_tpu, mat2_tpu );
  }
  unsigned long elapsed_us_per_loop = ( double ) timer.ElapsedUS() / loops;
  std::cout << "Benchmarking matmul:"
            << " Elapsed time = " << elapsed_us_per_loop << "us"
            << " mat1 shape = " << mat1_tpu.sizes()
            << " mat1 dtype = " << mat1_tpu.dtype()
            << " mat2 shape = " << mat2_tpu.sizes()
            << " mat2 dtype = " << mat2_tpu.dtype()
            << " output shape = " << output_tpu.sizes()
            << " output dtype = " << output_tpu.dtype()
            << std::endl;
}

int main()
{
  const int batch = 32;
  loops = 500;
  // mat1 shape, mat2 shape
  // Bert base forward & backward
  benchmark ( { batch * 384, 768 },  { 768, 768 } );
  benchmark ( { batch * 384, 768 },  { 768, 3072 } );
  benchmark ( { batch * 384, 3072 },  { 3072, 768 } );
  benchmark ( { batch * 384, 768 },  { 768, 2 } );
  benchmark ( { 768, batch * 384 },  { batch * 384, 2 } );
  benchmark ( { batch * 384, 2},  { 2, 768 } );
  benchmark ( { 3072, batch * 384 },  { batch * 384, 768 } );
  benchmark ( { 768, batch * 384 },  { batch * 384, 3072 } );
  benchmark ( { 768, batch * 384 },  { batch * 384, 768 } );
  return 0;
}
