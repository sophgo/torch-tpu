#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef input_shape, int64_t dim, bool use_half )
{
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  if( use_half )
  {
    input_tpu = input_tpu.to( torch::kHalf );
  }
  auto output_cpu = torch::_softmax ( input_cpu, dim, false );
  auto output_tpu = torch::_softmax ( input_tpu, dim, false );
  if( use_half )
  {
    output_tpu = output_tpu.to( torch::kFloat );
  }
  auto output_exp = output_cpu;
  auto output_got = output_tpu.to ( torch::Device ( "cpu" ) );
  std::cout << "Comparing softmax:"
            << " dim = " << dim
            << " input shape = " << input_tpu.sizes()
            << " input dtype = " << input_tpu.dtype()
            << " output shape = " << output_tpu.sizes()
            << " output dtype = " << output_tpu.dtype()
            << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp );
}

int main ()
{
  const int batch = 16;
  test ( { batch,   12, 384, 384 } , -1 , true ); // 0
  test ( { batch,  256,  56,  56 } , -1 , true ); // 2
  test ( { batch,  128,  56,  56 } , -1 , true ); // 3
  test ( { batch,  128,  28,  28 } , -1 , true ); // 4
  test ( { batch,  512,  28,  28 } , -1 , true ); // 5
  test ( { batch,  256,  28,  28 } , -1 , true ); // 6
  return 0;
}
