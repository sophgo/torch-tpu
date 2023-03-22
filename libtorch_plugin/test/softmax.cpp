#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef input_shape, int64_t dim )
{
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto output_cpu = torch::_softmax ( input_cpu, dim, false );
  auto output_tpu = torch::_softmax ( input_tpu, dim, false );
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
  test ( { batch,   12, 384, 384 }, -1 ); // 0
  return 0;
}
