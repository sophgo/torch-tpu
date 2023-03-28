#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef input_shape, int64_t dim )
{
  auto output_cpu = torch::randn ( input_shape ) - 0.5;
  auto output_tpu = output_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto grad_output_cpu = torch::randn ( input_shape ) - 0.5;
  auto grad_output_tpu = grad_output_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto grad_input_cpu = torch::_softmax_backward_data ( grad_output_cpu, output_cpu, dim, grad_output_tpu.dtype() );
  auto grad_input_tpu = torch::_softmax_backward_data ( grad_output_tpu, output_tpu, dim, grad_output_tpu.dtype() );
  auto grad_input_exp = grad_input_cpu;
  auto grad_input_got = grad_input_tpu.to ( torch::Device ( "cpu" ) );
  std::cout << "Comparing softmax backward:"
            << " grad_output shape = " << grad_output_tpu.sizes()
            << " grad_output dtype = " << grad_output_tpu.dtype()
            << " output shape = " << output_tpu.sizes()
            << " output dtype = " << output_tpu.dtype()
            << " grad_input shape = " << grad_input_got.sizes()
            << " grad_input dtype = " << grad_input_got.dtype()
            << std::endl;
  std::cout << "Compare grad_input" << std::endl;
  tpu::TPUCompareResult ( grad_input_got, grad_input_exp );
}

int main ()
{
  const int batch = 16;
  test ( { batch, 12, 384, 384 }, -1 ); // 0
  return 0;
}