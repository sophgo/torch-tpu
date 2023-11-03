#include <torch/torch.h>
#include <TPUTorchUtils.h>
#include <unistd.h>

static inline void test ( at::IntArrayRef input_shape, int64_t dim, bool use_half )
{
  torch::manual_seed ( 0 );
  auto output_cpu = torch::rand ( input_shape ) - 0.5;
  auto grad_output_cpu = torch::rand ( input_shape ) - 0.5;
  auto output_tpu = output_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto grad_output_tpu = grad_output_cpu.to ( tpu::TPUGetCurrentDevice() );
  if ( use_half )
  {
    output_tpu = output_tpu.to( torch::kHalf );
    grad_output_tpu = grad_output_tpu.to( torch::kHalf );
  }
  auto grad_input_cpu = torch::_softmax_backward_data ( grad_output_cpu, output_cpu, dim, grad_output_cpu.scalar_type() );
  auto grad_input_tpu = torch::_softmax_backward_data ( grad_output_tpu, output_tpu, dim, grad_output_tpu.scalar_type() );
  auto grad_input_exp = grad_input_cpu;
  auto grad_input_got = grad_input_tpu;
  if ( use_half )
  {
    grad_input_got = grad_input_tpu.to ( torch::kFloat ).to ( torch::Device ( "cpu" ) );
  }
  else
  {
    grad_input_got = grad_input_tpu.to ( torch::Device ( "cpu" ) );
  }
  std::cout << "\nComparing softmax backward:"
            << "\n grad_output shape = " << grad_output_tpu.sizes()
            << "\n grad_output dtype = " << grad_output_tpu.dtype()
            << "\n output shape = " << output_tpu.sizes()
            << "\n output dtype = " << output_tpu.dtype()
            << "\n grad_input shape = " << grad_input_got.sizes()
            << "\n grad_input dtype = " << grad_input_got.dtype()
            << std::endl;
  std::cout << "Compare grad_input" << std::endl;
  tpu::TPUCompareResult ( grad_input_got, grad_input_exp, 1e-3 );
}

int main ()
{
  int max_len = 100;
  int loop = 5;
  for (int i = 0; i < loop; i++)
  {
    srand(time(NULL));
    sleep(0.5);
    std::cout << "\ntest softmax backward case: " << i << std::endl;
    int n = rand() % max_len + 1;
    int c = rand() % max_len + 1;
    int h = rand() % max_len + 1;
    int w = rand() % max_len + 1;
    std::cout << "  n: " << n << "  c: " << c;
    std::cout << "  h: " << h << "  w: " << w << std::endl;
    test ( { n, c, h, w }, -1, true );
    test ( { n, c, h, w }, -1, false );
  }
  return 0;
}
