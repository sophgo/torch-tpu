#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef input_shape,
                          double          momentum,
                          double          eps,
                          bool            track_running_status )
{
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto num_features = input_shape[1];
  auto weight_cpu = torch::randn ( { num_features } ) - 0.5;
  auto bias_cpu = torch::randn ( { num_features } ) - 0.5;
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto weight_tpu = weight_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto bias_tpu = bias_cpu.to ( tpu::TPUGetCurrentDevice() );
  at::Tensor running_mean_cpu, running_var_cpu;
  at::Tensor running_mean_tpu, running_var_tpu;
  if ( track_running_status )
  {
    running_mean_cpu = torch::randn ( { num_features } ) - 0.5;
    running_var_cpu = torch::randn ( { num_features } );
    running_mean_tpu = running_mean_cpu.to ( tpu::TPUGetCurrentDevice() );
    running_var_tpu = running_var_cpu.to ( tpu::TPUGetCurrentDevice() );
  }
  auto outputs_cpu = native_batch_norm (
                     input_cpu,
                     c10::optional<at::Tensor> ( weight_cpu ),
                     c10::optional<at::Tensor> ( bias_cpu ),
                     c10::optional<at::Tensor> ( running_mean_cpu ),
                     c10::optional<at::Tensor> ( running_var_cpu ),
                     true,
                     momentum,
                     eps );
  auto outputs_tpu = native_batch_norm (
                     input_tpu,
                     c10::optional<at::Tensor> ( weight_tpu ),
                     c10::optional<at::Tensor> ( bias_tpu ),
                     c10::optional<at::Tensor> ( running_mean_tpu ),
                     c10::optional<at::Tensor> ( running_var_tpu ),
                     true,
                     momentum,
                     eps );
  auto output_got = std::get<0> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
  auto save_mean_got = std::get<1> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
  auto save_invstd_got = std::get<2> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
  auto output_exp = std::get<0> ( outputs_cpu );
  auto save_mean_exp = std::get<1> ( outputs_cpu );
  auto save_invstd_exp = std::get<2> ( outputs_cpu );
  std::cout << "Comparing batchnorm:"
            << " input shape = " << input_cpu.sizes()
            << " input dtype = " << input_cpu.dtype()
            << " weight shape = " << weight_tpu.sizes()
            << " weight dtype = " << weight_tpu.dtype()
            << " bias shape = " << bias_tpu.sizes()
            << " bias dtype = " << bias_tpu.dtype()
            << " running_mean shape = " << running_mean_tpu.sizes()
            << " running_mean dtype = " << running_mean_tpu.dtype()
            << " running_var shape = " << running_var_tpu.sizes()
            << " running_var dtype = " << running_var_tpu.dtype()
            << " output shape = " << output_got.sizes()
            << " output dtype = " << output_got.dtype()
            << " save_mean shape = " << save_mean_got.sizes()
            << " save_mean dtype = " << save_mean_got.dtype()
            << " save_invstd shape = " << save_invstd_got.sizes()
            << " save_invstd dtype = " << save_invstd_got.dtype()
            << std::endl;
  std::cout << "Compare output" << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp );
  std::cout << "Compare save_mean" << std::endl;
  tpu::TPUCompareResult ( save_mean_got, save_mean_exp );
  std::cout << "Compare save_invstd" << std::endl;
  tpu::TPUCompareResult ( save_invstd_got, save_invstd_exp );
  if ( track_running_status )
  {
    auto running_mean_got = running_mean_tpu.to ( torch::Device ( "cpu" ) );
    auto running_mean_exp = running_mean_cpu;
    std::cout << "Compare running_mean" << std::endl;
    tpu::TPUCompareResult ( running_mean_got, running_mean_exp );
    auto running_var_got = running_var_tpu.to ( torch::Device ( "cpu" ) );
    auto running_var_exp = running_var_cpu;
    std::cout << "Compare running_var" << std::endl;
    tpu::TPUCompareResult ( running_var_got, running_var_exp );
  }
}

int main()
{
  const int batch = 64;
  test ( { batch,   64, 112, 112 }, 0.1, 1e-5, true );
  test ( { batch,   64,  56,  56 }, 0.1, 1e-5, true );
  test ( { batch,  256,  56,  56 }, 0.1, 1e-5, true );
  test ( { batch,  128,  56,  56 }, 0.1, 1e-5, true );
  test ( { batch,  128,  28,  28 }, 0.1, 1e-5, true );
  test ( { batch,  512,  28,  28 }, 0.1, 1e-5, true );
  test ( { batch,  256,  28,  28 }, 0.1, 1e-5, true );
  test ( { batch,  256,  14,  14 }, 0.1, 1e-5, true );
  test ( { batch, 1024,  14,  14 }, 0.1, 1e-5, true );
  test ( { batch,  512,  14,  14 }, 0.1, 1e-5, true );
  test ( { batch,  512,   7,   7 }, 0.1, 1e-5, true );
  test ( { batch, 2048,   7,   7 }, 0.1, 1e-5, true );
  return 0;
}
