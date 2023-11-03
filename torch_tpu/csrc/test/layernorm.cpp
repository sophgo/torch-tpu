#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef     input_shape,
                          at::IntArrayRef     normalized_shape,
                          double              eps,
                          bool                use_half )
{
  auto input_cpu = torch::randn ( input_shape ) - 0.5 ;
  auto weight_cpu = torch::randn ( normalized_shape ) - 0.5 ;
  auto bias_cpu = torch::randn ( normalized_shape ) - 0.5 ;
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto weight_tpu = weight_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto bias_tpu = bias_cpu.to ( tpu::TPUGetCurrentDevice() );

  if ( use_half )
  {
    input_tpu = input_tpu.to( torch::kHalf );
    weight_tpu = weight_tpu.to( torch::kHalf );
    bias_tpu = bias_tpu.to( torch::kHalf );
  }
  auto outputs_cpu = torch::native_layer_norm (
                     input_cpu,
                     normalized_shape,
                     c10::optional<at::Tensor> ( weight_cpu ),
                     c10::optional<at::Tensor> ( bias_cpu ),
                     eps );
  auto outputs_tpu = torch::native_layer_norm (
                     input_tpu,
                     normalized_shape,
                     c10::optional<at::Tensor> ( weight_tpu ),
                     c10::optional<at::Tensor> ( bias_tpu ),
                     eps );
  at::Tensor output_got, mean_got, rstd_got;
  at::Tensor output_exp, mean_exp, rstd_exp;

  if ( use_half )
  {
    output_got = std::get<0> ( outputs_tpu ).to( torch::kFloat ).to ( torch::Device ( "cpu" ) );
    mean_got = std::get<1> ( outputs_tpu ).to( torch::kFloat ).to ( torch::Device ( "cpu" ) );
    rstd_got = std::get<2> ( outputs_tpu ).to( torch::kFloat ).to ( torch::Device ( "cpu" ) );
  }
  else
  {
    output_got = std::get<0> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
    mean_got = std::get<1> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
    rstd_got = std::get<2> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
  }

  output_exp = std::get<0> ( outputs_cpu );
  mean_exp = std::get<1> ( outputs_cpu );
  rstd_exp = std::get<2> ( outputs_cpu );

  std::cout << "Comparing layernorm:"
            << "\ninput shape = " << input_tpu.sizes()
            << "\ninput dtype = " << input_tpu.dtype()
            << "\nweight shape = " << weight_tpu.sizes()
            << "\nweight dtype = " << weight_tpu.dtype()
            << "\nbias shape = " << bias_tpu.sizes()
            << "\nbias dtype = " << bias_tpu.dtype()
            << "\noutput shape = " << output_got.sizes()
            << "\noutput dtype = " << output_got.dtype()
            << "\nmean shape = " << mean_got.sizes()
            << "\nmean dtype = " << mean_got.dtype()
            << "\nrstd shape = " << rstd_got.sizes()
            << "\nrstd dtype = " << rstd_got.dtype()
            << std::endl;

  std::cout << "Compare output" << std::endl;
  tpu::TPUCompareResult ( output_got, output_exp );
  std::cout << "Compare mean" << std::endl;
  tpu::TPUCompareResult ( mean_got, mean_exp );
  std::cout << "Compare rstd" << std::endl;
  tpu::TPUCompareResult ( rstd_got, rstd_exp );
}

int main()
{
  const int batch = 32;
  test ( { batch,   64, 68 }, { 68 }, 1e-5, true ); // 1
  test ( { batch,  256, 768 }, { 768 }, 1e-5, true ); // 2
  test ( { batch,  128, 768 }, { 768 }, 1e-5, true ); // 3
  test ( { batch,  128, 3072 }, { 3072 }, 1e-5, true ); // 4
  test ( { batch,  512, 3072 }, { 3072 }, 1e-5, true ); // 5
  test ( { batch,  256, 3072 }, { 3072 }, 1e-5, true ); // 6
  test ( { batch,  256,  768 }, {  768 }, 1e-5, true ); // 7
  test ( { batch, 1024,  768 }, {  768 }, 1e-5, true ); // 8
  return 0;
}
