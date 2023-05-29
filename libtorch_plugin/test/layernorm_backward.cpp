#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef     input_shape,
                          double              eps,
                          std::array<bool, 3> output_mask,
                          bool                use_half )
{
  auto grad_output_cpu = torch::randn ( input_shape ) - 0.5;
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto weight_cpu = torch::randn ( { input_shape[2] } ) - 0.5;
  auto bias_cpu = torch::randn ( { input_shape[2] } ) - 0.5;
  auto save_mean_cpu = torch::randn ( { input_shape[0], input_shape[1] } ) - 0.5;
  auto save_invstd_cpu = torch::randn ( { input_shape[0], input_shape[1] } );

  auto grad_output_tpu = grad_output_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto weight_tpu = weight_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto bias_tpu = bias_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto save_mean_tpu = save_mean_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto save_invstd_tpu = save_invstd_cpu.to ( tpu::TPUGetCurrentDevice() );

  if ( use_half )
  {
    grad_output_tpu = grad_output_tpu.to( torch::kHalf );
    input_tpu = input_tpu.to( torch::kHalf );
    weight_tpu = weight_tpu.to( torch::kHalf );
    bias_tpu = bias_tpu.to( torch::kHalf );
    save_mean_tpu = save_mean_tpu.to( torch::kHalf );
    save_invstd_tpu = save_invstd_tpu.to( torch::kHalf );
  }
  auto outputs_cpu = torch::native_layer_norm_backward (
                     grad_output_cpu,
                     input_cpu,
                     input_shape[2],
                     save_mean_cpu,
                     save_invstd_cpu,
                     c10::optional<at::Tensor> ( weight_cpu ),
                     c10::optional<at::Tensor> ( bias_cpu ),
                     output_mask );
  auto outputs_tpu = torch::native_layer_norm_backward (
                     grad_output_tpu,
                     input_tpu,
                     input_shape[2],
                     save_mean_tpu,
                     save_invstd_tpu,
                     c10::optional<at::Tensor> ( weight_tpu ),
                     c10::optional<at::Tensor> ( bias_tpu ),
                     output_mask );
  at::Tensor grad_input_got, grad_weight_got, grad_bias_got;
  at::Tensor grad_input_exp, grad_weight_exp, grad_bias_exp;

  if ( use_half )
  {
    if ( output_mask[0] == true )
    {
      grad_input_got = std::get<0> ( outputs_tpu ).to( torch::kFloat ).to ( torch::Device ( "cpu" ) );
      grad_input_exp = std::get<0> ( outputs_cpu );
    }
    if ( output_mask[1] == true )
    {
      grad_weight_got = std::get<1> ( outputs_tpu ).to( torch::kFloat ).to ( torch::Device ( "cpu" ) );
      grad_weight_exp = std::get<1> ( outputs_cpu );
    }
    if ( output_mask[2] == true )
    {
      grad_bias_got = std::get<2> ( outputs_tpu ).to( torch::kFloat ).to ( torch::Device ( "cpu" ) );
      grad_bias_exp = std::get<2> ( outputs_cpu );
    }
  }
  else
  {
    if ( output_mask[0] == true )
    {
      grad_input_got = std::get<0> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
      grad_input_exp = std::get<0> ( outputs_cpu );
    }
    if ( output_mask[1] == true )
    {
      grad_weight_got = std::get<1> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
      grad_weight_exp = std::get<1> ( outputs_cpu );
    }
    if ( output_mask[2] == true )
    {
      grad_bias_got = std::get<2> ( outputs_tpu ).to ( torch::Device ( "cpu" ) );
      grad_bias_exp = std::get<2> ( outputs_cpu );
    }
  }

  std::cout << "Comparing layernorm backward:"
            << " grad_output shape = " << grad_output_tpu.sizes()
            << " grad_output dtype = " << grad_output_tpu.dtype()
            << " input shape = " << input_tpu.sizes()
            << " input dtype = " << input_tpu.dtype()
            << " weight shape = " << weight_tpu.sizes()
            << " weight dtype = " << weight_tpu.dtype()
            << " bias shape = " << bias_tpu.sizes()
            << " bias dtype = " << bias_tpu.dtype()
            << " save_mean shape = " << save_mean_tpu.sizes()
            << " save_mean dtype = " << save_mean_tpu.dtype()
            << " save_invstd shape = " << save_invstd_tpu.sizes()
            << " save_invstd dtype = " << save_invstd_tpu.dtype()
            << " grad_input shape = " << grad_input_got.sizes()
            << " grad_input dtype = " << grad_input_got.dtype()
            << " grad_weight shape = " << grad_weight_got.sizes()
            << " grad_weight dtype = " << grad_weight_got.dtype()
            << " grad_bias shape = " << grad_bias_got.sizes()
            << " grad_bias dtype = " << grad_bias_got.dtype()
            << std::endl;
  if ( output_mask[0] == true )
  {
    std::cout << "Compare grad_input" << std::endl;
    tpu::TPUCompareResult ( grad_input_got, grad_input_exp );
  }
  if ( output_mask[1] == true )
  {
    std::cout << "Compare grad_weight" << std::endl;
    tpu::TPUCompareResult ( grad_weight_got, grad_weight_exp );
  }
  if ( output_mask[2] == true )
  {
    std::cout << "Compare grad_bias" << std::endl;
    tpu::TPUCompareResult ( grad_bias_got, grad_bias_exp );
  }
}

int main()
{
  const int batch = 16;
  test ( { batch,   64, 112 }, 1e-5, { true, true, true }, true ); // 0
  test ( { batch,   64,  56 }, 1e-5, { true, true, true }, true ); // 1
  test ( { batch,  256,  56 }, 1e-5, { true, true, true }, true ); // 2
  test ( { batch,  128,  56 }, 1e-5, { true, true, true }, true ); // 3
  test ( { batch,  128,  28 }, 1e-5, { true, true, true }, true ); // 4
  test ( { batch,  512,  28 }, 1e-5, { true, true, true }, true ); // 5
  test ( { batch,  256,  28 }, 1e-5, { true, true, true }, true ); // 6
  test ( { batch,  256,  14 }, 1e-5, { true, true, true }, true ); // 7
  test ( { batch, 1024,  14 }, 1e-5, { true, true, true }, true ); // 8
  test ( { batch,  512,  14 }, 1e-5, { true, true, true }, true ); // 9
  test ( { batch,  512,   7 }, 1e-5, { true, true, true }, true ); // 10
  test ( { batch, 2048,   7 }, 1e-5, { true, true, true }, true ); // 11
  return 0;
}
