#include <torch/torch.h>
#include <ATen/native/ConvUtils.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef     grad_output_shape,
                          at::IntArrayRef     input_shape,
                          at::IntArrayRef     weight_shape,
                          at::IntArrayRef     stride,
                          at::IntArrayRef     padding,
                          at::IntArrayRef     dilation,
                          int64_t             groups,
                          std::array<bool, 3> output_mask )
{
  auto grad_output_cpu = torch::randn ( grad_output_shape ) - 0.5;
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto weight_cpu = torch::randn ( weight_shape ) - 0.5;
  auto grad_output_tpu = grad_output_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto weight_tpu = weight_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto outputs_cpu = torch::convolution_backward (
                     grad_output_cpu,
                     input_cpu,
                     weight_cpu,
                     at::OptionalIntArrayRef ( { weight_shape[0] } ),
                     stride,
                     padding,
                     dilation,
                     false,
                     at::IntArrayRef ( { 0, 0 } ),
                     groups,
                     output_mask );
  auto outputs_tpu = torch::convolution_backward (
                     grad_output_tpu,
                     input_tpu,
                     weight_tpu,
                     at::OptionalIntArrayRef ( { weight_shape[0] } ),
                     stride,
                     padding,
                     dilation,
                     false,
                     at::IntArrayRef ( { 0, 0 } ),
                     groups,
                     output_mask );
  at::Tensor grad_input_got, grad_weight_got, grad_bias_got;
  at::Tensor grad_input_exp, grad_weight_exp, grad_bias_exp;
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
  std::cout << "Comparing convolution backward:"
            << " grad_output shape = " << grad_output_tpu.sizes()
            << " grad_output dtype = " << grad_output_tpu.dtype()
            << " input shape = " << input_tpu.sizes()
            << " input dtype = " << input_tpu.dtype()
            << " weight shape = " << weight_tpu.sizes()
            << " weight dtype = " << weight_tpu.dtype()
            << " grad_input shape = " << grad_input_got.sizes()
            << " grad_input dtype = " << grad_input_got.dtype()
            << " grad_weight shape = " << grad_weight_got.sizes()
            << " grad_weight dtype = " << grad_weight_got.dtype()
            << " grad_bias shape = " << grad_bias_got.sizes()
            << " grad_bias dtype = " << grad_bias_got.dtype()
            << " stride = " << stride
            << " padding = " << padding
            << " dilation = " << dilation
            << " groups = " << groups
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
    std::cout << "Compare grad_bias\n";
    tpu::TPUCompareResult ( grad_bias_got, grad_bias_exp );
  }
}

int main()
{
  const int batch = 64;
  // grad output shape, input shape, weight shape, stride, padding, dilation, groups
  test ( { batch, 64, 112, 112 }, { batch, 3, 224, 224 }, { 64, 3, 7, 7 }, { 2, 2 }, { 3, 3 }, { 1, 1 }, 1, { true, true, false } );
  return 0;
}
