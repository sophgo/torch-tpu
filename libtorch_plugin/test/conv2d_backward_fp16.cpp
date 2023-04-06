#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef grad_output_shape,
                          at::IntArrayRef input_shape,
                          at::IntArrayRef weight_shape,
                          at::IntArrayRef stride,
                          at::IntArrayRef padding,
                          at::IntArrayRef dilation,
                          int64_t groups,
                          std::array<bool, 3> output_mask )
{
  auto grad_output_cpu = torch::randn ( grad_output_shape, c10::kHalf ) - 0.5;
  auto input_cpu = torch::randn ( input_shape, c10::kHalf ) - 0.5;
  auto weight_cpu = torch::randn ( weight_shape, c10::kHalf ) - 0.5;
  auto grad_output_tpu = grad_output_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto weight_tpu = weight_cpu.to ( tpu::TPUGetCurrentDevice() );
  grad_output_cpu = grad_output_cpu.to ( c10::kFloat );
  input_cpu = input_cpu.to ( c10::kFloat );
  weight_cpu = weight_cpu.to ( c10::kFloat );
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
    tpu::TPUCompareResult ( grad_input_got.to ( c10::kFloat ), grad_input_exp.to ( c10::kFloat ) );
  }
  if ( output_mask[1] == true )
  {
    std::cout << "Compare grad_weight" << std::endl;
    tpu::TPUCompareResult ( grad_weight_got.to ( c10::kFloat ), grad_weight_exp.to ( c10::kFloat ) );
  }
  if ( output_mask[2] == true )
  {
    std::cout << "Compare grad_bias" << std::endl;
    tpu::TPUCompareResult ( grad_bias_got.to ( c10::kFloat ), grad_bias_exp.to ( c10::kFloat ) );
  }
}

int main()
{
  const int batch = 64;
  const bool has_bias = true;
  // grad output shape, input shape, weight shape, stride, padding, dilation, groups
  test ( { batch,   64, 112, 112 }, { batch,    3, 224, 224 }, {   64,    3, 7, 7 }, { 2, 2 }, { 3, 3 }, { 1, 1 }, 1, { true, true, has_bias } ); // 0
  test ( { batch,   64,  56,  56 }, { batch,   64,  56,  56 }, {   64,   64, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 1
  test ( { batch,   64,  56,  56 }, { batch,   64,  56,  56 }, {   64,   64, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, { true, true, has_bias } ); // 2
  test ( { batch,  256,  56,  56 }, { batch,   64,  56,  56 }, {  256,   64, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 3
  test ( { batch,   64,  56,  56 }, { batch,  256,  56,  56 }, {   64,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 4
  test ( { batch,  128,  56,  56 }, { batch,  256,  56,  56 }, {  128,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 5
  test ( { batch,  128,  28,  28 }, { batch,  128,  56,  56 }, {  128,  128, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, { true, true, has_bias } ); // 6
  test ( { batch,  512,  28,  28 }, { batch,  128,  28,  28 }, {  512,  128, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 7
  test ( { batch,  512,  28,  28 }, { batch,  256,  56,  56 }, {  512,  256, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 8
  test ( { batch,  128,  28,  28 }, { batch,  512,  28,  28 }, {  128,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 9
  test ( { batch,  128,  28,  28 }, { batch,  128,  28,  28 }, {  128,  128, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, { true, true, has_bias } ); // 10
  test ( { batch,  256,  28,  28 }, { batch,  512,  28,  28 }, {  256,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 11
  test ( { batch,  256,  14,  14 }, { batch,  256,  28,  28 }, {  256,  256, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, { true, true, has_bias } ); // 12
  test ( { batch, 1024,  14,  14 }, { batch,  256,  14,  14 }, { 1024,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 13
  test ( { batch, 1024,  14,  14 }, { batch,  512,  28,  28 }, { 1024,  512, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 14
  test ( { batch,  256,  14,  14 }, { batch, 1024,  14,  14 }, {  256, 1024, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 15
  test ( { batch,  256,  14,  14 }, { batch,  256,  14,  14 }, {  256,  256, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, { true, true, has_bias } ); // 16
  test ( { batch,  512,  14,  14 }, { batch, 1024,  14,  14 }, {  512, 1024, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 17
  test ( { batch,  512,   7,   7 }, { batch,  512,  14,  14 }, {  512,  512, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, { true, true, has_bias } ); // 18
  test ( { batch, 2048,   7,   7 }, { batch,  512,   7,   7 }, { 2048,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 19
  test ( { batch, 2048,   7,   7 }, { batch, 1024,  14,  14 }, { 2048, 1024, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 20
  test ( { batch,  512,   7,   7 }, { batch, 2048,   7,   7 }, {  512, 2048, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, has_bias } ); // 21
  test ( { batch,  512,   7,   7 }, { batch,  512,   7,   7 }, {  512,  512, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, { true, true, has_bias } ); // 22
  return 0;
}
