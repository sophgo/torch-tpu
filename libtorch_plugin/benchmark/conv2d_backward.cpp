#include <torch/torch.h>
#include <TPUTorchUtils.h>

static int loops = 0;

static inline void test ( at::IntArrayRef grad_output_shape,
                          at::IntArrayRef input_shape,
                          at::IntArrayRef weight_shape,
                          at::IntArrayRef stride,
                          at::IntArrayRef padding,
                          at::IntArrayRef dilation,
                          int64_t groups,
                          std::array<bool, 3> output_mask )
{
  auto grad_output_cpu = torch::randn ( grad_output_shape ) - 0.5;
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto weight_cpu = torch::randn ( weight_shape ) - 0.5;
  auto grad_output_tpu = grad_output_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto weight_tpu = weight_cpu.to ( tpu::TPUGetCurrentDevice() );
  std::tuple<at::Tensor, at::Tensor, at::Tensor> outputs_tpu;
  tpu::Timer timer;
  timer.Start();
  for ( int i = 0; i < loops; ++i )
  {
    outputs_tpu = torch::convolution_backward (
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
  }
  unsigned long elapsed_us_per_loop = ( double ) timer.ElapsedUS() / loops;
  std::cout << "Comparing convolution backward:"
            << " Elapsed time = " << elapsed_us_per_loop << "us"
            << " grad_output shape = " << grad_output_tpu.sizes()
            << " grad_output dtype = " << grad_output_tpu.dtype()
            << " input shape = " << input_tpu.sizes()
            << " input dtype = " << input_tpu.dtype()
            << " weight shape = " << weight_tpu.sizes()
            << " weight dtype = " << weight_tpu.dtype()
            << " grad_input shape = " << std::get<0> ( outputs_tpu ).sizes()
            << " grad_input dtype = " << std::get<0> ( outputs_tpu ).dtype()
            << " grad_weight shape = " << std::get<1> ( outputs_tpu ).sizes()
            << " grad_weight dtype = " << std::get<1> ( outputs_tpu ).dtype()
            << " grad_bias shape = " << std::get<2> ( outputs_tpu ).sizes()
            << " grad_bias dtype = " << std::get<2> ( outputs_tpu ).dtype()
            << " stride = " << stride
            << " padding = " << padding
            << " dilation = " << dilation
            << " groups = " << groups
            << std::endl;
}

int main()
{
  const int batch = 64;
  loops = 50;
  // grad output shape, input shape, weight shape, stride, padding, dilation, groups
  test ( { batch,   64, 112, 112 }, { batch,    3, 224, 224 }, {   64,    3, 7, 7 }, { 2, 2 }, { 3, 3 }, { 1, 1 }, 1, { true, true, false } ); // 0
  test ( { batch,   64,  56,  56 }, { batch,   64,  56,  56 }, {   64,   64, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 1
  test ( { batch,   64,  56,  56 }, { batch,   64,  56,  56 }, {   64,   64, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, { true, true, false } ); // 2
  test ( { batch,  256,  56,  56 }, { batch,   64,  56,  56 }, {  256,   64, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 3
  test ( { batch,   64,  56,  56 }, { batch,  256,  56,  56 }, {   64,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 4
  test ( { batch,  128,  56,  56 }, { batch,  256,  56,  56 }, {  128,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 5
  test ( { batch,  128,  28,  28 }, { batch,  128,  56,  56 }, {  128,  128, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, { true, true, false } ); // 6
  test ( { batch,  512,  28,  28 }, { batch,  128,  28,  28 }, {  512,  128, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 7
  test ( { batch,  512,  28,  28 }, { batch,  256,  56,  56 }, {  512,  256, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 8
  test ( { batch,  128,  28,  28 }, { batch,  512,  28,  28 }, {  128,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 9
  test ( { batch,  128,  28,  28 }, { batch,  128,  28,  28 }, {  128,  128, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, { true, true, false } ); // 10
  test ( { batch,  256,  28,  28 }, { batch,  512,  28,  28 }, {  256,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 11
  test ( { batch,  256,  14,  14 }, { batch,  256,  28,  28 }, {  256,  256, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, { true, true, false } ); // 12
  test ( { batch, 1024,  14,  14 }, { batch,  256,  14,  14 }, { 1024,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 13
  test ( { batch, 1024,  14,  14 }, { batch,  512,  28,  28 }, { 1024,  512, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 14
  test ( { batch,  256,  14,  14 }, { batch, 1024,  14,  14 }, {  256, 1024, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 15
  test ( { batch,  256,  14,  14 }, { batch,  256,  14,  14 }, {  256,  256, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, { true, true, false } ); // 16
  test ( { batch,  512,  14,  14 }, { batch, 1024,  14,  14 }, {  512, 1024, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 17
  test ( { batch,  512,   7,   7 }, { batch,  512,  14,  14 }, {  512,  512, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, { true, true, false } ); // 18
  test ( { batch, 2048,   7,   7 }, { batch,  512,   7,   7 }, { 2048,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 19
  test ( { batch, 2048,   7,   7 }, { batch, 1024,  14,  14 }, { 2048, 1024, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 20
  test ( { batch,  512,   7,   7 }, { batch, 2048,   7,   7 }, {  512, 2048, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, { true, true, false } ); // 21
  test ( { batch,  512,   7,   7 }, { batch,  512,   7,   7 }, {  512,  512, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, { true, true, false } ); // 22
  return 0;
}
