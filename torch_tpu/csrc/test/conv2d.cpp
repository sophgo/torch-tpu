#include <torch/torch.h>
#include <TPUTorchUtils.h>

static inline void test ( at::IntArrayRef input_shape,
                          at::IntArrayRef weight_shape,
                          at::IntArrayRef stride,
                          at::IntArrayRef padding,
                          at::IntArrayRef dilation,
                          int64_t groups,
                          bool has_bias )
{
  auto input_cpu = torch::randn ( input_shape ) - 0.5;
  auto weight_cpu = torch::randn ( weight_shape ) - 0.5;
  at::Tensor bias_cpu;
  if ( has_bias )
  {
    bias_cpu = torch::randn ( { weight_shape[0] } ) - 0.5;
  }
  auto input_tpu = input_cpu.to ( tpu::TPUGetCurrentDevice() );
  auto weight_tpu = weight_cpu.to ( tpu::TPUGetCurrentDevice() );
  at::Tensor bias_tpu;
  if ( has_bias )
  {
    bias_tpu = bias_cpu.to ( tpu::TPUGetCurrentDevice() );
  }
  input_cpu = input_cpu.to ( c10::kDouble );
  weight_cpu = weight_cpu.to ( c10::kDouble );
  if ( has_bias )
  {
    bias_cpu = bias_cpu.to ( c10::kDouble );
  }
  auto output_cpu = torch::convolution (
                    input_cpu,
                    weight_cpu,
                    c10::optional<at::Tensor> ( bias_cpu ),
                    stride,
                    padding,
                    dilation,
                    false,
                    at::IntArrayRef ( { 0, 0 } ),
                    groups );
  auto output_tpu = torch::convolution (
                    input_tpu,
                    weight_tpu,
                    c10::optional<at::Tensor> ( bias_tpu ),
                    stride,
                    padding,
                    dilation,
                    false,
                    at::IntArrayRef ( { 0, 0 } ),
                    groups );
  auto output_got = output_tpu.to ( torch::Device ( "cpu" ) );
  auto output_exp = output_cpu;
  std::cout << "Comparing convolution:"
            << " input shape = " << input_tpu.sizes()
            << " input dtype = " << input_tpu.dtype()
            << " weight shape = " << weight_tpu.sizes()
            << " weight dtype = " << weight_tpu.dtype()
            << " bias shape = " << bias_tpu.sizes()
            << " bias dtype = " << bias_tpu.dtype()
            << " output shape = " << output_tpu.sizes()
            << " output dtype = " << output_tpu.dtype()
            << " stride = " << stride
            << " padding = " << padding
            << " dilation = " << dilation
            << " groups = " << groups
            << std::endl;
#if 0
  tpu::TPUCompareResult ( output_got, output_exp, 1.0 / ( ( double ) weight_tpu.size ( 1 ) * weight_tpu.size ( 2 ) * weight_tpu.size ( 3 ) ) );
#else
  tpu::TPUCompareResult ( output_got.to ( c10::kFloat ), output_exp.to ( c10::kFloat ) );
#endif
}

int main()
{
  const int batch = 64;
  const bool has_bias = true;
  // input shape, weight shape, stride, padding, dilation, groups, has bias
  // Resnet50
  test ( { batch,    3, 224, 224 }, {   64,    3, 7, 7 }, { 2, 2 }, { 3, 3 }, { 1, 1 }, 1, has_bias ); // 0
  test ( { batch,   64,  56,  56 }, {   64,   64, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 1
  test ( { batch,   64,  56,  56 }, {   64,   64, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, has_bias ); // 2
  test ( { batch,   64,  56,  56 }, {  256,   64, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 3
  test ( { batch,  256,  56,  56 }, {   64,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 4
  test ( { batch,  256,  56,  56 }, {  128,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 5
  test ( { batch,  128,  56,  56 }, {  128,  128, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, has_bias ); // 6
  test ( { batch,  128,  28,  28 }, {  512,  128, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 7
  test ( { batch,  256,  56,  56 }, {  512,  256, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 8
  test ( { batch,  512,  28,  28 }, {  128,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 9
  test ( { batch,  128,  28,  28 }, {  128,  128, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, has_bias ); // 10
  test ( { batch,  512,  28,  28 }, {  256,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 11
  test ( { batch,  256,  28,  28 }, {  256,  256, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, has_bias ); // 12
  test ( { batch,  256,  14,  14 }, { 1024,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 13
  test ( { batch,  512,  28,  28 }, { 1024,  512, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 14
  test ( { batch, 1024,  14,  14 }, {  256, 1024, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 15
  test ( { batch,  256,  14,  14 }, {  256,  256, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, has_bias ); // 16
  test ( { batch, 1024,  14,  14 }, {  512, 1024, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 17
  test ( { batch,  512,  14,  14 }, {  512,  512, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, has_bias ); // 18
  test ( { batch,  512,   7,   7 }, { 2048,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 19
  test ( { batch, 1024,  14,  14 }, { 2048, 1024, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 20
  test ( { batch, 2048,   7,   7 }, {  512, 2048, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias ); // 21
  test ( { batch,  512,   7,   7 }, {  512,  512, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, has_bias ); // 22
  return 0;
}
