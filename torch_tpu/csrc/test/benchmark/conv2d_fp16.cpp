#include <torch/torch.h>
#include <TPUTorchUtils.h>

static int loops = 0;

static inline void benchmark ( at::IntArrayRef input_shape,
                               at::IntArrayRef weight_shape,
                               at::IntArrayRef stride,
                               at::IntArrayRef padding,
                               at::IntArrayRef dilation,
                               int64_t groups,
                               bool has_bias )
{
  auto input_cpu = torch::randn ( input_shape, c10::kHalf ) - 0.5;
  auto weight_cpu = torch::randn ( weight_shape, c10::kHalf ) - 0.5;
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
  at::Tensor output_tpu;
  tpu::Timer timer;
  timer.Start();
  for ( int i = 0; i < loops; ++i )
  {
    output_tpu = torch::convolution (
                 input_tpu,
                 weight_tpu,
                 c10::optional<at::Tensor> ( bias_tpu ),
                 stride,
                 padding,
                 dilation,
                 false,
                 at::IntArrayRef ( { 0, 0 } ),
                 groups );
  }
  unsigned long elapsed_us_per_loop = ( double ) timer.ElapsedUS() / loops;
  std::cout << "Benchmarking convolution:"
            << " Elapsed time = " << elapsed_us_per_loop << "us"
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
}

int main()
{
  const int batch = 64;
  const bool has_bias = false;
  loops = 500;
  // input shape, weight shape, stride, padding, dilation, groups, has bias
  // Resnet50
  benchmark ( { batch,    3, 224, 224 }, {   64,    3, 7, 7 }, { 2, 2 }, { 3, 3 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,   64,  56,  56 }, {   64,   64, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,   64,  56,  56 }, {   64,   64, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,   64,  56,  56 }, {  256,   64, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  256,  56,  56 }, {   64,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  256,  56,  56 }, {  128,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  128,  56,  56 }, {  128,  128, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  128,  28,  28 }, {  512,  128, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  256,  56,  56 }, {  512,  256, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  512,  28,  28 }, {  128,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  128,  28,  28 }, {  128,  128, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  512,  28,  28 }, {  256,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  256,  28,  28 }, {  256,  256, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  256,  14,  14 }, { 1024,  256, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  512,  28,  28 }, { 1024,  512, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch, 1024,  14,  14 }, {  256, 1024, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  256,  14,  14 }, {  256,  256, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch, 1024,  14,  14 }, {  512, 1024, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  512,  14,  14 }, {  512,  512, 3, 3 }, { 2, 2 }, { 1, 1 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  512,   7,   7 }, { 2048,  512, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch, 1024,  14,  14 }, { 2048, 1024, 1, 1 }, { 2, 2 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch, 2048,   7,   7 }, {  512, 2048, 1, 1 }, { 1, 1 }, { 0, 0 }, { 1, 1 }, 1, has_bias );
  benchmark ( { batch,  512,   7,   7 }, {  512,  512, 3, 3 }, { 1, 1 }, { 1, 1 }, { 1, 1 }, 1, has_bias );
  return 0;
}
