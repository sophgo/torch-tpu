#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

#define TPU_MAX_CONCAT_NUM 16
#ifdef USING_PPL
#include "Concat.h"
#define AT_DISPATCH_FLOAT_INT_TYPES(scalar_type, name, func)  \
AT_DISPATCH_SWITCH(                   \
scalar_type, name,                    \
AT_DISPATCH_CASE(at::kFloat, func)    \
AT_DISPATCH_CASE(at::kHalf, func)     \
AT_DISPATCH_CASE(at::kBFloat16, func) \
AT_DISPATCH_CASE(at::kInt, func)      \
AT_DISPATCH_CASE(at::kShort, func)    \
AT_DISPATCH_CASE(at::kChar, func)     \
AT_DISPATCH_CASE(at::kByte, func))

#define MAX_CONCATLAYER_INPUT_NUM 16
template <typename scalar_t>
    static void concat_impl(
    uint64_t output_addr,
    const uint64_t inputs[MAX_CONCATLAYER_INPUT_NUM],
    int input_num,
    int outer_num,
    int out_inner_num,
    int dim,
    const int inner_nums[MAX_CONCATLAYER_INPUT_NUM]
    )
{
  using kernel_func_t =
      int (*)(
#ifndef BACKEND_SG2260
              tpuStream_t,
              tpuKernelModule_t,
#else
              tpudnnHandle_t,
#endif
              unsigned long long, unsigned long long, unsigned long long,
              unsigned long long, unsigned long long, unsigned long long,
              unsigned long long, unsigned long long, unsigned long long,
              unsigned long long, unsigned long long, unsigned long long,
              unsigned long long, unsigned long long, unsigned long long,
              unsigned long long, unsigned long long, int32_t, int32_t, int32_t,
              int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
              int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
              int32_t, int32_t, int32_t);

  kernel_func_t kernel_func = nullptr;
  if constexpr (std::is_same_v<scalar_t, float>) {
      kernel_func = concat_fp32;
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      kernel_func = concat_fp16;
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      kernel_func = concat_bf16;
  } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
      kernel_func = concat_int32;
  } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
      kernel_func = concat_int16;
  } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
      kernel_func = concat_int8;
  }

  if (kernel_func == nullptr) {
      TORCH_CHECK(false, "Unsupported scalar type for concat operation");
      return;
  }
  int ret = kernel_func(
    c10_tpu::getCurrentTPUStream(),
#ifndef BACKEND_SG2260
    getPplModule(),
#endif
    output_addr,
    inputs[0], inputs[1], inputs[2], inputs[3],
    inputs[4], inputs[5], inputs[6], inputs[7],
    inputs[8], inputs[9], inputs[10], inputs[11],
    inputs[12], inputs[13], inputs[14], inputs[15],
    input_num, outer_num, out_inner_num, dim,
    inner_nums[0], inner_nums[1], inner_nums[2], inner_nums[3],
    inner_nums[4], inner_nums[5], inner_nums[6], inner_nums[7],
    inner_nums[8], inner_nums[9], inner_nums[10], inner_nums[11],
    inner_nums[12], inner_nums[13], inner_nums[14], inner_nums[15]);
  TORCH_CHECK(ret == 0, "concat failed");
}
#endif

namespace at
{
Tensor & cat_out_tpu ( const ITensorListRef & tensors, int64_t dim, Tensor & out )
{
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE ( out );
  // if input.dtype != output.dtype , use cpu to convert.
  int flag = 0;
  for (auto tensor : tensors) {
    if (tensor.scalar_type() != out.scalar_type()) {
      flag = 1;
      break;
    }
  }
  if ( flag )
  {
    CPU_IMPL_WARNING();
    std::vector<Tensor> tensors_cpu;
    for ( auto tensor : tensors )
    {
      CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( tensor );
      tensors_cpu.push_back ( tensor.cpu() );
      ITensorListRef tensors_lis_cpu ( tensors_cpu.data(), tensors_cpu.size() );
      auto out_cpu = cat ( tensors_lis_cpu, dim );
      tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
    }
  }
  else
  {
    auto stream = c10_tpu::getCurrentTPUStream();
    std::vector<tpudnnTensor_t> inputs;
    std::vector<Tensor> contiguous_tensors;
    for ( auto tensor : tensors )
    {
      if (tensor.numel() == 0) {
        continue;
      }
      CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS ( tensor );
      contiguous_tensors.push_back ( tensor.contiguous() );
      inputs.push_back ( tpu:: TPUGenerateTpudnnTensor (stream, contiguous_tensors.back() ) );
    }

    if (inputs.size()!= 0)
    {
#ifdef USING_PPL
      if (usePPLKernels())
      {  std::vector<std::vector<int64_t>> input_shapes;
        for (const auto& tensor : contiguous_tensors) {
          input_shapes.push_back(tensor.sizes().vec());
        }
        if ( dim < 0 ) {
          dim += out.dim();
        }

        int outer_num = 1;
        for (int i = 0; i < dim; ++i) {
            outer_num *= input_shapes[0][i];
        }

        int input_inner_nums[MAX_CONCATLAYER_INPUT_NUM] = {0};
        int out_inner_num = 0;
        for (size_t i = 0; i < input_shapes.size(); ++i) {
            int inner_num = 1;
            for (size_t j = dim; j < input_shapes[i].size(); ++j) {
                inner_num *= input_shapes[i][j];
            }
            input_inner_nums[i] = inner_num;
            out_inner_num += inner_num;
        }

        ScalarType dtype = contiguous_tensors[0].scalar_type();
        int input_num = inputs.size();

        int merge_2dim_shape[MAX_CONCATLAYER_INPUT_NUM][2];
        if (dim == 0 ){
          int out_inner_num = 0;
          // [batch, other_dims_product]
          for (size_t i = 0; i < input_shapes.size(); ++i) {
            merge_2dim_shape[i][0] = input_shapes[i][0];
            merge_2dim_shape[i][1] = 1;
            out_inner_num += merge_2dim_shape[i][0];
            for (size_t d = 1; d < input_shapes[i].size(); ++d) {
              merge_2dim_shape[i][1] *= input_shapes[i][d];
            }
          }
        }
        uint64_t input_addrs[MAX_CONCATLAYER_INPUT_NUM] = {0};
        for (int i = 0; i < input_num && i < MAX_CONCATLAYER_INPUT_NUM; ++i) {
            input_addrs[i] = reinterpret_cast<uint64_t>(contiguous_tensors[i].data_ptr());
        }
        int special_inner_nums[MAX_CONCATLAYER_INPUT_NUM];
        for (int i = 0; i < MAX_CONCATLAYER_INPUT_NUM; ++i) {
            special_inner_nums[i] = (i < input_num) ? merge_2dim_shape[i][0] : 0;
        }
        AT_DISPATCH_FLOAT_INT_TYPES(dtype, "concat", ([&] {
          concat_impl<scalar_t>(
              reinterpret_cast<uint64_t>(out.data_ptr()),
              input_addrs,
              input_num,
              dim == 0 ? merge_2dim_shape[0][1] : outer_num,
              out_inner_num,
              dim,
              dim == 0 ? special_inner_nums : input_inner_nums
          );
        }));
      } else
#endif
      {  auto status = tpudnnConcatAsync(
        stream,
        inputs.data(),
        inputs.size(),
        dim,
        tpu::TPUGenerateTpudnnTensor(stream, out));
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
      }
    }

  }
  TIMING_END;
  SHOW_TENSOR_OP(out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "cat.out", cat_out_tpu );
}
} // namespace at