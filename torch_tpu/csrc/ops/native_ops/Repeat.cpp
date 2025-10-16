#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>


#include "TPUTorchUtils.h"
#include "common/config.h"

#ifdef USING_PPL
#include "Repeat.h"
#include <algorithm>
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

void simply_params(int repeat_dim, int* repeat_times,
              int input_dim,  int* input_shape,
              int* simply_dim, int* simply_repeat_times, int* simply_shape)
  {
    bool can_simply = true;
    int repeat_size = 1;
    *simply_dim = input_dim;
    for (int i = input_dim - 1; i >= 0; i--)
    {
      simply_shape[i] = input_shape[i];
      if (can_simply)
      {  if ( input_shape[i] == 1) {
          repeat_size *= repeat_times[i];
          simply_repeat_times[i] = repeat_size;
        } else {
          can_simply = false;
          simply_repeat_times[i] = repeat_times[i];
          *simply_dim = std::min(i + 2, input_dim);
        }
      }
      else
      {
        simply_repeat_times[i] = repeat_times[i];
      }
    }
  }

template <typename scalar_t>
static void repeat_impl(
  uint64_t output_addr,
  uint64_t input_addr,
  int rows,
  int cols,
  int repeat
  )
{
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return repeat_fp32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr,
        rows, cols, repeat);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return repeat_fp16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr,
        rows, cols, repeat);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return repeat_bf16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr,
        rows, cols, repeat);
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return repeat_int32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr,
        rows, cols, repeat);
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
      return repeat_int16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr,
        rows, cols, repeat);
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
      return repeat_int8(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr,
        rows, cols, repeat);
    } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
      return repeat_uint8(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr,
        rows, cols, repeat);
    }
    return -1;
  };

  auto stream = c10_tpu::getCurrentTPUStream();
  tpuKernelModule_t ppl_module = getPplModule();
  int ret = kernel(stream, ppl_module);
  TORCH_CHECK(ret == 0, "repeat failed");
}
#endif
namespace at {

Tensor &repeat_out_tpu(const Tensor &self, const IntArrayRef repeats,
                       Tensor &out) {
  TIMING_START;
  if (self.nbytes() == 0) {  TIMING_END; SHOW_TENSOR_OP(self, out); return out;}
  Tensor contiguous_self = self.is_contiguous() ? self : self.contiguous();
  std::vector<int> repeat_times;
#if 0
    CPU_IMPL_WARNING();
    Tensor out_cpu = out.cpu();
    repeat_out(out_cpu, self.cpu(), repeats);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
#else
  for (int i = 0; i < (int)repeats.size(); ++i) {
      repeat_times.push_back((int)repeats[i]);
    }
#ifdef USING_PPL
  if (usePPLKernels()) {
    auto sizes = contiguous_self.sizes();
    std::vector<int> input_shape(sizes.begin(), sizes.end());

    int rows = 1, cols = 1, repeat = 1;
    int simply_dim = contiguous_self.dim();
    std::vector<int> simply_shape(simply_dim);
    std::vector<int> simply_repeat_times(simply_dim);

    simply_params(repeats.size(), repeat_times.data(),
                  contiguous_self.dim(), input_shape.data(),
                  &simply_dim, simply_repeat_times.data(), simply_shape.data());

    cols = simply_shape[simply_dim-1];
    repeat = simply_repeat_times[simply_dim - 1];
    for (int i = 0; i < simply_dim - 1; ++i) {
      rows *= simply_shape[i];
      printf("simply_shape[%d]=%d\n", i, simply_shape[i]);
    }
  // multi-core opt only support repeatting last dim now
    AT_DISPATCH_FLOAT_INT_TYPES(contiguous_self.scalar_type(), "repeat", ([&] {
            repeat_impl<scalar_t>(
                reinterpret_cast<uint64_t>(out.data_ptr()),
                reinterpret_cast<uint64_t>(contiguous_self.data_ptr()),
                rows, cols, repeat);
      }));
    } else
#endif
    { auto stream = c10_tpu::getCurrentTPUStream();
      auto status = tpudnnRepeatAsync(
          stream,
          tpu::TPUGenerateTpudnnTensor(stream, contiguous_self),
          repeat_times.data(),
          repeats.size(),
          tpu::TPUGenerateTpudnnTensor(stream, out));
      TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
//  - func: repeat(Tensor self, SymInt[] repeats) -> Tensor
Tensor repeat_tpu(const Tensor &self, const IntArrayRef repeats) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
    TORCH_CHECK((int64_t)repeats.size() >= self.dim());
  }

  IntArrayRef size(repeats);
  std::vector<int64_t> size_vec(size.begin(), size.end());
  if (repeats.size() == (size_t)self.dim()) {
    for (int i = 0; i < self.dim(); ++i) {
      size_vec[i] = repeats[i] * self.size(i);
    }
  } else {
    int dist = repeats.size() - self.dim();
    for (int i = 0; i < (int)repeats.size(); ++i) {
      if (i < dist) {
        continue;
      } else {
        size_vec[i] *= self.size(i - dist);
      }
    }
  }
  size = torch::IntArrayRef(size_vec);
  auto out = empty(size, self.options());
  repeat_out_tpu(self, repeats, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("repeat.out", repeat_out_tpu);
  m.impl("repeat", repeat_tpu);
}

} // namespace at