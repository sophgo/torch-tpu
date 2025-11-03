#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/quantized/QTensorImpl.h>

#include "TPUTorchUtils.h"
#include "TPUStream.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"
#include <cmath>
#include <float.h>

#include <tpuDNN.h>

#undef USING_PPL // FIXME Fix Compare.pl & remove this

#ifdef USING_PPL
#include "Binary.h"
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

auto process_shapes = [](std::vector<int>& self_shape, std::vector<int>& other_shape) {
  auto pad_to_left = [](std::vector<int>& v, int target) {
    if ((int)v.size() < target) {
      std::vector<int> nv(target, 1);
      for (int i = 0; i < (int)v.size(); ++i) nv[target - 1 - i] = v[v.size() - 1 - i];
      v.swap(nv);
    }
  };
  auto can_be_merged = [](int a0, int a1, int b0, int b1) -> bool {
    bool d0_ok = (a0 == b0) || (a0 == 1) || (b0 == 1);
    bool d1_ok = (a1 == b1) || (a1 == 1) || (b1 == 1);
    return d0_ok && d1_ok;
  };
  auto merge_two_dims = [](std::vector<int>& as, std::vector<int>& bs, int i) {
    as[i] = as[i] * as[i + 1];
    bs[i] = bs[i] * bs[i + 1];
    as.erase(as.begin() + i + 1);
    bs.erase(bs.begin() + i + 1);
  };

  int dim = (int)other_shape.size();
  if (dim < 4) {
    pad_to_left(self_shape, 4);
    pad_to_left(other_shape, 4);
    dim = 4;
  } else if (dim > 4) {
    int i = 0;
    while (i < dim - 1) {
      if (can_be_merged(self_shape[i], self_shape[i+1], other_shape[i], other_shape[i+1])) {
        merge_two_dims(self_shape, other_shape, i);
        --dim;
        if (dim == 4) break;
      } else {
        ++i;
      }
    }
    TORCH_CHECK(dim == 4, "can not reduce dims to 4: got ", dim);
  }
};

template <typename scalar_t>
static void binary_const_impl(
  uint64_t output_addr,
  uint64_t input_addr,
  uint64_t other_addr,
  const int binary_type,
  int length,
  float alpha = 1.0f,
  bool is_scalar_left = false)
{
  auto kernel = [&](tpuStream_t stream, tpuKernelModule_t ppl_module,
                    uint32_t tile_size) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return binary_async_fp32_scalar(
        stream, ppl_module, output_addr, input_addr,
        binary_type, tile_size,
        length,alpha,is_scalar_left);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return binary_async_fp16_scalar(
        stream, ppl_module, output_addr, input_addr,
        binary_type, tile_size,
        length,alpha,is_scalar_left);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return binary_async_bf16_scalar(
        stream, ppl_module, output_addr, input_addr,
        binary_type, tile_size,
        length,alpha,is_scalar_left);
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
      if (binary_type == 3){
        return div_int32_fp32_scalar(
          stream, ppl_module, output_addr, input_addr, binary_type, tile_size,
          length,alpha,is_scalar_left);
      } else {
        return binary_async_int32_scalar(
          stream, ppl_module, output_addr, input_addr,
          binary_type, tile_size,
          length,alpha,is_scalar_left);
      }
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
      if (binary_type == 3){
        return div_int16_fp32_scalar(
          stream, ppl_module, output_addr, input_addr, binary_type, tile_size,
          length,alpha,is_scalar_left);
      } else {
        return binary_async_int16_scalar(
          stream, ppl_module, output_addr, input_addr,
          binary_type, tile_size,
          length,alpha,is_scalar_left);
      }
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
      if (binary_type == 3){
        return div_int8_fp32_scalar(
          stream, ppl_module, output_addr, input_addr, binary_type, tile_size,
          length,alpha,is_scalar_left);
      } else {
        return binary_async_int8_scalar(
          stream, ppl_module, output_addr, input_addr,
          binary_type, tile_size,
          length,alpha,is_scalar_left);
      }
    } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
      if (binary_type == 3){
        return div_uint8_fp32_scalar(
          stream, ppl_module, output_addr, input_addr, binary_type, tile_size,
          length,alpha,is_scalar_left);
      } else {
        return binary_async_uint8_scalar(
          stream, ppl_module, output_addr, input_addr,
          binary_type, tile_size,
          length,alpha,is_scalar_left);
      }
    }
    return -1;
  };

  tpuStream_t stream = c10_tpu::getCurrentTPUStream().stream();
  tpuKernelModule_t ppl_module = getPplModule();
  uint32_t tile_size = length;

  while (tile_size >= 1) {
    int ret = kernel(stream, ppl_module, tile_size);
    if (ret == 0) return;
    tile_size = tile_size / 2;
  }
  TORCH_CHECK(false, "binary_const_impl failed");
}

template <typename scalar_t>
static void binary_async_impl(
  uint64_t output_addr,
  uint64_t input_addr,
  uint64_t other_addr,
  const int binary_type,
  int outer_size,
  int inner_size)
{
  auto kernel = [&](tpuStream_t stream, tpuKernelModule_t ppl_module,
                    uint32_t tile_size) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return binary_async_fp32(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        outer_size, inner_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return binary_async_fp16(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        outer_size, inner_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return binary_async_bf16(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        outer_size, inner_size);
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
      if (binary_type == 3){
        return div_int32_fp32(
          stream, ppl_module, output_addr, input_addr, other_addr, tile_size,
          outer_size, inner_size);
      } else {
        return binary_async_int32(
          stream, ppl_module, output_addr, input_addr, other_addr,
          binary_type, tile_size,
          outer_size, inner_size);
      }
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
      if (binary_type == 3){
        return div_int16_fp32(
          stream, ppl_module, output_addr, input_addr, other_addr, tile_size,
          outer_size, inner_size);
      } else {
        return binary_async_int16(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        outer_size, inner_size);
      }
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
      if (binary_type == 3){
        return div_int8_fp32(
          stream, ppl_module, output_addr, input_addr, other_addr, tile_size,
          outer_size, inner_size);
      } else {
        return binary_async_int8(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        outer_size, inner_size);
      }
    } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
      if (binary_type == 3){
        return div_uint8_fp32(
          stream, ppl_module, output_addr, input_addr, other_addr, tile_size,
          outer_size, inner_size);
      } else {
        return binary_async_uint8(
          stream, ppl_module, output_addr, input_addr, other_addr,
          binary_type, tile_size,
          outer_size, inner_size);
        }

      return -1;
    }
  };

  tpuStream_t stream = c10_tpu::getCurrentTPUStream().stream();
  tpuKernelModule_t ppl_module = getPplModule();
  uint32_t tile_size = inner_size;

  while (tile_size >= 1) {
    int ret = kernel(stream, ppl_module, tile_size);
    if (ret == 0) return;
    tile_size = tile_size / 2;
  }
  TORCH_CHECK(false, "binary_async_impl failed");
}

template <typename scalar_t>
static void binary_bcast_impl(
  uint64_t output_addr,
  uint64_t input_addr,
  uint64_t other_addr,
  const int binary_type,
  std::vector<int> self_shape,
  std::vector<int> other_shape)
{
  std::vector<int> A4 = self_shape;
  std::vector<int> B4 = other_shape;
  process_shapes(A4, B4);
  auto kernel = [&](tpuStream_t stream, tpuKernelModule_t ppl_module,
                    uint32_t tile_size) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return binary_bcast_fp32(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return binary_bcast_fp16(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return binary_bcast_bf16(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
      if (binary_type == 3){
        return div_bcast_int32_fp32(
          stream, ppl_module, output_addr, input_addr, other_addr, tile_size,
          A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
      } else {
        return binary_bcast_int32(
          stream, ppl_module, output_addr, input_addr, other_addr,
          binary_type, tile_size,
          A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
      }
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
      if (binary_type == 3){
        return div_bcast_int16_fp32(
          stream, ppl_module, output_addr, input_addr, other_addr, tile_size,
          A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
      } else {
        return binary_bcast_int16(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
      }
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
      if (binary_type == 3){
        return div_bcast_int8_fp32(
          stream, ppl_module, output_addr, input_addr, other_addr, tile_size,
          A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
      } else {
        return binary_bcast_int8(
        stream, ppl_module, output_addr, input_addr, other_addr,
        binary_type, tile_size,
        A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
      }
    } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
      if (binary_type == 3){
        return div_bcast_uint8_fp32(
          stream, ppl_module, output_addr, input_addr, other_addr, tile_size,
          A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
      } else {
        return binary_bcast_uint8(
          stream, ppl_module, output_addr, input_addr, other_addr,
          binary_type, tile_size,
          A4[0],A4[1],A4[2],A4[3], B4[0],B4[1],B4[2],B4[3]);
        }
      return -1;
    }
  };

  tpuStream_t stream = c10_tpu::getCurrentTPUStream().stream();
  tpuKernelModule_t ppl_module = getPplModule();
  uint32_t tile_size = A4[3]*A4[2];

  while (tile_size >= 1) {
    int ret = kernel(stream, ppl_module, tile_size);
    if (ret == 0) return;
    tile_size = tile_size / 2;
  }
  TORCH_CHECK(false, "binary_bcast_impl failed");
}
// ------------------------------------------------
#include "Compare.h"

enum class BinaryOpType {
  MINMAX,    // min/max
  COMPARE,   // eq/ge/gt/le/lt/ne
  SHIFT,     // left/right
  BITWISE,    // and/or/xor
  POW,        // pow
  ATAN2      // atan2
};

template <BinaryOpType Op, typename scalar_t, typename Mode = void>
struct BinaryOpDispatcher;

template <typename scalar_t, typename Mode>
struct BinaryOpDispatcher<BinaryOpType::MINMAX, scalar_t, Mode> {
  static int invoke(tpuStream_t stream, tpuKernelModule_t ppl_module,
                   uint64_t output_addr, uint64_t input_addr, float scalar,
                   bool mode, bool out_is_int, uint32_t outer_size, uint32_t inner_size,
                   uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return minmax_int32(stream, ppl_module, output_addr, input_addr,
                         scalar, mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      return minmax_fp32(stream, ppl_module, output_addr, input_addr,
                        scalar, mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return minmax_fp16(stream, ppl_module, output_addr, input_addr,
                        scalar, mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return minmax_bf16(stream, ppl_module, output_addr, input_addr,
                        scalar, mode, outer_size, inner_size, tile_size);
    }
    return -1;
  }
};

template <typename scalar_t, typename Mode>
struct BinaryOpDispatcher<BinaryOpType::SHIFT, scalar_t, Mode> {
  static int invoke(tpuStream_t stream, tpuKernelModule_t ppl_module,
                   uint64_t output_addr, uint64_t input_addr, float shift_c,
                   bool mode, bool out_is_int, uint32_t outer_size, uint32_t inner_size,
                   uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return shift_const_int32(stream, ppl_module, output_addr, input_addr,
                         shift_c, outer_size, inner_size, tile_size);
    }
    return -1;
  }
};

template <typename scalar_t, typename Mode>
struct BinaryOpDispatcher<BinaryOpType::COMPARE, scalar_t, Mode> {
  static int invoke(tpuStream_t stream, tpuKernelModule_t ppl_module,
                   uint64_t output_addr, uint64_t input_addr, float scalar,
                   Mode mode, bool out_is_int, uint32_t outer_size, uint32_t inner_size,
                   uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return compare_const_int32(stream, ppl_module, output_addr, input_addr,
                               scalar, mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      return compare_const_fp32(stream, ppl_module, output_addr, input_addr,
                              scalar, mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return compare_const_fp16(stream, ppl_module, output_addr, input_addr,
                              scalar, mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return compare_const_bf16(stream, ppl_module, output_addr, input_addr,
                              scalar, mode, outer_size, inner_size, tile_size);
    }
    return -1;
  }
};

template <typename scalar_t, typename Mode>
struct BinaryOpDispatcher<BinaryOpType::BITWISE, scalar_t, Mode> {
  static int invoke(tpuStream_t stream, tpuKernelModule_t ppl_module,
                   uint64_t output_addr, uint64_t input_addr, float scalar,
                   bool mode, bool out_is_int, uint32_t outer_size, uint32_t inner_size,
                   uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return bitwise_const_int32(stream, ppl_module, output_addr, input_addr,
                         scalar, mode, outer_size, inner_size, tile_size);
    }
    return -1;
  }
};

template <typename scalar_t, typename Mode>
struct BinaryOpDispatcher<BinaryOpType::POW, scalar_t, Mode> {
  static int invoke(tpuStream_t stream, tpuKernelModule_t ppl_module,
                   uint64_t output_addr, uint64_t input_addr, double scalar,
                   bool mode, bool out_is_int, uint32_t outer_size, uint32_t inner_size,
                   uint32_t tile_size) {
    if (mode == 0 ){
      if constexpr (std::is_same_v<scalar_t, int32_t>) {
        if (out_is_int) {
          return powc_const_int32_int32(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
        } else {
          return powc_const_int32(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
        }
      } else if constexpr (std::is_same_v<scalar_t, float>) {
        return powc_const_fp32(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        return powc_const_fp16(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        return powc_const_bf16(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
      }
    } else {
      if constexpr (std::is_same_v<scalar_t, int32_t>) {
        if (out_is_int) {
          return cpow_const_int32_int32(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
        } else {
          return cpow_const_int32(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
        }
      } else if constexpr (std::is_same_v<scalar_t, float>) {
        return cpow_const_fp32(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
      } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        return cpow_const_fp16(stream, ppl_module, output_addr, input_addr,
                                scalar, outer_size, inner_size, tile_size);
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        return cpow_const_bf16(stream, ppl_module, output_addr, input_addr,
                                scalar,outer_size, inner_size, tile_size);
      }
    }
    return -1;
  }
};

template <typename scalar_t, typename Mode>
struct BinaryOpDispatcher<BinaryOpType::ATAN2, scalar_t, Mode> {
  static int invoke(tpuStream_t stream, tpuKernelModule_t ppl_module,
                   uint64_t output_addr, uint64_t input_addr, double scalar,
                   bool mode, bool out_is_int, uint32_t outer_size, uint32_t inner_size,
                   uint32_t tile_size) {
    if ( mode == 0 ){
      if constexpr (std::is_same_v<scalar_t, float>) {
        return atan2_const_fp32(stream, ppl_module, output_addr, input_addr,
                                scalar, mode, outer_size, inner_size, tile_size);
      } else {
        TORCH_CHECK(false, "Only float type is supported for powc with mode 0");
      }
    }
    else {
      if constexpr (std::is_same_v<scalar_t, float>) {
        return const_atan2_fp32(stream, ppl_module, output_addr, input_addr,
                                scalar, mode, outer_size, inner_size, tile_size);
      } else {
        TORCH_CHECK(false, "Only float type is supported for cpow with mode 1");
      }
    }
    return -1;
  }
};

template <BinaryOpType Op, typename scalar_t, typename Mode = void>
static void unified_const_impl(
    uint64_t output_addr,
    uint64_t input_addr,
    float scalar,
    Mode mode,
    uint32_t outer_size,
    uint32_t inner_size,
    bool out_is_int = false) {

  auto kernel = [&](tpuStream_t stream, tpuKernelModule_t ppl_module,
                   uint32_t tile_size) -> int {
    return BinaryOpDispatcher<Op, scalar_t, Mode>::invoke(
        stream, ppl_module, output_addr, input_addr, scalar,
        mode, out_is_int, outer_size, inner_size, tile_size);
  };

  tpuStream_t stream = c10_tpu::getCurrentTPUStream().stream();
  tpuKernelModule_t ppl_module = getPplModule();
  uint32_t tile_size = inner_size;

  while (tile_size >= 1) {
    int ret = kernel(stream, ppl_module, tile_size);
    if (ret == 0) {
      return;
    } else {
      tile_size = tile_size / 2;
      continue;
    }
  }

  TORCH_CHECK(false, "Unified const operation failed for op type: ", static_cast<int>(Op));
}

// 非const
template <BinaryOpType Op, typename scalar_t, typename Mode = void>
struct BinaryForwardOpDispatcher;

template <typename scalar_t, typename Mode>
struct BinaryForwardOpDispatcher<BinaryOpType::MINMAX, scalar_t, Mode> {
  static int invoke_same_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                             uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                             Mode mode, uint32_t outer_size, uint32_t inner_size,
                             uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return minmax_forward_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                                mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      return minmax_forward_fp32(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return minmax_forward_fp16(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return minmax_forward_bf16(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size);
    }
    return -1;
  }

  static int invoke_bcast_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                              uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                              Mode mode, uint32_t outer_size, uint32_t inner_size,
                              uint32_t tile_size, const std::vector<int>& self_shape,
                              const std::vector<int>& other_shape) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return minmax_bcast_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                              mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      return minmax_bcast_fp32(stream, ppl_module, output_addr, input_addr, other_addr,
                             mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return minmax_bcast_fp16(stream, ppl_module, output_addr, input_addr, other_addr,
                             mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return minmax_bcast_bf16(stream, ppl_module, output_addr, input_addr, other_addr,
                             mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    }
    return -1;
  }
};

template <typename scalar_t, typename Mode>
struct BinaryForwardOpDispatcher<BinaryOpType::SHIFT, scalar_t, Mode> {
  static int invoke_same_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                             uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                             Mode mode, uint32_t outer_size, uint32_t inner_size,
                             uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return shift_forward_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                                mode, outer_size, inner_size, tile_size);
    } else {
      return -1;
    }
  }
    static int invoke_bcast_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                              uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                              Mode mode, uint32_t outer_size, uint32_t inner_size,
                              uint32_t tile_size, const std::vector<int>& self_shape,
                              const std::vector<int>& other_shape) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return shift_bcast_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else {
      return -1;
    }
  }
};

template <typename scalar_t, typename Mode>
struct BinaryForwardOpDispatcher<BinaryOpType::BITWISE, scalar_t, Mode> {
  static int invoke_same_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                             uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                             Mode mode, uint32_t outer_size, uint32_t inner_size,
                             uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return bitwise_forward_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                                mode, outer_size, inner_size, tile_size);
    } else {
      return -1;
    }
  }

    static int invoke_bcast_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                              uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                              Mode mode, uint32_t outer_size, uint32_t inner_size,
                              uint32_t tile_size, const std::vector<int>& self_shape,
                              const std::vector<int>& other_shape) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return bitwise_bcast_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else {
      return -1;
    }
  }
};

template <typename scalar_t, typename Mode>
struct BinaryForwardOpDispatcher<BinaryOpType::ATAN2, scalar_t, Mode> {
  static int invoke_same_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                             uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                             Mode mode, uint32_t outer_size, uint32_t inner_size,
                             uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return atan2_forward_fp32(stream, ppl_module, output_addr, input_addr, other_addr,
                                mode, outer_size, inner_size, tile_size);
    } else {
      return -1;
    }
  }

    static int invoke_bcast_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                              uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                              Mode mode, uint32_t outer_size, uint32_t inner_size,
                              uint32_t tile_size, const std::vector<int>& self_shape,
                              const std::vector<int>& other_shape) {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return atan2_bcast_fp32(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else {
      return -1;
    }
  }
};

template <typename scalar_t, typename Mode>
struct BinaryForwardOpDispatcher<BinaryOpType::POW, scalar_t, Mode> {
  static int invoke_same_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                             uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                             Mode mode, uint32_t outer_size, uint32_t inner_size,
                             uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return pow_forward_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                                mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      return pow_forward_fp32(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return pow_forward_fp16(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return pow_forward_bf16(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size);
    }
    return -1;
  }

  static int invoke_bcast_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                              uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                              Mode mode, uint32_t outer_size, uint32_t inner_size,
                              uint32_t tile_size, const std::vector<int>& self_shape,
                              const std::vector<int>& other_shape) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return pow_bcast_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                              mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      return pow_bcast_fp32(stream, ppl_module, output_addr, input_addr, other_addr,
                             mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return pow_bcast_fp16(stream, ppl_module, output_addr, input_addr, other_addr,
                             mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return pow_bcast_bf16(stream, ppl_module, output_addr, input_addr, other_addr,
                             mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    }
    return -1;
  }
};

template <typename scalar_t, typename Mode>
struct BinaryForwardOpDispatcher<BinaryOpType::COMPARE, scalar_t, Mode> {
  static int invoke_same_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                             uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                             Mode mode, uint32_t outer_size, uint32_t inner_size,
                             uint32_t tile_size) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return compare_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                                 mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      return compare_fp32(stream, ppl_module, output_addr, input_addr, other_addr,
                                mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return compare_fp16(stream, ppl_module, output_addr, input_addr, other_addr,
                                mode, outer_size, inner_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return compare_bf16(stream, ppl_module, output_addr, input_addr, other_addr,
                                mode, outer_size, inner_size, tile_size);
    }
    return -1;
  }

  static int invoke_bcast_shape(tpuStream_t stream, tpuKernelModule_t ppl_module,
                              uint64_t output_addr, uint64_t input_addr, uint64_t other_addr,
                              Mode mode, uint32_t outer_size, uint32_t inner_size,
                              uint32_t tile_size, const std::vector<int>& self_shape,
                              const std::vector<int>& other_shape) {
    if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return compare_bcast_int32(stream, ppl_module, output_addr, input_addr, other_addr,
                               mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, float>) {
      return compare_bcast_fp32(stream, ppl_module, output_addr, input_addr, other_addr,
                              mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return compare_bcast_fp16(stream, ppl_module, output_addr, input_addr, other_addr,
                              mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return compare_bcast_bf16(stream, ppl_module, output_addr, input_addr, other_addr,
                              mode, outer_size, inner_size, tile_size,
                              self_shape[0], self_shape[1], self_shape[2], self_shape[3],
                              other_shape[0], other_shape[1], other_shape[2], other_shape[3]);
    }
    return -1;
  }
};

// 统一非常量操作实现
template <BinaryOpType Op, typename scalar_t, typename Mode = void>
static void unified_forward_impl(
    uint64_t output_addr,
    uint64_t input_addr,
    uint64_t other_addr,
    Mode mode,
    uint32_t outer_size,
    uint32_t inner_size,
    std::vector<int> self_shape,
    std::vector<int> other_shape
    )
    {
    auto is_same_shape = [](const std::vector<int>& shape1, const std::vector<int>& shape2) -> bool {
        if (shape1.size() != shape2.size()) {
            return false;
        }
        for (size_t i = 0; i < shape1.size(); ++i) {
            if (shape1[i] != shape2[i]) {
                return false;
            }
        }
        return true;
    };

    if (is_same_shape(self_shape, other_shape)){
      auto kernel = [&](tpuStream_t stream, tpuKernelModule_t ppl_module,
                      uint32_t tile_size) -> int {
        return BinaryForwardOpDispatcher<Op, scalar_t, Mode>::invoke_same_shape(
            stream, ppl_module, output_addr, input_addr, other_addr,
            mode, outer_size, inner_size, tile_size);
      };

      tpuStream_t stream = c10_tpu::getCurrentTPUStream().stream();
      tpuKernelModule_t ppl_module = getPplModule();
      uint32_t tile_size = inner_size;

      while (tile_size >= 1) {
        int ret = kernel(stream, ppl_module, tile_size);
        if (ret == 0) {
            return;
        } else {
            tile_size = tile_size / 2;
            continue;
        }
      }

    } else{
      std::vector<int> processed_self = self_shape;
      std::vector<int> processed_other = other_shape;
      process_shapes(processed_self, processed_other);

      auto kernel = [&](tpuStream_t stream, tpuKernelModule_t ppl_module,
                      uint32_t tile_size) -> int {
        return BinaryForwardOpDispatcher<Op, scalar_t, Mode>::invoke_bcast_shape(
            stream, ppl_module, output_addr, input_addr, other_addr,
            mode, outer_size, inner_size, tile_size, processed_self, processed_other);
      };

      tpuStream_t stream = c10_tpu::getCurrentTPUStream().stream();
      tpuKernelModule_t ppl_module = getPplModule();
      uint32_t tile_size = std::max(processed_self[0], processed_other[0]);

      while (tile_size >= 1) {
        int ret = kernel(stream, ppl_module, tile_size);
        if (ret == 0) {
            return;
        } else {
            tile_size = tile_size / 2;
            continue;
        }
      }
    }

    TORCH_CHECK(false, "unified_forward_impl failed!");
}
#endif

namespace at {

Tensor &binary_op_tpu(const Tensor &self, const Tensor &other,
                      const Scalar &alpha, Tensor &out, int binary_type) {
  if (out.numel() == 0) return out;
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(out);
  TORCH_CHECK( out.scalar_type()   != ScalarType::Long );

  if ( other.dim() == 0 && IS_CPU_TENSOR(other) ) {
    CHECK_TENSOR_IN_DEVICE(self);
#ifdef USING_PPL
    int outer_size = 1;
    for (const auto i : c10::irange(self.dim())) {
        outer_size *= self.size(i);
    }
    AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "constbinary", ([&] {
      using scalar_t_ = scalar_t;
      binary_const_impl<scalar_t_>(
      reinterpret_cast<uint64_t>(out.data_ptr()),
      reinterpret_cast<uint64_t>(self.data_ptr()),
      reinterpret_cast<uint64_t>(other.data_ptr()),
      binary_type,
      outer_size,
      other.item().toFloat() * alpha.toFloat(), 1
      );
      }));
#else
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnBinaryCAsync(
        stream, tpu::TPUGenerateTpudnnTensor(stream, self),
        other.item().toFloat() * alpha.toFloat(),
        tpu::TPUGenerateTpudnnTensor(stream, out), binary_type, 0);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  }
  else if ( self.dim() == 0 && IS_CPU_TENSOR(self) ) {
    CHECK_TENSOR_IN_DEVICE(other);
#ifdef USING_PPL
    int outer_size = 1;
    for (const auto i : c10::irange(other.dim())) {
        outer_size *= other.size(i);
    }
    AT_DISPATCH_FLOAT_INT_TYPES(other.scalar_type(), "binaryconst", ([&] {
      using scalar_t_ = scalar_t;
      binary_const_impl<scalar_t_>(
      reinterpret_cast<uint64_t>(out.data_ptr()),
      reinterpret_cast<uint64_t>(other.data_ptr()),
      reinterpret_cast<uint64_t>(self.data_ptr()),
      binary_type,
      outer_size,
      self.item().toFloat() * alpha.toFloat(), 0
      );
      }));
#else
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnBinaryCAsync(
        stream, tpu::TPUGenerateTpudnnTensor(stream, other),
        self.item().toFloat() * alpha.toFloat(),
        tpu::TPUGenerateTpudnnTensor(stream, out), binary_type, 1);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  }
  else if (tpu::TPUIsSameShape(self, other)) {
    CHECK_TENSOR_IN_DEVICE(self);
    CHECK_TENSOR_IN_DEVICE(other);
    TORCH_CHECK( self.scalar_type()  != ScalarType::Long );
    TORCH_CHECK( other.scalar_type() != ScalarType::Long );
#ifdef USING_PPL
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(self.dim()-1)) {
        outer_size *= self.size(i);
    }
    uint32_t inner_size = self.size(self.dim()-1);

    AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "binaryasync", ([&] {
      using scalar_t_ = scalar_t;
      binary_async_impl<scalar_t_>(
      reinterpret_cast<uint64_t>(out.data_ptr()),
      (self.scalar_type() == out.scalar_type() || binary_type == 3) ?
        reinterpret_cast<uint64_t>(self.data_ptr()) : reinterpret_cast<uint64_t>(other.data_ptr()),
      (self.scalar_type() == out.scalar_type() || binary_type == 3) ?
        reinterpret_cast<uint64_t>(other.data_ptr()): reinterpret_cast<uint64_t>(self.data_ptr()),
      binary_type,
      outer_size, inner_size
      );
      }));
#else
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnBinaryAsync(
        stream,
        (self.scalar_type() == out.scalar_type() || binary_type == 3) ?
          tpu::TPUGenerateTpudnnTensor(stream, self) : tpu::TPUGenerateTpudnnTensor(stream, other),
        (self.scalar_type() == out.scalar_type() || binary_type == 3) ?
          tpu::TPUGenerateTpudnnTensor(stream, other) : tpu::TPUGenerateTpudnnTensor(stream, self),
        alpha.toFloat(),
        tpu::TPUGenerateTpudnnTensor(stream, out), binary_type);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  }
  else {
    CHECK_TENSOR_IN_DEVICE(self);
    CHECK_TENSOR_IN_DEVICE(other);
    TORCH_CHECK( self.scalar_type()  != ScalarType::Long );
    TORCH_CHECK( other.scalar_type() != ScalarType::Long );

    int self_dim = self.dim(), other_dim = other.dim();
    int max_dim = std::max(self_dim, other_dim);
    std::vector<int> self_shape(max_dim), other_shape(max_dim);
    for (int i = max_dim - 1; i >= 0; i--) {
      if (i >= max_dim - self_dim) {
        self_shape[i] = self.size(i + self_dim - max_dim);
      } else {
        self_shape[i] = 1;
      }
      if (i >= max_dim - other_dim) {
        other_shape[i] = other.size(i + other_dim - max_dim);
      } else {
        other_shape[i] = 1;
      }
    }
    for (int i = 0; i < max_dim; i++) {
      TORCH_CHECK(self_shape[i] == other_shape[i] || self_shape[i] == 1 ||
                      other_shape[i] == 1,
                  "The size of tensor a (%d) must match the size of tensor b "
                  "(%d) at non-signleton dimension %d",
                  self_shape[i], other_shape[i], i)
    }

#ifdef USING_PPL
    auto A = self.contiguous();
    auto B = other.contiguous();
    auto O = out.contiguous();

    AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "binarybcast", ([&] {
    using scalar_t_ = scalar_t;
    binary_bcast_impl<scalar_t_>(
        reinterpret_cast<uint64_t>(O.data_ptr()),
        (self.scalar_type() == out.scalar_type() || binary_type == 3)?
          reinterpret_cast<uint64_t>(A.data_ptr()) : reinterpret_cast<uint64_t>(B.data_ptr()),
        (self.scalar_type() == out.scalar_type() || binary_type == 3)?
          reinterpret_cast<uint64_t>(B.data_ptr()) : reinterpret_cast<uint64_t>(A.data_ptr()),
        binary_type,
        self_shape,
        other_shape
      );
    }));
#else
    auto stream = c10_tpu::getCurrentTPUStream();
    auto other_t = tpu::TPUGenerateTpudnnTensor(stream, other);
    auto self_t  = tpu::TPUGenerateTpudnnTensor(stream, self);
    auto out_t   = tpu::TPUGenerateTpudnnTensor(stream, out);

    if (self_dim != other_dim) {
      auto &change_t = self_dim > other_dim ? other_t : self_t;
      const auto &change_shape =
          self_dim > other_dim ? other_shape : self_shape;
      for (int i = max_dim - 1; i >= 0; i--) {
        change_t.shape[i] = change_shape[i];
        change_t.stride[i] =
            i == max_dim - 1 ? 1 : change_t.stride[i + 1] * change_shape[i + 1];
      }
      change_t.dim = max_dim;
    }

    auto status = tpudnnBinaryBcastAsync(stream,
                        (self.scalar_type() == out.scalar_type() || binary_type == 3) ? self_t  : other_t,
                        (self.scalar_type() == out.scalar_type() || binary_type == 3) ? other_t : self_t,
                                   alpha.toFloat(), out_t, binary_type);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  }
  TIMING_END;
  return out;
}

#define CONTIGUOUS_FALLBACK(OP_FUNC, self, other, out, alpha)                    \
do {                                                                              \
  if (!(self).is_contiguous() || !(other).is_contiguous() || !(out).is_contiguous())\
  {                                                                               \
    CONTIGUOUS_WARNING();                                                         \
    if ((out).is_contiguous()) {                                                  \
      (out) = OP_FUNC((self).contiguous(), (other).contiguous(), (alpha));        \
    } else {                                                                      \
      auto out_ = OP_FUNC((self).contiguous(), (other).contiguous(), (alpha));    \
      auto handle = c10_tpu::getCurrentTPUStream();                               \
      tpudnnStridedCopyAsync(                                                     \
          handle,                                                                 \
          tpu::TPUGenerateTpudnnTensor(handle, out_),                             \
          tpu::TPUGenerateTpudnnTensor(handle, (out)));                           \
    }                                                                             \
    SHOW_TENSOR_OP((self), (other), (out));                                       \
    return (out);                                                                 \
  }                                                                               \
} while (0)

#define CONTIGUOUS_FALLBACK_v2(OP_FUNC, self, other, out)                       \
do {                                                                              \
  if (!(self).is_contiguous() || !(other).is_contiguous() || !(out).is_contiguous())\
  {                                                                               \
    CONTIGUOUS_WARNING();                                                         \
    if ((out).is_contiguous()) {                                                  \
      (out) = OP_FUNC((self).contiguous(), (other).contiguous());                 \
    } else {                                                                      \
      auto out_ = OP_FUNC((self).contiguous(), (other).contiguous());             \
      auto handle = c10_tpu::getCurrentTPUStream();                               \
      tpudnnStridedCopyAsync(                                                     \
          handle,                                                                 \
          tpu::TPUGenerateTpudnnTensor(handle, out_),                             \
          tpu::TPUGenerateTpudnnTensor(handle, (out)));                           \
    }                                                                             \
    SHOW_TENSOR_OP((self), (other), (out));                                       \
    return (out);                                                                 \
  }                                                                               \
} while (0)

// pattern1: self.shape == other.shape = out.shape, 
//           self.stride[0] != other.stride[0]
#define PATTERN1(OP_FUNC, self, other, out, alpha)                              \
do {                                                                            \
  if (other.is_contiguous() && out.is_contiguous() && !self.is_contiguous() &&  \
    tpu::TPUIsSameShapeWithstride(self, other))                                 \
  {                                                                             \
    TIMING_START;                                                               \
    int binary_type = -1;                                                       \
    if ( strcmp(#OP_FUNC, "add") == 0 ) binary_type = 0;                        \
    else if ( strcmp(#OP_FUNC, "sub") == 0 ) binary_type = 1;                   \
    else TORCH_CHECK( false, "only support add/sub");                           \
    auto stream = c10_tpu::getCurrentTPUStream();                               \
    auto status = tpudnnBinaryWStrideAsync(                                            \
        stream,                                                                 \
        tpu::TPUGenerateTpudnnTensor(stream, self),                             \
        tpu::TPUGenerateTpudnnTensor(stream, other),                            \
        alpha.toFloat(),                                                        \
        tpu::TPUGenerateTpudnnTensor(stream, out), binary_type);                \
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);                                \
    TIMING_END;                                                                  \
    return out;                                                                  \
  }                                                                              \
}while (0)                                                                      

Tensor &add_out_tpu(const Tensor &self, const Tensor &other,
                    const Scalar &alpha, Tensor &out) {
  PATTERN1(add, self, other, out, alpha);
  CONTIGUOUS_FALLBACK(add, self, other, out, alpha);
  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self, other, alpha, out, 0);
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("add.out", add_out_tpu); }

Tensor &sub_out_tpu(const Tensor &self, const Tensor &other,
                    const Scalar &alpha, Tensor &out) {
  PATTERN1(sub, self, other, out, alpha);
  CONTIGUOUS_FALLBACK(sub, self, other, out, alpha);
  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self, other, alpha, out, 1);
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("sub.out", sub_out_tpu); }

Tensor &mul_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  CONTIGUOUS_FALLBACK_v2(mul, self, other, out);
  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self, other, 1, out, 2);
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("mul.out", mul_out_tpu); }

Tensor &div_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  CONTIGUOUS_FALLBACK_v2(div, self, other, out);
  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self, other, 1, out, 3);
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("div.out", div_out_tpu); }

/* ******************************************************************************************** */
#ifdef USING_PPL
template <typename Mode>
void ppl_compare_impl(const Tensor& self, const Tensor& other, Tensor& out,
                     Mode mode, uint32_t outer_size, uint32_t inner_size) {
  AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "binary_forward", ([&] {
    if (other.dim() == 0) {
      unified_const_impl<BinaryOpType::COMPARE, scalar_t, Mode>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        other.item().toFloat(),
        mode,
        outer_size,
        inner_size
      );
    } else if (self.dim() == 0) {
      unified_const_impl<BinaryOpType::COMPARE, scalar_t, Mode>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(other.data_ptr()),
        self.item().toFloat(),
        mode,
        outer_size,
        inner_size
      );
    } else {
      int self_dim = self.dim();
      std::vector<int> self_shape(self_dim);
      for (int i = self_dim - 1; i >= 0; i--) {
        self_shape[i] = self.size(i);
      }

      int other_dim = other.dim();
      std::vector<int> other_shape(other_dim);
      for (int i = other_dim - 1; i >= 0; i--) {
        other_shape[i] = other.size(i);
      }

      unified_forward_impl<BinaryOpType::COMPARE, scalar_t, Mode>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        reinterpret_cast<uint64_t>(other.data_ptr()),
        mode,
        outer_size,
        inner_size,
        self_shape,
        other_shape
      );
    }
  }));
}

template <typename Mode>
void ppl_shift_impl(const Tensor& self, const Tensor& other, Tensor& out,
                   Mode mode, uint32_t outer_size, uint32_t inner_size) {
  AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "shift_const", ([&] {
    if (other.dim() == 0) {
      scalar_t other_val = (mode == 0) ?
                    -other.item().to<scalar_t>() :
                      other.item().to<scalar_t>();
      unified_const_impl<BinaryOpType::SHIFT, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        other_val,
        0,
        outer_size,
        inner_size
      );
    } else if (self.dim() == 0) {
      scalar_t self_val = (mode == 0) ?
                    -self.item().to<scalar_t>() :
                      self.item().to<scalar_t>();
      unified_const_impl<BinaryOpType::SHIFT, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(other.data_ptr()),
        self_val,
        0,
        outer_size,
        inner_size
      );
    }  else {
      int self_dim = self.dim();
      std::vector<int> self_shape(self_dim);
      for (int i = self_dim - 1; i >= 0; i--) {
        self_shape[i] = self.size(i);
      }

      int other_dim = other.dim();
      std::vector<int> other_shape(other_dim);
      for (int i = other_dim - 1; i >= 0; i--) {
        other_shape[i] = other.size(i);
      }

      unified_forward_impl<BinaryOpType::SHIFT, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        reinterpret_cast<uint64_t>(other.data_ptr()),
        0,
        outer_size,
        inner_size,
        self_shape,
        other_shape
      );
    }
  }));
}

template <typename Mode>
void ppl_minmax_impl(const Tensor& self, const Tensor& other, Tensor& out,
                   Mode mode, uint32_t outer_size, uint32_t inner_size) {
  AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "minmax_const", ([&] {
    if (other.dim() == 0) {
      unified_const_impl<BinaryOpType::MINMAX, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        other.item().toFloat(),
        mode,
        outer_size,
        inner_size);
    } else if (self.dim() == 0) {
      unified_const_impl<BinaryOpType::MINMAX, scalar_t, bool>(
          reinterpret_cast<uint64_t>(out.data_ptr()),
          reinterpret_cast<uint64_t>(other.data_ptr()),
          self.item().toFloat(),
          mode,
          outer_size,
          inner_size);
    } else {
      int self_dim = self.dim();
      std::vector<int> self_shape(self_dim);
      for (int i = self_dim - 1; i >= 0; i--) {
        self_shape[i] = self.size(i);
      }

      int other_dim = other.dim();
      std::vector<int> other_shape(other_dim);
      for (int i = other_dim - 1; i >= 0; i--) {
        other_shape[i] = other.size(i);
      }

      unified_forward_impl<BinaryOpType::MINMAX, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        reinterpret_cast<uint64_t>(other.data_ptr()),
        mode,
        outer_size,
        inner_size,
        self_shape,
        other_shape
      );
    }
  }));
}

template <typename Mode>
void ppl_bitwise_impl(const Tensor& self, const Tensor& other, Tensor& out,
                   Mode mode, uint32_t outer_size, uint32_t inner_size) {
  AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "bitwise", ([&] {
    if (other.dim() == 0) {
      unified_const_impl<BinaryOpType::BITWISE, scalar_t, Mode>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        other.item().toFloat(),
        mode,
        outer_size,
        inner_size);
    } else if (self.dim() == 0) {
      unified_const_impl<BinaryOpType::BITWISE, scalar_t, Mode>(
          reinterpret_cast<uint64_t>(out.data_ptr()),
          reinterpret_cast<uint64_t>(other.data_ptr()),
          self.item().toFloat(),
          mode,
          outer_size,
          inner_size);
    } else {
      int self_dim = self.dim();
      std::vector<int> self_shape(self_dim);
      for (int i = self_dim - 1; i >= 0; i--) {
        self_shape[i] = self.size(i);
      }

      int other_dim = other.dim();
      std::vector<int> other_shape(other_dim);
      for (int i = other_dim - 1; i >= 0; i--) {
        other_shape[i] = other.size(i);
      }

      unified_forward_impl<BinaryOpType::BITWISE, scalar_t, Mode>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        reinterpret_cast<uint64_t>(other.data_ptr()),
        mode,
        outer_size,
        inner_size,
        self_shape,
        other_shape
      );
    }
  }));
}

template <typename Mode>
void ppl_pow_impl(const Tensor& self, const Tensor& other, Tensor& out,
                   Mode mode, uint32_t outer_size, uint32_t inner_size) {
  AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "pow", ([&] {
    int self_dim = self.dim();
    std::vector<int> self_shape(self_dim);
    for (int i = self_dim - 1; i >= 0; i--) {
      self_shape[i] = self.size(i);
    }

    int other_dim = other.dim();
    std::vector<int> other_shape(other_dim);
    for (int i = other_dim - 1; i >= 0; i--) {
      other_shape[i] = other.size(i);
    }

    unified_forward_impl<BinaryOpType::POW, scalar_t, bool>(
      reinterpret_cast<uint64_t>(out.data_ptr()),
      reinterpret_cast<uint64_t>(self.data_ptr()),
      reinterpret_cast<uint64_t>(other.data_ptr()),
      mode,
      outer_size,
      inner_size,
      self_shape,
      other_shape
    );
  }));
}

template <typename Mode, typename UnaryFunc, typename BinaryFunc, typename CpuFunc, typename PplFunc>
void binary_impl(const Tensor &self, const Tensor &other, Tensor &out,
                 Mode mode, UnaryFunc unary_func,
                 BinaryFunc binary_func, CpuFunc cpu_func,
                 PplFunc ppl_func) {
  if (out.numel() == 0) return;
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(out);
  if ( other.dim() == 0 && IS_CPU_TENSOR(other) )
  {
    CHECK_TENSOR_IN_DEVICE(self);
    TORCH_CHECK( self.scalar_type()  != ScalarType::Long );
#ifdef USING_PPL
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(self.dim()-1)) {
        outer_size *= self.size(i);
    }
    uint32_t inner_size = self.size(self.dim()-1);
    ppl_func(self, other, out, mode, outer_size, inner_size);
#else
    const auto handle = c10_tpu::getCurrentTPUStream();
    auto status = unary_func(handle, self, other, out);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  }
  else if (self.dim() == 0 && IS_CPU_TENSOR(self))
  {
    CHECK_TENSOR_IN_DEVICE(other);
    TORCH_CHECK( other.scalar_type()  != ScalarType::Long );
#ifdef USING_PPL
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(other.dim()-1)) {
        outer_size *= other.size(i);
    }
    uint32_t inner_size = other.size(other.dim()-1);
    ppl_func(other, self, out, mode, outer_size, inner_size);
#else
    const auto handle = c10_tpu::getCurrentTPUStream();
    auto status = unary_func(handle, self, other, out);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  }
  else {
    CHECK_TENSOR_IN_DEVICE(self);
    CHECK_TENSOR_IN_DEVICE(other);
    TORCH_CHECK( self.scalar_type()  != ScalarType::Long );
    TORCH_CHECK( other.scalar_type() != ScalarType::Long );
#ifdef USING_PPL
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(self.dim()-1)) {
        outer_size *= self.size(i);
    }
    uint32_t inner_size = self.size(self.dim()-1);
    int self_dim = self.dim();
    std::vector<int> self_shape(self_dim);
    for (int i = self_dim - 1; i >= 0; i--) {
      self_shape[i] = self.size(i);
    }
    int other_dim = other.dim();
    std::vector<int> other_shape(other_dim);
    for (int i = other_dim - 1; i >= 0; i--) {
      other_shape[i] = other.size(i);
    }
    ppl_func(self, other, out, mode, outer_size, inner_size);
#else
    const auto handle = c10_tpu::getCurrentTPUStream();
    auto status = binary_func(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self),
        tpu::TPUGenerateTpudnnTensor(handle, other),
        tpu::TPUGenerateTpudnnTensor(handle, out),
        mode);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  }
  TIMING_END;
}
#endif
/* ******************************************************************************************** */
template <typename Mode, typename UnaryFunc, typename BinaryFunc, typename CpuFunc>
void binary_impl(const Tensor &self, const Tensor &other, Tensor &out,
                 Mode mode, UnaryFunc unary_func,
                 BinaryFunc binary_func, CpuFunc cpu_func) {
  if (out.numel() == 0) return;
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(out);
  const auto handle = c10_tpu::getCurrentTPUStream();
  if ( other.dim() == 0 && IS_CPU_TENSOR(other) )
  {
    CHECK_TENSOR_IN_DEVICE(self);
    TORCH_CHECK( self.scalar_type()  != ScalarType::Long );
    auto status = unary_func(handle, self, other, out);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
  else if (self.dim() == 0 && IS_CPU_TENSOR(self))
  {
    CHECK_TENSOR_IN_DEVICE(other);
    TORCH_CHECK( other.scalar_type()  != ScalarType::Long );
    auto status = unary_func(handle, self, other, out);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
  else {
    CHECK_TENSOR_IN_DEVICE(self);
    CHECK_TENSOR_IN_DEVICE(other);
    TORCH_CHECK( self.scalar_type()  != ScalarType::Long );
    TORCH_CHECK( other.scalar_type() != ScalarType::Long );
    auto status = binary_func(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self),
        tpu::TPUGenerateTpudnnTensor(handle, other),
        tpu::TPUGenerateTpudnnTensor(handle, out),
        mode);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
  TIMING_END;
}

template <typename UnaryFunc, typename BinaryFunc, typename CpuFunc>
void binary_impl(const Tensor &self, const Tensor &other, Tensor &out,
                UnaryFunc unary_func, BinaryFunc binary_func,
                 CpuFunc cpu_func) {
  if (out.numel() == 0) return;
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(out);
  const auto handle = c10_tpu::getCurrentTPUStream();
 if ( (self.dim() == 0 && IS_CPU_TENSOR(self)) || (other.dim() == 0 && IS_CPU_TENSOR(other)) ) {
    if ( self.dim() == 0  ) CHECK_TENSOR_IN_DEVICE(other);
    if ( other.dim() == 0 ) CHECK_TENSOR_IN_DEVICE(self);
    TORCH_CHECK( other.scalar_type()  != ScalarType::Long );

    auto status = unary_func(handle, self, other, out);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  } else {
    CHECK_TENSOR_IN_DEVICE(self);
    CHECK_TENSOR_IN_DEVICE(other);
    TORCH_CHECK( self.scalar_type()  != ScalarType::Long );
    TORCH_CHECK( other.scalar_type() != ScalarType::Long );
    auto status = binary_func(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self),
        tpu::TPUGenerateTpudnnTensor(handle, other),
        tpu::TPUGenerateTpudnnTensor(handle, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
  TIMING_END;
}
/* ******************************************************************************************** */
Tensor &bitwise_xor_out_tpu(const Tensor &self, const Tensor &other,
                            Tensor &out) {
    if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
      CONTIGUOUS_WARNING();
      if (out.is_contiguous()) {
        out = bitwise_xor(self.contiguous(), other.contiguous());
      } else {
        auto out_ = bitwise_xor(self.contiguous(), other.contiguous());
        auto handle = c10_tpu::getCurrentTPUStream();
        tpudnnStridedCopyAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, out_),
            tpu::TPUGenerateTpudnnTensor(handle, out));
      }
      SHOW_TENSOR_OP(self, other, out);
      return out;
    }

    binary_impl(
      self, other, out,
      BitwiseMode_t::XOR,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other,
         Tensor &out) {
        return tpudnnBitwiseConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toInt() : other.item().toInt(),
            BitwiseMode_t::XOR);
      },
      tpudnnBitwiseAsync,
      [](const Tensor &a, const Tensor &b) { return bitwise_xor(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, BitwiseMode_t mode,
      uint32_t outer_size, uint32_t inner_size) {
      AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "bitwise_xor", ([&] {
          ppl_bitwise_impl<BitwiseMode_t>(self, other, out, mode, outer_size, inner_size);
      }));
      }
#endif
    );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_xor.Tensor_out", bitwise_xor_out_tpu);
}

Tensor &bitwise_and_out_tpu(const Tensor &self, const Tensor &other,
                            Tensor &out) {
    if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
      CONTIGUOUS_WARNING();
      if (out.is_contiguous()) {
        out = bitwise_and(self.contiguous(), other.contiguous());
      } else {
        auto out_ = bitwise_and(self.contiguous(), other.contiguous());
        auto handle = c10_tpu::getCurrentTPUStream();
        tpudnnStridedCopyAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, out_),
            tpu::TPUGenerateTpudnnTensor(handle, out));
      }
      SHOW_TENSOR_OP(self, other, out);
      return out;
    }

    binary_impl(
      self, other, out,
      BitwiseMode_t::AND,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnBitwiseConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toInt() : other.item().toInt(),
            BitwiseMode_t::AND);
      },
      tpudnnBitwiseAsync,
      [](const Tensor &a, const Tensor &b) { return bitwise_and(a, b); }
#ifdef USING_PPL
      ,[](const Tensor &self, const Tensor &other, Tensor &out, BitwiseMode_t mode,
      uint32_t outer_size, uint32_t inner_size) {
      AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "bitwise_and", ([&] {
          ppl_bitwise_impl<BitwiseMode_t>(self, other, out, mode, outer_size, inner_size);
      }));}
#endif
      );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_and.Tensor_out", bitwise_and_out_tpu);
}

Tensor &bitwise_or_out_tpu(const Tensor &self, const Tensor &other,
                           Tensor &out) {
    if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
      CONTIGUOUS_WARNING();
      if (out.is_contiguous()) {
        out = bitwise_or(self.contiguous(), other.contiguous());
      } else {
        auto out_ = bitwise_or(self.contiguous(), other.contiguous());
        auto handle = c10_tpu::getCurrentTPUStream();
        tpudnnStridedCopyAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, out_),
            tpu::TPUGenerateTpudnnTensor(handle, out));
      }
      SHOW_TENSOR_OP(self, other, out);
      return out;
    }

    binary_impl(
      self, other, out, BitwiseMode_t::OR,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other,
         Tensor &out) {
        return tpudnnBitwiseConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toInt() : other.item().toInt(),
            BitwiseMode_t::OR);
      },
      tpudnnBitwiseAsync,
      [](const Tensor &a, const Tensor &b) { return bitwise_or(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, BitwiseMode_t mode,
      uint32_t outer_size, uint32_t inner_size) {
      AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "bitwise_or", ([&] {
          ppl_bitwise_impl<BitwiseMode_t>(self, other, out, mode, outer_size, inner_size);
      }));}
#endif
      );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_or.Tensor_out", bitwise_or_out_tpu);
}

Tensor &equal_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = eq(self.contiguous(), other.contiguous());
    } else {
      auto out_ = eq(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  binary_impl(
      self, other, out, CompareMode_t::TPUDNN_EQ,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_EQ, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return eq(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, CompareMode_t mode,
          uint32_t outer_size, uint32_t inner_size) {
          AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "equal", ([&] {
              ppl_compare_impl<CompareMode_t>(self, other, out, mode, outer_size, inner_size);
          }));}
#endif
      );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("eq.Tensor_out", equal_out_tpu); }

Tensor &greater_or_equal_out_tpu(const Tensor &self, const Tensor &other,
                                 Tensor &out) {
    if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
      CONTIGUOUS_WARNING();
      if (out.is_contiguous()) {
        out = ge(self.contiguous(), other.contiguous());
      } else {
        auto out_ = ge(self.contiguous(), other.contiguous());
        auto handle = c10_tpu::getCurrentTPUStream();
        tpudnnStridedCopyAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, out_),
            tpu::TPUGenerateTpudnnTensor(handle, out));
      }
      SHOW_TENSOR_OP(self, other, out);
      return out;
    }

    binary_impl(
      self, other, out,
      CompareMode_t::TPUDNN_GE,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_GE, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return ge(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, CompareMode_t mode,
          uint32_t outer_size, uint32_t inner_size) {
          AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "ge", ([&] {
              ppl_compare_impl<CompareMode_t>(self, other, out, mode, outer_size, inner_size);
          }));}
#endif
      );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ge.Tensor_out", greater_or_equal_out_tpu);
}

Tensor &greater_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = gt(self.contiguous(), other.contiguous());
    } else {
      auto out_ = gt(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  binary_impl(
      self, other, out, CompareMode_t::TPUDNN_GT,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_GT, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return gt(a, b); }
#ifdef USING_PPL
      ,[](const Tensor &self, const Tensor &other, Tensor &out, CompareMode_t mode,
          uint32_t outer_size, uint32_t inner_size) {
          ppl_compare_impl<CompareMode_t>(self, other, out, mode, outer_size, inner_size);
        }
#endif
      );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("gt.Tensor_out", greater_out_tpu); }

Tensor &less_than_or_equal_out_tpu(const Tensor &self, const Tensor &other,
                                   Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = le(self.contiguous(), other.contiguous());
    } else {
      auto out_ = le(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  binary_impl(
      self, other, out,
      CompareMode_t::TPUDNN_LE,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_LE, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return le(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, CompareMode_t mode,
          uint32_t outer_size, uint32_t inner_size) {
          AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "le", ([&] {
              ppl_compare_impl<CompareMode_t>(self, other, out, mode, outer_size, inner_size);
          }));}
#endif
      );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("le.Tensor_out", less_than_or_equal_out_tpu);
}

Tensor &less_than_out_tpu(const Tensor &self, const Tensor &other,
                          Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = lt(self.contiguous(), other.contiguous());
    } else {
      auto out_ = lt(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  binary_impl(
      self, other, out, CompareMode_t::TPUDNN_LT,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_LT, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return lt(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, CompareMode_t mode,
          uint32_t outer_size, uint32_t inner_size) {
          AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "lt", ([&] {
              ppl_compare_impl<CompareMode_t>(self, other, out, mode, outer_size, inner_size);
          }));}
#endif
      );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("lt.Tensor_out", less_than_out_tpu); }

Tensor &not_equal_out_tpu(const Tensor &self, const Tensor &other,
                          Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = ne(self.contiguous(), other.contiguous());
    } else {
      auto out_ = ne(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  binary_impl(
      self, other, out, CompareMode_t::TPUDNN_NE,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_NE, 0);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return ne(a, b); }
#ifdef USING_PPL
    , [](const Tensor &self, const Tensor &other, Tensor &out, CompareMode_t mode,
          uint32_t outer_size, uint32_t inner_size) {
          AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "ne", ([&] {
              ppl_compare_impl<CompareMode_t>(self, other, out, mode, outer_size, inner_size);
          }));
        }
#endif
    );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("ne.Tensor_out", not_equal_out_tpu); }

Tensor &shift_left_out_tpu(const Tensor &self, const Tensor &other,
                           Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = bitwise_left_shift(self.contiguous(), other.contiguous());
    } else {
      auto out_ = bitwise_left_shift(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  binary_impl(
      self, other, out, true,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnShiftConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toChar() : other.item().toChar(),
            true);
      },
      tpudnnShiftAsync,
      [](const Tensor &a, const Tensor &b) { TORCH_CHECK(0); return a; }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, bool mode,
    uint32_t outer_size, uint32_t inner_size) {
    AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "leftshift", ([&] {
        ppl_shift_impl<bool>(self, other, out, true, outer_size, inner_size);
    }));}
#endif
    );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_left_shift.Tensor_out", shift_left_out_tpu);
}

Tensor &shift_right_arithmetic_out_tpu(const Tensor &self, const Tensor &other,
                                       Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = bitwise_right_shift(self.contiguous(), other.contiguous());
    } else {
      auto out_ = bitwise_right_shift(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  binary_impl(
      self, other, out, false,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnShiftConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toChar() : other.item().toChar(),
            false);
      },
      tpudnnShiftAsync,
      [](const Tensor &a, const Tensor &b) { TORCH_CHECK(0);return a; }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, bool mode,
    uint32_t outer_size, uint32_t inner_size) {
    AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "rightshift", ([&] {
        ppl_shift_impl<bool>(self, other, out, false, outer_size, inner_size);
    }));}
#endif
    );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_right_shift.Tensor_out", shift_right_arithmetic_out_tpu);
}

Tensor &minimum_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = minimum(self.contiguous(), other.contiguous());
    } else {
      auto out_ = minimum(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  TORCH_CHECK(other.is_contiguous() && out.is_contiguous());
  binary_impl(
      self, other, out,
#ifdef USING_PPL
      true,
#endif
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnMinimumConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toFloat() : other.item().toFloat());
      },
      tpudnnMinimumAsync,
      [](const Tensor &a, const Tensor &b) { return minimum(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, bool mode,
      uint32_t outer_size, uint32_t inner_size) {
      AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "minimum", ([&] {
          ppl_minmax_impl<bool>(self, other, out, true, outer_size, inner_size);
      }));}
#endif
    );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("minimum.out", minimum_out_tpu); }

Tensor &maximum_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = maximum(self.contiguous(), other.contiguous());
    } else {
      auto out_ = maximum(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  binary_impl(
      self, other, out,
#ifdef USING_PPL
      false,
#endif
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnMaximumConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, (self.dim() == 0 && IS_CPU_TENSOR(self))  ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            (self.dim() == 0 && IS_CPU_TENSOR(self))  ? self.item().toFloat() : other.item().toFloat());
      },
      tpudnnMaximumAsync,
      [](const Tensor &a, const Tensor &b) { return maximum(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, bool mode,
      uint32_t outer_size, uint32_t inner_size) {
      AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "maximum", ([&] {
          ppl_minmax_impl<bool>(self, other, out, false, outer_size, inner_size);
      }));}
#endif
    );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("maximum.out", maximum_out_tpu); }

Tensor &fmax_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  return maximum_out_tpu(self, other, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("fmax.out", fmax_out_tpu); }

Tensor &fmin_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  return minimum_out_tpu(self, other, out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("fmin.out", fmin_out_tpu); }

Tensor &pow_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    CONTIGUOUS_WARNING();
    if (out.is_contiguous()) {
      out = pow(self.contiguous(), other.contiguous());
    } else {
      auto out_ = pow(self.contiguous(), other.contiguous());
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  binary_impl(
      self, other, out,
#ifdef USING_PPL
      false,
#endif
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other,
         Tensor &out) { return TPUDNN_STATUS_FAILED; },
      tpudnnPowerAsync,
      [](const Tensor &a, const Tensor &b) { return pow(a, b); }
#ifdef USING_PPL
      , [](const Tensor &self, const Tensor &other, Tensor &out, bool mode,
      uint32_t outer_size, uint32_t inner_size) {
      AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "pow", ([&] {
          ppl_pow_impl<bool>(self, other, out, false, outer_size, inner_size);
      }));}
#endif
    );
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("pow.Tensor_Tensor_out", pow_out_tpu);
}

Tensor &pow_c_out_tpu(const Tensor &self, const Scalar &exponent, Tensor &out) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
  TORCH_CHECK(exponent.toFloat() > 0);
  if (self.dim() == 0) {
    Tensor out_cpu = pow(self.cpu(), exponent);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    SHOW_TENSOR_OP(self, out);
    return out;
  }
  if (exponent.toFloat() == 1.0) {
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), self.data_ptr(), out.nbytes());
    SHOW_TENSOR_OP(self, out);
    return out;
  }
  TORCH_CHECK(exponent.toFloat() > 0);
#ifdef USING_PPL
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(self.dim()-1)) {
        outer_size *= self.size(i);
    }
    uint32_t inner_size = self.size(self.dim()-1);
    AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "powc_const", ([&] {
      unified_const_impl<BinaryOpType::POW, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        exponent.toFloat(),
        0,
        outer_size,
        inner_size,
        out.scalar_type() == at::kInt);
    }));
#else
  auto handle = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnPowerScalarAsync(
      handle,
      tpu::TPUGenerateTpudnnTensor(handle, self),
      tpu::TPUGenerateTpudnnTensor(handle, out),
      exponent.toFloat());
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("pow.Tensor_Scalar_out", pow_c_out_tpu);
}

Tensor &c_pow_out_tpu(const Scalar &self, const Tensor &exponent, Tensor &out) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(exponent);
  CHECK_TENSOR_IN_DEVICE(out);
  TORCH_CHECK(self.toFloat() > 0);
  if (exponent.dim() == 0) {
    Tensor out_cpu = pow(self, exponent.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    SHOW_TENSOR_OP(exponent, out);
    return out;
  }
#ifdef USING_PPL
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(exponent.dim()-1)) {
        outer_size *= exponent.size(i);
    }
    uint32_t inner_size = exponent.size(exponent.dim()-1);
    AT_DISPATCH_FLOAT_INT_TYPES(exponent.scalar_type(), "cpow_const", ([&] {
      unified_const_impl<BinaryOpType::POW, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(exponent.data_ptr()),
        self.toFloat(),
        1,
        outer_size,
        inner_size,
        out.scalar_type() == at::kInt);
    }));
#else
  auto handle = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnScalarPowerAsync(
      handle,
      tpu::TPUGenerateTpudnnTensor(handle, exponent),
      tpu::TPUGenerateTpudnnTensor(handle, out),
      self.toFloat());
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(exponent, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("pow.Scalar_out", c_pow_out_tpu); }


Tensor &atan2_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  TIMING_START;
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(other);
  // self is scalar and other is scalar
 if (self.dim() == 0) {
#ifdef USING_PPL
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(other.dim()-1)) {
        outer_size *= other.size(i);
    }
    uint32_t inner_size = other.size(other.dim()-1);
    AT_DISPATCH_FLOAT_INT_TYPES(other.scalar_type(), "atan2_const", ([&] {
      unified_const_impl<BinaryOpType::ATAN2, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(other.data_ptr()),
        self.item().toFloat(),
        1,
        outer_size,
        inner_size,
        out.scalar_type() == at::kInt);
    }));
#else
    auto handle = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnScalarAtan2Async(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, other.to(out.dtype())),
        tpu::TPUGenerateTpudnnTensor(handle, out),
        self.item().toFloat());
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  } else if (other.dim() == 0) {
#ifdef USING_PPL
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(self.dim()-1)) {
        outer_size *= self.size(i);
    }
    uint32_t inner_size = self.size(self.dim()-1);
    AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "const_atan2", ([&] {
      unified_const_impl<BinaryOpType::ATAN2, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        other.item().toFloat(),
        0,
        outer_size,
        inner_size,
        out.scalar_type() == at::kInt);
    }));
#else
    auto handle = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnAtan2ScalarAsync(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self.to(out.dtype())),
        tpu::TPUGenerateTpudnnTensor(handle, out),
        other.item().toFloat());
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  } else {
#ifdef USING_PPL
    int self_dim = self.dim();
    std::vector<int> self_shape(self_dim);
    for (int i = self_dim - 1; i >= 0; i--) {
      self_shape[i] = self.size(i);
    }

    int other_dim = other.dim();
    std::vector<int> other_shape(other_dim);
    for (int i = other_dim - 1; i >= 0; i--) {
      other_shape[i] = other.size(i);
    }
    uint32_t outer_size = 1;
    for (const auto i : c10::irange(self.dim()-1)) {
        outer_size *= self.size(i);
    }
    uint32_t inner_size = self.size(self.dim()-1);
    AT_DISPATCH_FLOAT_INT_TYPES(self.scalar_type(), "atan2", ([&] {
      unified_forward_impl<BinaryOpType::ATAN2, scalar_t, bool>(
        reinterpret_cast<uint64_t>(out.data_ptr()),
        reinterpret_cast<uint64_t>(self.data_ptr()),
        reinterpret_cast<uint64_t>(other.data_ptr()),
        0,
        outer_size,
        inner_size,
        self_shape,
        other_shape
      );
    }));
#else
    auto handle = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnAtan2Async(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self.to(torch::kFloat)),
        tpu::TPUGenerateTpudnnTensor(handle, other.to(torch::kFloat)),
        tpu::TPUGenerateTpudnnTensor(handle, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  }
  TIMING_END;
  SHOW_TENSOR_OP(self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("atan2.out", atan2_out_tpu); }
#if 0
Tensor &hypot_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (self.dim() >> 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = hypot(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;

    auto status = sgdnnHypotC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::HYPOT);
  } else if (other.dim() == 0) {
    TIMING_START;

    auto status = sgdnnHypotC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::HYPOT);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;

      auto status = sgdnnHypot(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::HYPOT);
    } else {
      TIMING_START;

      auto status = sgdnnHypotBcast(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::HYPOT);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("hypot.out", hypot_out_tpu); }

Tensor &nextafter_out_tpu(const Tensor &self, const Tensor &other,
                          Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(other);
  // self is scalar and other is scalar
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = nextafter(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat32).cpu();
    TIMING_START;

    auto status =
        sgdnnNextafterC(tpu::TPUGetDeviceResource(), *scalar.data_ptr<float>(),
                        tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::NEXTAFTER);
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat32).cpu();
    TIMING_START;

    auto status = sgdnnNextafter_C(
        tpu::TPUGetDeviceResource(),
        tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::NEXTAFTER);
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
      TIMING_START;

      auto status =
          sgdnnNextafter(tpu::TPUGetDeviceResource(),
                         tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::NEXTAFTER);
    } else {
      // The shapes of self and other are not the same, need to broadcast
      TIMING_START;

      auto status = sgdnnNextafterBcast(
          tpu::TPUGetDeviceResource(),
          tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
          tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::NEXTAFTER);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("nextafter.out", nextafter_out_tpu); }
#endif

Tensor &less_than_or_equal_scalar_out_tpu(const Tensor &self, const Scalar &other, Tensor &out) {
  return less_than_or_equal_out_tpu(self, torch::scalar_tensor(other), out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("le.Scalar_out", less_than_or_equal_scalar_out_tpu);
}

Tensor &less_than_scalar_out_tpu(const Tensor &self, const Scalar &other, Tensor &out) {
  return less_than_out_tpu(self, torch::scalar_tensor(other), out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("lt.Scalar_out", less_than_scalar_out_tpu);
}

Tensor &greater_or_equal_scalar_out_tpu(const Tensor &self, const Scalar &other, Tensor &out) {
  return greater_or_equal_out_tpu(self, torch::scalar_tensor(other), out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ge.Scalar_out", greater_or_equal_scalar_out_tpu);
}

Tensor &greater_scalar_out_tpu(const Tensor &self, const Scalar &other, Tensor &out) {
  return greater_out_tpu(self, torch::scalar_tensor(other), out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("gt.Scalar_out", greater_scalar_out_tpu);
}

Tensor &not_equal_scalar_out_tpu(const Tensor &self, const Scalar &other, Tensor &out) {
  return not_equal_out_tpu(self, torch::scalar_tensor(other), out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ne.Scalar_out", not_equal_scalar_out_tpu);
}

Tensor &equal_scalar_out_tpu(const Tensor &self, const Scalar &other, Tensor &out) {
  return equal_out_tpu(self, torch::scalar_tensor(other), out);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("eq.Scalar_out", equal_scalar_out_tpu);
}

} // namespace at

// FIXME Fix Compare.pl & uncomment this
#if (defined(BACKEND_SG2260) || defined(BACKEND_SG2260E)) && !defined(SOC_MODE)
#define USING_PPL
#endif
