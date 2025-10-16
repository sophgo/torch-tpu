#include "common/config.h"
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>
#ifdef USING_PPL
#include "Fill.h"
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

template <typename scalar_t>
static void fill_async_impl(
	uint64_t output_addr,
  scalar_t value,
  uint32_t outer_size
  ){
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
        uint32_t tile_size) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return fill_fp32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        return fill_fp16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        return fill_bf16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
        return fill_int32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
        return fill_uint8(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr,
            value,
            outer_size,
            tile_size
            );
    }
    return -1;
  };

	auto stream = c10_tpu::getCurrentTPUStream();
	tpuKernelModule_t ppl_module = getPplModule();
  constexpr uint32_t MAX_TILE_SIZE = 1024;
  uint32_t tile_size = std::min(outer_size, MAX_TILE_SIZE);
  while (tile_size >= 1) {
      int ret = kernel(stream, ppl_module, tile_size);
      if (ret == 0) {
          return;
      } else {
          tile_size = tile_size / 2;
          continue;
      }
  }
	TORCH_CHECK(false, "fill failed !");
}
#endif
namespace at {
Tensor &fill__Scalar_tpu(Tensor &self, const Scalar &value) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  if ( self.numel() == 0) return self;
#if 0
  auto self_cpu = TENSOR_TO_CPU ( self );
  self_cpu.fill_ ( value );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), self_cpu.contiguous().data_ptr(), self.nbytes() );
#else
#ifdef USING_PPL
  if (usePPLKernels())
  {  int outer_size = 1;
    for (const auto i : c10::irange(self.dim())) {
        outer_size *= self.size(i);
    }
    AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "fill", [&] {
            fill_async_impl<scalar_t>(
                reinterpret_cast<uint64_t>(self.data_ptr()),
                value.to<scalar_t>(),
                outer_size
                );
        });
  } else
#endif
  { int64_t value_;
    if (self.scalar_type() == at::ScalarType::Float) {
      float value_f = value.toFloat();
      memcpy(&value_, &value_f, sizeof(float));
    } else if (self.scalar_type() == at::ScalarType::Half) {
      at::Half fp16 = value.toHalf();
      memcpy(&value_, &fp16, sizeof(at::Half));
    } else if (self.scalar_type() == at::ScalarType::BFloat16) {
      at::BFloat16 bf16 = value.toBFloat16();
      memcpy(&value_, &bf16, sizeof(at::BFloat16));
    } else if (self.scalar_type() == at::ScalarType::Int) {
      value_ = value.toInt();
    } else if (self.scalar_type() == at::ScalarType::Bool) {
      value_ = value.toBool();
    } else if (self.scalar_type() == at::ScalarType::Byte) {
      value_ = value.toByte();
    } else {
      TORCH_CHECK(false, "unsupport scalar type");
    }

    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnFillAsync(
        stream,
        &value_,
        tpu::TPUGenerateTpudnnTensor(stream, self));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);}
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self);
  return self;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("fill_.Scalar", fill__Scalar_tpu); }

Tensor &zero__tpu(Tensor &self) {
  CHECK_TENSOR_IN_DEVICE(self);
#if 0
  char * buffer = new char [self.nbytes()];
  memset ( buffer, 0x0, self.nbytes() );
  tpu::TPUCopyHostToDevice ( self.data_ptr(), buffer, self.nbytes() );
  delete [] buffer;
#else
  fill__Scalar_tpu(self, 0);
#endif
  return self;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("zero_", zero__tpu); }

Tensor &masked_fill_Scalar_tpu(Tensor &self, const Tensor &mask,
                               const Scalar &value) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(mask);
  auto mask_dim = mask.dim();
  if(mask_dim == 0)
    return self;
#if 0
  LOG(WARNING) << "masked_fill_.Scalar use cpu impl";
  auto self_cpu = self.cpu();
  self_cpu.masked_fill_(mask.cpu(), value);
  tpu::TPUCopyHostToDevice(self.data_ptr(), self_cpu.contiguous().data_ptr(),
                           self.nbytes());
#else
  Tensor maski = mask.to(self.dtype());
  Tensor &mask_int = maski;

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnMaskedFillAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, mask_int),
      value.toFloat(),
      tpu::TPUGenerateTpudnnTensor(stream, self));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, mask);
  return self;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("masked_fill_.Scalar", masked_fill_Scalar_tpu);
}

} // namespace at
