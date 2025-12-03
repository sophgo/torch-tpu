#include <iostream>
#include <ATen/core/TensorBase.h>
#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"
#include "TPUTorchUtils.h"
#include "torch_tpu/csrc/aten/TPUNativeFunctions.h"

#ifdef USING_PPL
#define FW_MAX_SHAPE_DIMS      8
#include "CopyFrom.h"
static void copy_impl(
    uint64_t output_addr,
    uint64_t input_addr,
    int is_bool,
    uint32_t outer_size,
    uint32_t inner_size,
    at::ScalarType dst_type,
    at::ScalarType src_type
)
{
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
    int tile_size, uint32_t inner_size, uint32_t outer_size) -> int {

    if (src_type == at::kHalf) {
      if (dst_type == at::kFloat){
        return convert_impl_half_to_float(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      } else if (dst_type == at::kHalf){
        return convert_impl_half_to_half(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      }
    } else if (src_type == at::kBFloat16) {
      if (dst_type == at::kFloat){
        return convert_impl_bf16_to_float(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      } else if (dst_type == at::kBFloat16){
        return convert_impl_bf16_to_bf16(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      }
    } else if (src_type == at::kFloat) {
      if (dst_type == at::kFloat) {
        return convert_impl_float_to_float(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      } else if (dst_type == at::kHalf) {
        return convert_impl_float_to_half(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      } else if (dst_type == at::kBFloat16) {
        return convert_impl_float_to_bf16(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      }
    } else if (src_type == at::kInt) {
      if (dst_type == at::kInt){
        return convert_impl_int32_to_int32(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      } else if(dst_type == at::kFloat){
        return convert_impl_int32_to_float(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      } else if(dst_type == at::kHalf){
        return convert_impl_int32_to_half(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      } else if(dst_type == at::kBFloat16){
        return convert_impl_int32_to_bf16(stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                output_addr, input_addr, tile_size, inner_size, outer_size);
      }
    }
    return -1;
  };

  auto stream = c10_tpu::getCurrentTPUStream();
  tpuKernelModule_t ppl_module = getPplModule();
  uint32_t tile_size = inner_size;
  while (tile_size >= 1) {
    int ret = kernel(stream, ppl_module, tile_size, inner_size, outer_size);
    if (ret == 0) {
      return;
    } else {
      tile_size = tile_size / 2;
      continue;
    }
  }

}
#endif


#ifdef USING_PPL
#include "Gdmad2d.h"
#define AT_DISPATCH_FLOAT_AND_INT_TYPES(scalar_type, name, func)  \
AT_DISPATCH_SWITCH(                   \
scalar_type, name,                    \
AT_DISPATCH_CASE(at::kFloat, func)    \
AT_DISPATCH_CASE(at::kHalf, func)     \
AT_DISPATCH_CASE(at::kBFloat16, func) \
AT_DISPATCH_CASE(at::kFloat8_e4m3fn, func) \
AT_DISPATCH_CASE(at::kFloat8_e5m2, func) \
AT_DISPATCH_CASE(at::kInt, func)      \
AT_DISPATCH_CASE(at::kShort, func)    \
AT_DISPATCH_CASE(at::kChar, func)     \
AT_DISPATCH_CASE(at::kByte, func))

template <typename scalar_t>
static void gdmad2d_async_impl(
    uint64_t input_addr,
    uint64_t output_addr,
    uint32_t outer_size
  )
{
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
          uint32_t tile_size) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return gdmad2d_fp32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, input_addr, outer_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        return gdmad2d_fp16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, input_addr, outer_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        return gdmad2d_bf16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, input_addr, outer_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
        return gdmad2d_int32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, input_addr, outer_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
        return gdmad2d_int16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, input_addr, outer_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
        return gdmad2d_int8(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, input_addr, outer_size, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, uint8_t> ||
      std::is_same_v<scalar_t, Float8_e4m3fn> ||
      std::is_same_v<scalar_t, Float8_e5m2>) {
        return gdmad2d_uint8(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, input_addr, outer_size, tile_size);
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
  TORCH_CHECK(false, "gdmad2d failed !");
}

template <typename scalar_t>
static void stridedcopy_async_impl(
    uint64_t input_addr,
    uint64_t output_addr,
    int* shape,
    int* src_stride,
    int* dst_stride,
    int dim
  )
{
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,
          uint32_t tile_size) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return stridedcopy_fp32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, shape[0], shape[1], shape[2], shape[3], shape[4],
        src_stride[0], src_stride[1], src_stride[2], src_stride[3], src_stride[4],
        dst_stride[0], dst_stride[1], dst_stride[2], dst_stride[3], dst_stride[4],
        dim, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
      return stridedcopy_fp16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, shape[0], shape[1], shape[2], shape[3], shape[4],
        src_stride[0], src_stride[1], src_stride[2], src_stride[3], src_stride[4],
        dst_stride[0], dst_stride[1], dst_stride[2], dst_stride[3], dst_stride[4],
        dim, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
      return stridedcopy_bf16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, shape[0], shape[1], shape[2], shape[3], shape[4],
        src_stride[0], src_stride[1], src_stride[2], src_stride[3], src_stride[4],
        dst_stride[0], dst_stride[1], dst_stride[2], dst_stride[3], dst_stride[4],
        dim, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
      return stridedcopy_int32(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, shape[0], shape[1], shape[2], shape[3], shape[4],
        src_stride[0], src_stride[1], src_stride[2], src_stride[3], src_stride[4],
        dst_stride[0], dst_stride[1], dst_stride[2], dst_stride[3], dst_stride[4],
        dim, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
      return stridedcopy_int16(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, shape[0], shape[1], shape[2], shape[3], shape[4],
        src_stride[0], src_stride[1], src_stride[2], src_stride[3], src_stride[4],
        dst_stride[0], dst_stride[1], dst_stride[2], dst_stride[3], dst_stride[4],
        dim, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
      return stridedcopy_int8(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, shape[0], shape[1], shape[2], shape[3], shape[4],
        src_stride[0], src_stride[1], src_stride[2], src_stride[3], src_stride[4],
        dst_stride[0], dst_stride[1], dst_stride[2], dst_stride[3], dst_stride[4],
        dim, tile_size);
    } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
      return stridedcopy_uint8(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, shape[0], shape[1], shape[2], shape[3], shape[4],
        src_stride[0], src_stride[1], src_stride[2], src_stride[3], src_stride[4],
        dst_stride[0], dst_stride[1], dst_stride[2], dst_stride[3], dst_stride[4],
        dim, tile_size);
    }
    return -1;
  };

  auto stream = c10_tpu::getCurrentTPUStream();
  tpuKernelModule_t ppl_module = getPplModule();
  uint32_t tile_size = dim >=4 ? shape[1] : shape[0];

  while (tile_size >= 1) {
    int ret = kernel(stream, ppl_module, tile_size);
    if (ret == 0) return;
    else tile_size /= 2;
  }

  TORCH_CHECK(false, "stridedcopy failed !");
}
#endif

namespace at {
#ifdef USING_PPL
static inline void simplify(int* shape, int* src_stride, int* dst_stride, int* dim) {
  for (int i = 0; i < (*dim) - 1; ) {
    bool src_contiguous = (src_stride[i+1] * shape[i+1] == src_stride[i]);
    bool dst_contiguous = (dst_stride[i+1] * shape[i+1] == dst_stride[i]);

    if (src_contiguous && dst_contiguous && shape[i] * shape[i+1] < (1 << 16)) {
      shape[i] *= shape[i+1];
      src_stride[i] = src_stride[i+1];
      dst_stride[i] = dst_stride[i+1];

      for (int j = i + 1; j < (*dim) - 1; ++j) {
        shape[j] = shape[j+1];
        src_stride[j] = src_stride[j+1];
        dst_stride[j] = dst_stride[j+1];
      }
      --(*dim);
    } else {
      ++i;
    }
  }

  for (int i = *dim; i < FW_MAX_SHAPE_DIMS; ++i) {
    shape[i] = 1;
    src_stride[i] = 1;
    dst_stride[i] = 1;
  }

  if (*dim == 5 && shape[1] == 1 && shape[2] == 1) {
    return;
  }

  if (*dim > 4) {
    int merged_b = 1;
    int preserved_stride_src = src_stride[0];
    int preserved_stride_dst = dst_stride[0];

    for (int i = 0; i < *dim - 4; ++i) {
      merged_b *= shape[i];
      preserved_stride_src = std::min(preserved_stride_src, src_stride[i]);
      preserved_stride_dst = std::min(preserved_stride_dst, dst_stride[i]);
    }

    int new_shape[5] = {merged_b, shape[*dim-4], shape[*dim-3], shape[*dim-2], shape[*dim-1]};
    int new_src_stride[5] = {
      preserved_stride_src,
      src_stride[*dim-4],
      src_stride[*dim-3],
      src_stride[*dim-2],
      src_stride[*dim-1]
    };
    int new_dst_stride[5] = {
      preserved_stride_dst,
      dst_stride[*dim-4],
      dst_stride[*dim-3],
      dst_stride[*dim-2],
      dst_stride[*dim-1]
    };

    for (int i = 0; i < 5; ++i) {
      shape[i] = new_shape[i];
      src_stride[i] = new_src_stride[i];
      dst_stride[i] = new_dst_stride[i];
    }
    *dim = 5;
  }
}
#endif

Tensor _copy_from_tpu(const Tensor &self, const Tensor &dst,
                      bool non_blocking) {
  if (self.numel() == 0) {return dst;}
  if (self.dtype() == dst.dtype()) {
    TORCH_CHECK(self.nbytes() == dst.nbytes(),
                "SELF and dst number bytes must be the same");
    if (IS_CPU_TENSOR(self) && IS_TPU_TENSOR(dst)) {
      if (!self.is_contiguous()) {
        //avoid self.contiguous() be destructed when non_blocking is true
        non_blocking = false;
      }
      if (dst.is_contiguous()) {
        tpu::TPUCopyHostToDevice(dst.data_ptr(), self.contiguous().data_ptr(),
                                 dst.nbytes(), non_blocking);
      } else {
        dst.copy_(self.contiguous().to(dst.device()), non_blocking);
      }
    } else if (IS_TPU_TENSOR(self) && IS_CPU_TENSOR(dst)) {
      if (dst.is_contiguous()) {
        tpu::TPUCopyDeviceToHost(dst.data_ptr(), self.contiguous().data_ptr(),
                                 dst.nbytes(), false);
      } else {
        dst.copy_(self.contiguous().to(dst.device()), non_blocking);
      }
    } else if (IS_TPU_TENSOR(self) && IS_TPU_TENSOR(dst)) {
      if (self.is_contiguous() && dst.is_contiguous()) {
#ifdef USING_PPL
        if (usePPLKernels())
        {uint32_t outer_size = 1;
          for (const auto i : c10::irange(self.dim())) {
              outer_size *= self.size(i);
          }
          AT_DISPATCH_FLOAT_AND_INT_TYPES( self.scalar_type(), "gdmad2d", [&] {
            gdmad2d_async_impl<scalar_t>(
              reinterpret_cast<uint64_t>(self.data_ptr()),
              reinterpret_cast<uint64_t>(dst.data_ptr()),
              outer_size);
          });
        } else
#endif
        {
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnGDMAD2DAsync(
          stream,
          tpu::TPUGenerateTpudnnTensor(stream, self),
          tpu::TPUGenerateTpudnnTensor(stream, dst),
          dst.nbytes());
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
        }
      } else {
#ifdef USING_PPL
        if (usePPLKernels())
        {  int dim = self.dim();
          int shape[FW_MAX_SHAPE_DIMS] = {1};
          int src_stride[FW_MAX_SHAPE_DIMS] = {1};
          int dst_stride[FW_MAX_SHAPE_DIMS] = {1};

          for (int i = 0; i < dim; ++i) {
            shape[i] = self.size(i);
            src_stride[i] = self.stride(i);
            dst_stride[i] = dst.stride(i);
          }
          simplify(shape, src_stride, dst_stride, &dim);
          AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "stridedcopy", [&] {
            scalar_t* input_ptr = self.data_ptr<scalar_t>();
            scalar_t* output_ptr = dst.data_ptr<scalar_t>();
            if (dim > 4 ){
              const int batch_size = shape[0];
              shape[0] = 1;
              for (int b = 0; b < batch_size; ++b) {
                uint64_t cur_input_addr = reinterpret_cast<uint64_t>(input_ptr + b * src_stride[0]);
                uint64_t cur_output_addr = reinterpret_cast<uint64_t>(output_ptr + b * dst_stride[0]);
                stridedcopy_async_impl<scalar_t>(
                  cur_input_addr,
                  cur_output_addr,
                  shape, src_stride, dst_stride, dim);
              }
            } else {
              stridedcopy_async_impl<scalar_t>(
                reinterpret_cast<uint64_t>(input_ptr),
                reinterpret_cast<uint64_t>(output_ptr),
                shape, src_stride, dst_stride, dim);
            }

          });
        } else
#endif
      {
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnStridedCopyAsync(
          stream,
          tpu::TPUGenerateTpudnnTensor(stream, self),
          tpu::TPUGenerateTpudnnTensor(stream, dst));
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
      }
    }
    } else {
      TORCH_CHECK(false, "Unsupported copy from device ", self.device(),
                  " to device ", dst.device());
    }
  } else {
    if (IS_CPU_TENSOR(self) && IS_TPU_TENSOR(dst)) {
      if ((!self.is_contiguous()) || (self.dtype() != dst.dtype())) {
        non_blocking = false;
      }
      tpu::TPUCopyHostToDevice ( dst.data_ptr(), self.contiguous().to(dst.dtype()).data_ptr(), dst.nbytes(), non_blocking );
    } else if (IS_TPU_TENSOR(self) && IS_CPU_TENSOR(dst)) {
      dst.copy_(self.to(dst.dtype()), non_blocking);
    } else if (IS_TPU_TENSOR(self) && IS_TPU_TENSOR(dst)) {
#if 0
      auto dst_cpu = self.cpu().to ( dst.dtype() );
      tpu::TPUCopyHostToDevice ( dst.data_ptr(), dst_cpu.contiguous().data_ptr(), dst.nbytes(), non_blocking );
#else
      if ( !tpu::IsSupportDtype(self.dtype()) || !tpu::IsSupportDtype( dst.dtype() ))
      {
        CPU_IMPL_WARNING("unsupport dtype.");
        non_blocking = false; // avoid dst_cpu be destructed when non_blocking is true
        auto dst_cpu = self.cpu().to ( dst.dtype() );
        tpu::TPUCopyHostToDevice ( dst.data_ptr(), dst_cpu.contiguous().data_ptr(), dst.nbytes(), non_blocking );
      }
      else
      {
        auto self_ = self.contiguous();
        if (dst.is_contiguous()) {
#ifdef USING_PPL
    if (usePPLKernels())
    {
          int is_bool = (dst.dtype() == caffe2::TypeMeta::Make<bool>());
          uint32_t outer_size = 1;
          uint32_t inner_size = 1;
          int axis = 0;
          if (self_.dim() > 2){
            axis = 2;
          } else if (self_.dim() > 0){
            axis = 1;
          }
          for (const auto i : c10::irange(axis)) {
              outer_size *= self_.size(i);
          }
          for (const auto i : c10::irange(axis, self_.dim())) {
              inner_size *= self_.size(i);
          }
          at::ScalarType self_type = self_.scalar_type();
          at::ScalarType dst_type = dst.scalar_type();

          copy_impl(
            reinterpret_cast<uint64_t>(dst.data_ptr()),
            reinterpret_cast<uint64_t>(self_.data_ptr()),
            is_bool,
            outer_size,
            inner_size,
            dst_type,
            self_type);
    } else
#endif
    {

          auto stream = c10_tpu::getCurrentTPUStream();
          int is_bool = (dst.dtype() == caffe2::TypeMeta::Make<bool>());
          auto status = tpudnnConvertAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, self_),
            tpu::TPUGenerateTpudnnTensor(stream, dst),
            is_bool);
          TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);

    }
          SHOW_TENSOR_OP(self_, dst);
        } else {
          dst.copy_(self_.to(dst.dtype()), non_blocking);
        }
      }

#endif
    } else {
      TORCH_CHECK(false, "Unsupported copy from device ", self.device(),
                  " to device ", dst.device());
    }
  }
  return dst;
}
// _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None,
// Device? device=None, bool? pin_memory=None, bool non_blocking=False,
// MemoryFormat? memory_format=None) -> Tensor
Tensor _to_copy_tpu(const Tensor &self, c10::optional<ScalarType> dtype_opt,
                    c10::optional<Layout> layout_opt,
                    c10::optional<Device> device_opt,
                    c10::optional<bool> pin_memory_opt, bool non_blocking,
                    c10::optional<MemoryFormat> memory_format_opt) {
  auto option = self.options();
  auto dtype = dtype_opt.has_value() ? dtype_opt.value()
                                     : typeMetaToScalarType(option.dtype());
  auto self_ = self;
  if ( IS_CPU_TENSOR(self) && dtype == ScalarType::Long )
  {
    dtype = ScalarType::Int;
    self_ = self.to(dtype);
    // self_ is a temp var, maybe destructed after to(dtype) when non_blocking is true
    // non_blocking need to be false
    non_blocking = false;

  }

  auto layout = layout_opt.has_value() ? layout_opt.value() : option.layout();
  auto device = device_opt.has_value() ? device_opt.value() : option.device();
  auto pin_memory = pin_memory_opt.has_value() ? pin_memory_opt.value()
                                               : option.pinned_memory();
  auto memory_format = memory_format_opt.has_value()
                           ? memory_format_opt
                           : option.memory_format_opt();

  auto dst =
      empty(self.sizes(), dtype, layout, device, pin_memory, memory_format);

  // copy formated tpu Tensor to cpu, recover tpu tensor's format firstly
  if ( device.type() == c10::DeviceType::CPU  &&
        !at_tpu::StorageDescHelper::IsBaseFormatType(self_) )
  {
    auto self_ori = at_tpu::TPUNativeFunctions::tpu_format_cast_back_to_origin(self_);
    return _copy_from_tpu(self_ori, dst, non_blocking);
  }
  return _copy_from_tpu(self_, dst, non_blocking);
}

// contiguous
Tensor &clone_out_tpu(const Tensor &self,
                      c10::optional<MemoryFormat> memory_format_opt,
                      Tensor &dst) {
  TIMING_START;
  if (self.numel() == 0) {return dst;}
  TORCH_CHECK( self.scalar_type() == dst.scalar_type(), "[clone_out] IO must have same dtype" );
  TORCH_CHECK( self.device() == dst.device(), "[clone_out] IO must on same device" );

  if (self.is_contiguous() && dst.is_contiguous()) {
#ifdef USING_PPL
    if (usePPLKernels())
    {uint32_t outer_size = 1;
      for (const auto i : c10::irange(self.dim())) {
          outer_size *= self.size(i);
      }
      AT_DISPATCH_FLOAT_AND_INT_TYPES( self.scalar_type(), "gdmad2d", [&] {
        gdmad2d_async_impl<scalar_t>(
          reinterpret_cast<uint64_t>(self.data_ptr()),
          reinterpret_cast<uint64_t>(dst.data_ptr()),
          outer_size);
      });
    } else
#endif
    { auto stream = c10_tpu::getCurrentTPUStream();
      auto status = tpudnnGDMAD2DAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        tpu::TPUGenerateTpudnnTensor(stream, dst),
        dst.nbytes());
      TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }
  } else {
#ifdef USING_PPL
        if (usePPLKernels())
        {  int dim = self.dim();
          int shape[FW_MAX_SHAPE_DIMS] = {1};
          int src_stride[FW_MAX_SHAPE_DIMS] = {1};
          int dst_stride[FW_MAX_SHAPE_DIMS] = {1};

          for (int i = 0; i < dim; ++i) {
            shape[i] = self.size(i);
            src_stride[i] = self.stride(i);
            dst_stride[i] = dst.stride(i);
          }
          simplify(shape, src_stride, dst_stride, &dim);
          AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "stridedcopy", [&] {
            scalar_t* input_ptr = self.data_ptr<scalar_t>();
            scalar_t* output_ptr = dst.data_ptr<scalar_t>();
            if (dim > 4 ){
              const int batch_size = shape[0];
              shape[0] = 1;
              for (int b = 0; b < batch_size; ++b) {
                uint64_t cur_input_addr = reinterpret_cast<uint64_t>(input_ptr + b * src_stride[0]);
                uint64_t cur_output_addr = reinterpret_cast<uint64_t>(output_ptr + b * dst_stride[0]);
                stridedcopy_async_impl<scalar_t>(
                  cur_input_addr,
                  cur_output_addr,
                  shape, src_stride, dst_stride, dim);
              }
            } else {
              stridedcopy_async_impl<scalar_t>(
                reinterpret_cast<uint64_t>(input_ptr),
                reinterpret_cast<uint64_t>(output_ptr),
                shape, src_stride, dst_stride, dim);
            }

          });
        } else
#endif
    {
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnStridedCopyAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, dst));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    }
  }
  TIMING_END;
  return dst;
}

Tensor clone_tpu(const Tensor &self,
                 c10::optional<MemoryFormat> memory_format_opt) {
  auto memory_format_opt_ = memory_format_opt;
  if (memory_format_opt.has_value() && memory_format_opt.value() == MemoryFormat::Preserve) {
    memory_format_opt_ = self.options().memory_format_opt();
  }
  auto dst = empty(self.sizes(), self.options(), memory_format_opt_);
  return clone_out_tpu(self, memory_format_opt, dst);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("_copy_from", _copy_from_tpu);
  m.impl("_to_copy", _to_copy_tpu);
  m.impl("clone", clone_tpu);
  m.impl("clone.out", clone_out_tpu);
}

at::Tensor contiguous_tpu(const at::Tensor & self, at::MemoryFormat memory_format)
{
  if (self.is_contiguous()) return self;
  TIMING_START;
  auto out = empty(self.sizes(), self.options(), memory_format);
#ifdef USING_PPL
  if (usePPLKernels())
  {  int dim = self.dim();
    int shape[FW_MAX_SHAPE_DIMS] = {1};
    int src_stride[FW_MAX_SHAPE_DIMS] = {1};
    int dst_stride[FW_MAX_SHAPE_DIMS] = {1};

    for (int i = 0; i < dim; ++i) {
      shape[i] = self.size(i);
      src_stride[i] = self.stride(i);
      dst_stride[i] = out.stride(i);
    }
    simplify(shape, src_stride, dst_stride, &dim);
    AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "stridedcopy", [&] {
      scalar_t* input_ptr = self.data_ptr<scalar_t>();
      scalar_t* output_ptr = out.data_ptr<scalar_t>();
      if (dim > 4 ){
        const int batch_size = shape[0];
        shape[0] = 1;
        for (int b = 0; b < batch_size; ++b) {
          uint64_t cur_input_addr = reinterpret_cast<uint64_t>(input_ptr + b * src_stride[0]);
          uint64_t cur_output_addr = reinterpret_cast<uint64_t>(output_ptr + b * dst_stride[0]);
          stridedcopy_async_impl<scalar_t>(
            cur_input_addr,
            cur_output_addr,
            shape, src_stride, dst_stride, dim);
        }
      } else {
        stridedcopy_async_impl<scalar_t>(
          reinterpret_cast<uint64_t>(input_ptr),
          reinterpret_cast<uint64_t>(output_ptr),
          shape, src_stride, dst_stride, dim);
      }

    });
  } else
#endif
  {  auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnStridedCopyAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
  TIMING_END;
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("contiguous", contiguous_tpu);
}
} // namespace at


//////////////// ****************** autogradtpu key ****************** ////////////////
namespace torch {
namespace autograd {
class ContiguousFunction : public torch::autograd::Function<ContiguousFunction>
{
public:
  static at::Tensor forward (
    AutogradContext *ctx, const at::Tensor & self, at::MemoryFormat memory_format )
  {
    auto out =at::contiguous_tpu(self, memory_format);
    return out;
  }

  static tensor_list backward ( AutogradContext *ctx, tensor_list grad_outputs )
  {
    auto grad_input = clone_tpu(grad_outputs[0], c10::nullopt);
    return {grad_input,  at::Tensor()};
  }
};
} //namespace autograd
} //namespace torch

namespace at {
Tensor contiguous_autogradtpu ( const at::Tensor& self, at::MemoryFormat memory_format)
{
  return torch::autograd::ContiguousFunction::apply ( self, memory_format );
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("contiguous", contiguous_autogradtpu);
}
} // namespace at
