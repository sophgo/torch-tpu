#include <iostream>
#include <ATen/core/TensorBase.h>
#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"
#include "TPUTorchUtils.h"
#include "torch_tpu/csrc/aten/TPUNativeFunctions.h"

#ifdef USING_PPL
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

namespace at {

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
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnGDMAD2DAsync(
          stream,
          tpu::TPUGenerateTpudnnTensor(stream, self),
          tpu::TPUGenerateTpudnnTensor(stream, dst),
          dst.nbytes());
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
      } else {
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnStridedCopyAsync(
          stream,
          tpu::TPUGenerateTpudnnTensor(stream, self),
          tpu::TPUGenerateTpudnnTensor(stream, dst));
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
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
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnGDMAD2DAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, dst),
      dst.nbytes());
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  } else {
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnStridedCopyAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, dst));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
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
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnStridedCopyAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, self),
    tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
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
