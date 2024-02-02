#include <ATen/core/TensorBase.h>

#include "TPUTorchUtils.h"
#include <c10/util/Logging.h>
#include <iostream>

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {

Tensor _copy_from_tpu(const Tensor &self, const Tensor &dst,
                      bool non_blocking) {
  if (self.dtype() == dst.dtype()) {
    TORCH_CHECK(self.nbytes() == dst.nbytes(),
                "SELF and dst number bytes must be the same");
    if (IS_CPU_TENSOR(self) && IS_TPU_TENSOR(dst)) {
      if (dst.is_contiguous()) {
        TIMING_START;
        tpu::TPUCopyHostToDevice(dst.data_ptr(), self.contiguous().data_ptr(),
                                 dst.nbytes(), non_blocking);
        TIMING_END(tpu::CDMA_S2D);
      } else {
        dst.copy_(self.contiguous().to(dst.device()), non_blocking);
      }
    } else if (IS_TPU_TENSOR(self) && IS_CPU_TENSOR(dst)) {
      if (dst.is_contiguous()) {
        TIMING_START;
        tpu::TPUCopyDeviceToHost(dst.data_ptr(), self.contiguous().data_ptr(),
                                 dst.nbytes(), non_blocking);
        TIMING_END(tpu::CDMA_D2S);
      } else {
        dst.copy_(self.contiguous().to(dst.device()), non_blocking);
      }
    } else if (IS_TPU_TENSOR(self) && IS_TPU_TENSOR(dst)) {
      if (self.is_contiguous() && dst.is_contiguous()) {
        TIMING_START;
        tpu::TPUCopyDeviceToDevice(dst.data_ptr(), self.data_ptr(),
                                   dst.nbytes(), non_blocking);
        TIMING_END(tpu::COPY);
      } else {
        TIMING_START;
        #if defined BACKEND_1684X
        auto status = sgdnnStridedCopy(tpu::TPUGetDeviceHandle(),
                                              tpu::TPUGenerateSgdnnTensor(self),
                                              tpu::TPUGenerateSgdnnTensor(dst));
        TORCH_CHECK(status == BM_SUCCESS);
        #elif defined BACKEND_SG2260
        auto status = sgdnnStridedCopy(c10_tpu::getCurrentTPUStream(),
                                              tpu::TPUGenerateSgdnnTensor(self),
                                              tpu::TPUGenerateSgdnnTensor(dst),
                                              non_blocking);
        TORCH_CHECK(status == tpuRtSuccess);
        #endif
        TIMING_END(tpu::STRIDED_COPY);
      }
    } else {
      TORCH_CHECK(false, "Unsupported copy from device ", self.device(),
                  " to device ", dst.device());
    }
  } else {
    if (IS_CPU_TENSOR(self) && IS_TPU_TENSOR(dst)) {
      dst.copy_(self.to(dst.device()), non_blocking);
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
        TIMING_START;
        auto dst_cpu = self.cpu().to ( dst.dtype() );
        tpu::TPUCopyHostToDevice ( dst.data_ptr(), dst_cpu.contiguous().data_ptr(), dst.nbytes(), non_blocking );
        TIMING_END(tpu::CPU_LAYER);
      }
      else
      {
        auto self_ = self.contiguous();
        if (dst.is_contiguous()) {
          TIMING_START;
          #if defined BACKEND_1684X
          auto status = sgdnnConvert(tpu::TPUGetDeviceHandle(),
                                    tpu::TPUGenerateSgdnnTensor(self_),
                                    tpu::TPUGenerateSgdnnTensor(dst));
          TORCH_CHECK(status == BM_SUCCESS);
          #elif defined BACKEND_SG2260
          auto status = sgdnnConvert(c10_tpu::getCurrentTPUStream(),
                                    tpu::TPUGenerateSgdnnTensor(self_),
                                    tpu::TPUGenerateSgdnnTensor(dst),
                                    non_blocking);
          TORCH_CHECK(status == tpuRtSuccess);
          #endif
          TIMING_END(tpu::DTYPE_CONVERT);
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
  auto layout = layout_opt.has_value() ? layout_opt.value() : option.layout();
  auto device = device_opt.has_value() ? device_opt.value() : option.device();
  auto pin_memory = pin_memory_opt.has_value() ? pin_memory_opt.value()
                                               : option.pinned_memory();
  auto memory_format = memory_format_opt.has_value()
                           ? memory_format_opt
                           : option.memory_format_opt();

  auto dst =
      empty(self.sizes(), dtype, layout, device, pin_memory, memory_format);

  return _copy_from_tpu(self, dst, non_blocking);
}

Tensor _to_copy_out_tpu(const Tensor &self, Tensor &out,
                        c10::optional<ScalarType> dtype_opt,
                        c10::optional<Layout> layout_opt,
                        c10::optional<Device> device_opt,
                        c10::optional<bool> pin_memory_opt, bool non_blocking,
                        c10::optional<MemoryFormat> memory_format_opt) {

  auto option = self.options();
  auto dtype = dtype_opt.has_value() ? dtype_opt.value()
                                     : typeMetaToScalarType(option.dtype());
  auto layout = layout_opt.has_value() ? layout_opt.value() : option.layout();
  auto device = device_opt.has_value() ? device_opt.value() : option.device();
  auto pin_memory = pin_memory_opt.has_value() ? pin_memory_opt.value()
                                               : option.pinned_memory();
  auto memory_format = memory_format_opt.has_value()
                           ? memory_format_opt
                           : option.memory_format_opt();

  out.to(dtype, layout, device, pin_memory, false, false, memory_format);
  return _copy_from_tpu(self, out, non_blocking);
}

Tensor clone_tpu(const Tensor &self,
                 c10::optional<MemoryFormat> memory_format_opt) {
  auto memory_format_opt_ = memory_format_opt;
  if (memory_format_opt.has_value() && memory_format_opt.value() == MemoryFormat::Preserve) {
    memory_format_opt_ = self.options().memory_format_opt();
  }
  auto dst = empty(self.sizes(), self.options(), memory_format_opt_);
  return _copy_from_tpu(self, dst, false);
}

Tensor &clone_out_tpu(const Tensor &self,
                      c10::optional<MemoryFormat> memory_format_opt,
                      Tensor &out) {
  _copy_from_tpu(self, out, false);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("_copy_from", _copy_from_tpu);
  m.impl("_to_copy", _to_copy_tpu);
  m.impl("clone", clone_tpu);
  m.impl("clone.out", clone_out_tpu);
}
} // namespace at
