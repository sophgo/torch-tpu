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

namespace at {

Tensor &binary_op_tpu(const Tensor &self, const Tensor &other,
                      const Scalar &alpha, Tensor &out, int binary_type) {
  TIMING_START;

  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  auto self_ = self;
  if(self_.dtype() == torch::kInt64){
    self_ = self.to(torch::kInt32);
  }
  auto other_ = other;
  if(other_.dtype() == torch::kInt64){
    other_ = other.to(torch::kInt32);
  }
  auto out_ = out;
  if(out_.dtype() == torch::kInt64){
    out_ = out.to(torch::kInt32);
  }

  if (self.dim() == 0 || other.dim() == 0) {
    // tensor op scalar
    auto stream = c10_tpu::getCurrentTPUStream();
    if (self.dim() == 0) {
      auto status = tpudnnBinaryCAsync(
          stream, tpu::TPUGenerateTpudnnTensor(stream, other_),
          self.item().toFloat() * alpha.toFloat(),
          tpu::TPUGenerateTpudnnTensor(stream, out_), binary_type, 1);
      TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
      TIMING_END(tpu::BINARYOP_C);
    } else {
      auto status = tpudnnBinaryCAsync(
          stream, tpu::TPUGenerateTpudnnTensor(stream, self_),
          other.item().toFloat() * alpha.toFloat(),
          tpu::TPUGenerateTpudnnTensor(stream, out_), binary_type, 0);
      TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
      TIMING_END(tpu::BINARYOP_C);
    }
    if(out.dtype() == torch::kInt64){
      out = out_.to(torch::kInt64);
    }else{
      out = out_;
    }
    return out;
  }

  if (tpu::TPUIsSameShape(self, other)) {
    auto other_ =
        other.dtype() == self.dtype() ? other : other.to(self.dtype());

    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnBinaryAsync(
        stream,
        tpu::TPUGenerateTpudnnTensor(stream, self),
        tpu::TPUGenerateTpudnnTensor(stream, other_),
        alpha.toDouble(),
        tpu::TPUGenerateTpudnnTensor(stream, out), binary_type);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(tpu::BINARYOP)
  } else {
    int self_dim = self.dim(), other_dim = other.dim();
    int max_dim = std::max(self_dim, other_dim);
    int self_shape[max_dim], other_shape[max_dim];
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

    auto stream = c10_tpu::getCurrentTPUStream();

    for (int i = 0; i < max_dim; i++) {
      TORCH_CHECK(self_shape[i] == other_shape[i] || self_shape[i] == 1 ||
                      other_shape[i] == 1,
                  "The size of tensor a (%d) must match the size of tensor b "
                  "(%d) at non-signleton dimension %d",
                  self_shape[i], other_shape[i], i)
    }
    tpudnnTensor_t other_t;
    if (self.dtype() != other.dtype()) {
      // 创建一个新的 Tensor 对象，使用 self 的数据类型和 other 的大小和设备选项
      at::Tensor other_new = at::empty_like(other, self.options());
      other_new.copy_(other);
      // 使用新的 Tensor 对象进行操作，而不修改原始的 const Tensor &
      other_t = tpu::TPUGenerateTpudnnTensor(stream, other_new);
    } else {
      other_t = tpu::TPUGenerateTpudnnTensor(stream, other);
    }
    auto self_t = tpu::TPUGenerateTpudnnTensor(stream, self);
    auto out_t = tpu::TPUGenerateTpudnnTensor(stream, out);

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

    auto status = tpudnnBinaryBcastAsync(stream, self_t, other_t,
                                   alpha.toDouble(), out_t, binary_type);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(tpu::BINARYOP_BCAST)
  }
  return out;
}

Tensor &add_out_tpu(const Tensor &self, const Tensor &other,
                    const Scalar &alpha, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    if (out.is_contiguous()) {
      out = add(self.contiguous(), other.contiguous(), alpha);
    } else {
      auto out_ = add(self.contiguous(), other.contiguous(), alpha);
      TIMING_START;
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
      TIMING_END(tpu::STRIDED_COPY);
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = add(self.cpu(), other.cpu(), alpha);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self, other, alpha, out, 0);
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("add.out", add_out_tpu); }

Tensor &sub_out_tpu(const Tensor &self, const Tensor &other,
                    const Scalar &alpha, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    if (out.is_contiguous()) {
      out = sub(self.contiguous(), other.contiguous(), alpha);
    } else {
      auto out_ = sub(self.contiguous(), other.contiguous(), alpha);
      TIMING_START;
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out)); 
      TIMING_END(tpu::STRIDED_COPY);
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = sub(self.cpu(), other.cpu(), alpha);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self, other, alpha, out, 1);
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("sub.out", sub_out_tpu); }

Tensor &mul_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    if (out.is_contiguous()) {
      out = mul(self.contiguous(), other.contiguous());
    } else {
      auto out_ = mul(self.contiguous(), other.contiguous());
      TIMING_START;
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
      TIMING_END(tpu::STRIDED_COPY);
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = mul(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }
  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self, other, 1, out, 2);
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("mul.out", mul_out_tpu); }

Tensor &div_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    if (out.is_contiguous()) {
      out = div(self.contiguous(), other.contiguous());
    } else {
      auto out_ = div(self.contiguous(), other.contiguous());
      TIMING_START;
      auto handle = c10_tpu::getCurrentTPUStream();
      tpudnnStridedCopyAsync(
          handle,
          tpu::TPUGenerateTpudnnTensor(handle, out_),
          tpu::TPUGenerateTpudnnTensor(handle, out));
      TIMING_END(tpu::STRIDED_COPY);
    }
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = div(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
    SHOW_TENSOR_OP(self, other, out);
    return out;
  }

  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self, other, 1, out, 3);
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("div.out", div_out_tpu); }

template <typename Mode, typename UnaryFunc, typename BinaryFunc, typename CpuFunc>
void binary_impl(const Tensor &self, const Tensor &other, Tensor &out,
                 tpu::OpType type, Mode mode, UnaryFunc unary_func,
                 BinaryFunc binary_func, CpuFunc cpu_func) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  const auto handle = c10_tpu::getCurrentTPUStream();
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = cpu_func(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(type);
  } else if (self.dim() == 0 || other.dim() == 0) {
    TIMING_START;
    auto self_ = self, other_ = other;
    if (self.dtype() == torch::kInt64) {
      self_ = self.to(torch::kInt32);
    } else if (self.dtype() == torch::kFloat64) {
      self_ = self.to(torch::kFloat32);
    }
    if (other.dtype() == torch::kInt64) {
      other_ = other.to(torch::kInt32); 
    } else if (other.dtype() == torch::kFloat64) {
      other_ = other.to(torch::kFloat32); 
    }
    auto status = unary_func(handle, self_, other_, out);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(type);
  } else {
    TIMING_START;
    auto self_ = self.dtype() == out.dtype() ? self : self.to(out.dtype());
    auto other_ = other.dtype() == out.dtype()? other : other.to(out.dtype());
    auto status = binary_func(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self_),
        tpu::TPUGenerateTpudnnTensor(handle, other_),
        tpu::TPUGenerateTpudnnTensor(handle, out),
        mode);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(type);
  }
}
template <typename UnaryFunc, typename BinaryFunc, typename CpuFunc>
void binary_impl(const Tensor &self, const Tensor &other, Tensor &out,
                 tpu::OpType type, UnaryFunc unary_func, BinaryFunc binary_func,
                 CpuFunc cpu_func) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  const auto handle = c10_tpu::getCurrentTPUStream();
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = cpu_func(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(type);
  } else if (self.dim() == 0 || other.dim() == 0) {
    TIMING_START;
    auto self_ = self, other_ = other;
    if (self.dtype() == torch::kInt64) {
      self_ = self.to(torch::kInt32);
    } else if (self.dtype() == torch::kFloat64) {
      self_ = self.to(torch::kFloat32);
    }
    if (other.dtype() == torch::kInt64) {
      other_ = other.to(torch::kInt32); 
    } else if (other.dtype() == torch::kFloat64) {
      other_ = other.to(torch::kFloat32); 
    }
    auto status = unary_func(handle, self_, other_, out);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(type);
  } else {
    TIMING_START;
    auto self_ = self.dtype() == out.dtype() ? self : self.to(out.dtype());
    auto other_ = other.dtype() == out.dtype()? other : other.to(out.dtype());
    auto status = binary_func(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self_),
        tpu::TPUGenerateTpudnnTensor(handle, other_),
        tpu::TPUGenerateTpudnnTensor(handle, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(type);
  }
}
Tensor &bitwise_xor_out_tpu(const Tensor &self, const Tensor &other,
                            Tensor &out) {
  binary_impl(
      self, other, out, tpu::BITWISE_XOR,
      BitwiseMode_t::XOR,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other,
         Tensor &out) {
        return tpudnnBitwiseConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toInt() : other.item().toInt(),
            BitwiseMode_t::XOR);
      },
      tpudnnBitwiseAsync,
      [](const Tensor &a, const Tensor &b) { return bitwise_xor(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_xor.Tensor_out", bitwise_xor_out_tpu);
}

Tensor &bitwise_and_out_tpu(const Tensor &self, const Tensor &other,
                            Tensor &out) {
  binary_impl(
      self, other, out, tpu::BITWISE_AND,
      BitwiseMode_t::AND,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnBitwiseConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toInt() : other.item().toInt(),
            BitwiseMode_t::AND);
      },
      tpudnnBitwiseAsync,
      [](const Tensor &a, const Tensor &b) { return bitwise_and(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_and.Tensor_out", bitwise_and_out_tpu);
}

Tensor &bitwise_or_out_tpu(const Tensor &self, const Tensor &other,
                           Tensor &out) {
  binary_impl(
      self, other, out, tpu::BITWISE_OR, BitwiseMode_t::OR,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other,
         Tensor &out) {
        return tpudnnBitwiseConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toInt() : other.item().toInt(),
            BitwiseMode_t::OR);
      },
      tpudnnBitwiseAsync,
      [](const Tensor &a, const Tensor &b) { return bitwise_or(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_or.Tensor_out", bitwise_or_out_tpu);
}

Tensor &equal_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  binary_impl(
      self, other, out, tpu::EQUAL, CompareMode_t::TPUDNN_EQ,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_EQ, self.dim() == 0 ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return eq(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("eq.Tensor_out", equal_out_tpu); }

Tensor &greater_or_equal_out_tpu(const Tensor &self, const Tensor &other,
                                 Tensor &out) {
  binary_impl(
      self, other, out, tpu::GREATER_OR_EQUAL,
      CompareMode_t::TPUDNN_GE,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_GE, self.dim() == 0 ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return ge(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ge.Tensor_out", greater_or_equal_out_tpu);
}

Tensor &greater_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  binary_impl(
      self, other, out, tpu::GREATER, CompareMode_t::TPUDNN_GT,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_GT, self.dim() == 0 ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return gt(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("gt.Tensor_out", greater_out_tpu); }

Tensor &less_than_or_equal_out_tpu(const Tensor &self, const Tensor &other,
                                   Tensor &out) {
  binary_impl(
      self, other, out, tpu::LESS_THAN_OR_EQUAL,
      CompareMode_t::TPUDNN_LE,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_LE, self.dim() == 0 ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return le(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("le.Tensor_out", less_than_or_equal_out_tpu);
}

Tensor &less_than_out_tpu(const Tensor &self, const Tensor &other,
                          Tensor &out) {
  binary_impl(
      self, other, out, tpu::LESS_THAN, CompareMode_t::TPUDNN_LT,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_LT, self.dim() == 0 ? 0 : 1);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return lt(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("lt.Tensor_out", less_than_out_tpu); }

Tensor &not_equal_out_tpu(const Tensor &self, const Tensor &other,
                          Tensor &out) {
  binary_impl(
      self, other, out, tpu::NOT_EQUAL, CompareMode_t::TPUDNN_NE,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnCompareConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toFloat() : other.item().toFloat(),
            CompareMode_t::TPUDNN_NE, 0);
      },
      tpudnnCompareAsync,
      [](const Tensor &a, const Tensor &b) { return ne(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("ne.Tensor_out", not_equal_out_tpu); }

Tensor &shift_left_out_tpu(const Tensor &self, const Tensor &other,
                           Tensor &out) {
  binary_impl(
      self, other, out, tpu::SHIFT_LEFT, true,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnShiftConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toChar() : other.item().toChar(),
            true);
      },
      tpudnnShiftAsync,
      [](const Tensor &a, const Tensor &b) { TORCH_CHECK(0); return a; });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_left_shift.Tensor_out", shift_left_out_tpu);
}

Tensor &shift_right_arithmetic_out_tpu(const Tensor &self, const Tensor &other,
                                       Tensor &out) {
  binary_impl(
      self, other, out, tpu::SHIFT_RIGHT_ARITHMETIC, false,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnShiftConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toChar() : other.item().toChar(),
            false);
      },
      tpudnnShiftAsync,
      [](const Tensor &a, const Tensor &b) { TORCH_CHECK(0);return a; });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_right_shift.Tensor_out", shift_right_arithmetic_out_tpu);
}

Tensor &minimum_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  binary_impl(
      self, other, out, tpu::MINIMUM,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnMinimumConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toFloat() : other.item().toFloat());
      },
      tpudnnMinimumAsync,
      [](const Tensor &a, const Tensor &b) { return minimum(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("minimum.out", minimum_out_tpu); }

Tensor &maximum_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  binary_impl(
      self, other, out, tpu::MAXIMUM,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other, Tensor &out) {
        return tpudnnMaximumConstAsync(
            handle,
            tpu::TPUGenerateTpudnnTensor(handle, self.dim() == 0 ? other : self),
            tpu::TPUGenerateTpudnnTensor(handle, out),
            self.dim() == 0 ? self.item().toFloat() : other.item().toFloat());
      },
      tpudnnMaximumAsync,
      [](const Tensor &a, const Tensor &b) { return maximum(a, b); });
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
  binary_impl(
      self, other, out, tpu::POW,
      [](tpudnnHandle_t handle, const Tensor &self, const Tensor &other,
         Tensor &out) { return TPUDNN_STATUS_FAILED; },
      tpudnnPowerAsync,
      [](const Tensor &a, const Tensor &b) { return pow(a, b); });
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("pow.Tensor_Tensor_out", pow_out_tpu);
}

Tensor &pow_c_out_tpu(const Tensor &self, const Scalar &exponent, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
  TORCH_CHECK(exponent.toDouble() > 0);
  if (self.dim() == 0) {
    Tensor out_cpu = pow(self.cpu(), exponent);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    SHOW_TENSOR_OP(self, out);
    return out;
  }
  if (exponent.toDouble() == 1.0) {
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), self.data_ptr(), out.nbytes());
    SHOW_TENSOR_OP(self, out);
    return out;
  }
  TORCH_CHECK(exponent.toDouble() > 0);

  TIMING_START;
  auto handle = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnPowerScalarAsync(
      handle,
      tpu::TPUGenerateTpudnnTensor(handle, self),
      tpu::TPUGenerateTpudnnTensor(handle, out),
      exponent.toFloat());
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::POWC);
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("pow.Tensor_Scalar_out", pow_c_out_tpu);
}

Tensor &c_pow_out_tpu(const Scalar &self, const Tensor &exponent, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(exponent);
  CHECK_TENSOR_IN_DEVICE(out);
  TORCH_CHECK(self.toDouble() > 0);
  if (exponent.dim() == 0) {
    Tensor out_cpu = pow(self, exponent.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    SHOW_TENSOR_OP(exponent, out);
    return out;
  }

  TIMING_START;
  auto handle = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnScalarPowerAsync(
      handle,
      tpu::TPUGenerateTpudnnTensor(handle, exponent),
      tpu::TPUGenerateTpudnnTensor(handle, out),
      self.toFloat());
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END(tpu::CPOW);
  SHOW_TENSOR_OP(exponent, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("pow.Scalar_out", c_pow_out_tpu); }


Tensor &atan2_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(other);
  // self is scalar and other is scalar
  auto handle = c10_tpu::getCurrentTPUStream();
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = atan2(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    auto status = tpudnnScalarAtan2Async(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, other.to(out.dtype())),
        tpu::TPUGenerateTpudnnTensor(handle, out),
        self.item().toFloat());
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(tpu::ATAN2);
  } else if (other.dim() == 0) {
    TIMING_START;

    auto status = tpudnnAtan2ScalarAsync(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self.to(out.dtype())),
        tpu::TPUGenerateTpudnnTensor(handle, out),
        other.item().toFloat());
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    TIMING_END(tpu::ATAN2);
  } else {
    TIMING_START;
    auto status = tpudnnAtan2Async(
        handle,
        tpu::TPUGenerateTpudnnTensor(handle, self.to(torch::kFloat)),
        tpu::TPUGenerateTpudnnTensor(handle, other.to(torch::kFloat)),
        tpu::TPUGenerateTpudnnTensor(handle, out));
    TIMING_END(tpu::ATAN2);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
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
