#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/quantized/QTensorImpl.h>

#include "TPUTorchUtils.h"

#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"
#include <cmath>
#include <float.h>

#include <tpuDNN.h>

namespace at {

Tensor &binary_op_tpu(const Tensor &self, const Tensor &other,
                      const Scalar &alpha, Tensor &out, int binary_type) {
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
    if (self.dim() == 0) {
      TIMING_START;
      auto status = sgdnnBinaryC(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other_),
          self.item().toFloat() * alpha.toFloat(),
          tpu::TPUGenerateSgdnnTensor(out_), binary_type, 1);
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::BINARYOP_C);
    } else {
      TIMING_START;
      auto status = sgdnnBinaryC(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self_),
          other.item().toFloat() * alpha.toFloat(),
          tpu::TPUGenerateSgdnnTensor(out_), binary_type, 0);
      TORCH_CHECK(status == SG_SUCCESS);
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
    TIMING_START;
    auto other_ =
        other.dtype() == self.dtype() ? other : other.to(self.dtype());

    // auto stream = c10_tpu::getCurrentTPUStream();
    // auto status = tpudnnBinaryAsync(
    //     stream,
    //     tpu::TPUGenerateTpudnnTensor(stream, self),
    //     tpu::TPUGenerateTpudnnTensor(stream, other_),
    //     alpha.toDouble(),
    //     tpu::TPUGenerateTpudnnTensor(stream, out), binary_type);
    // TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
    auto status = sgdnnBinary(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        tpu::TPUGenerateSgdnnTensor(other_), alpha.toDouble(),
        tpu::TPUGenerateSgdnnTensor(out), binary_type);
    TORCH_CHECK(status == SG_SUCCESS);
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
    for (int i = 0; i < max_dim; i++) {
      TORCH_CHECK(self_shape[i] == other_shape[i] || self_shape[i] == 1 ||
                      other_shape[i] == 1,
                  "The size of tensor a (%d) must match the size of tensor b "
                  "(%d) at non-signleton dimension %d",
                  self_shape[i], other_shape[i], i)
    }
    SgdnnTensor_t other_t;
    if (self.dtype() != other.dtype()) {
      // 创建一个新的 Tensor 对象，使用 self 的数据类型和 other 的大小和设备选项
      at::Tensor other_new = at::empty_like(other, self.options());
      other_new.copy_(other);
      // 使用新的 Tensor 对象进行操作，而不修改原始的 const Tensor &
      other_t = tpu::TPUGenerateSgdnnTensor(other_new);
    } else {
      other_t = tpu::TPUGenerateSgdnnTensor(other);
    }
    SgdnnTensor_t self_t = tpu::TPUGenerateSgdnnTensor(self);
    // SgdnnTensor_t other_t = tpu::TPUGenerateSgdnnTensor(other);
    SgdnnTensor_t out_t = tpu::TPUGenerateSgdnnTensor(out);

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

    TIMING_START;
    auto status = sgdnnBinaryBcast(tpu::TPUGetDeviceResource(), self_t, other_t,
                                   alpha.toDouble(), out_t, binary_type);
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::BINARYOP_BCAST)
  }
  return out;
}

Tensor &add_out_tpu(const Tensor &self, const Tensor &other,
                    const Scalar &alpha, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    // LOG(WARNING) << "add_out not contiguous, use stride copy"
    //              << " self.is_contiguous : " << self.is_contiguous()
    //              << " other.is_contiguous : " << other.is_contiguous()
    //              << " out.is_contiguous : " << out.is_contiguous()
    //              << " [TODO]  no use strided copy";
    if (out.is_contiguous()) {
      out = add(self.contiguous(), other.contiguous(), alpha);
    } else {
      auto out_ = add(self.contiguous(), other.contiguous(), alpha);
      TIMING_START;
      sgdnnStridedCopy(tpu::TPUGetDeviceResource(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
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
    // LOG(WARNING) << "add_out not contiguous, use stride copy"
    //              << " self.is_contiguous : " << self.is_contiguous()
    //              << " other.is_contiguous : " << other.is_contiguous()
    //              << " out.is_contiguous : " << out.is_contiguous()
    //              << " [TODO]  no use strided copy";
    if (out.is_contiguous()) {
      out = sub(self.contiguous(), other.contiguous(), alpha);
    } else {
      auto out_ = sub(self.contiguous(), other.contiguous(), alpha);
      TIMING_START;
      sgdnnStridedCopy(tpu::TPUGetDeviceResource(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
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
    // LOG(WARNING) << "mul_out not contiguous, use stride copy"
    //              << " self.is_contiguous : " << self.is_contiguous()
    //              << " other.is_contiguous : " << other.is_contiguous()
    //              << " out.is_contiguous : " << out.is_contiguous()
    //              << " [TODO]  no use strided copy";
    if (out.is_contiguous()) {
      out = mul(self.contiguous(), other.contiguous());
    } else {
      auto out_ = mul(self.contiguous(), other.contiguous());
      TIMING_START;
      sgdnnStridedCopy(tpu::TPUGetDeviceResource(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
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
    // LOG(WARNING) << "div out use strided copy because of, "
    //              << " self_is_contiguous : " << self.is_contiguous()
    //              << " other_is_contiguos : " << other.is_contiguous()
    //              << " out_is_congiguous : " << out.is_contiguous()
    //              << " [TODO] no use strided copy";
    if (out.is_contiguous()) {
      out = div(self.contiguous(), other.contiguous());
    } else {
      auto out_ = div(self.contiguous(), other.contiguous());
      TIMING_START;
      sgdnnStridedCopy(tpu::TPUGetDeviceResource(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
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

Tensor &bitwise_xor_out_tpu(const Tensor &self, const Tensor &other,
                            Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = bitwise_xor(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {
    TIMING_START;
    auto status = sgdnnElementBitwiseC(tpu::TPUGetDeviceResource(),
                                       tpu::TPUGenerateSgdnnTensor(other),
                                       self.item().toInt(),
                                       0, // 0 for xor, 1 for and, 2 for or
                                       tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::BITWISE_XOR_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnElementBitwiseC(tpu::TPUGetDeviceResource(),
                                       tpu::TPUGenerateSgdnnTensor(self),
                                       other.item().toInt(),
                                       0, // 0 for xor, 1 for and, 2 for or
                                       tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::BITWISE_XOR_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnElementBitwise(tpu::TPUGetDeviceResource(),
                                        tpu::TPUGenerateSgdnnTensor(self),
                                        tpu::TPUGenerateSgdnnTensor(other),
                                        0, // 0 for xor, 1 for and, 2 for or
                                        tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::BITWISE_XOR);
    } else {
      TIMING_START;
      auto status = sgdnnElementBitwiseBcast(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          0, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::BITWISE_XOR_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_xor.Tensor_out", bitwise_xor_out_tpu);
}

Tensor &bitwise_and_out_tpu(const Tensor &self, const Tensor &other,
                            Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = bitwise_and(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    auto status = sgdnnElementBitwiseC(tpu::TPUGetDeviceResource(),
                                       tpu::TPUGenerateSgdnnTensor(other),
                                       self.item().toInt(),
                                       1, // 0 for xor, 1 for and, 2 for or
                                       tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::BITWISE_AND_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnElementBitwiseC(tpu::TPUGetDeviceResource(),
                                       tpu::TPUGenerateSgdnnTensor(self),
                                       other.item().toInt(),
                                       1, // 0 for xor, 1 for and, 2 for or
                                       tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::BITWISE_AND_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto self_ =
          other.dtype() == self.dtype() ? self : self.to(other.dtype());
      out = out.dtype() == other.dtype() ? out : out.to(other.dtype());
      auto status = sgdnnElementBitwise(tpu::TPUGetDeviceResource(),
                                        tpu::TPUGenerateSgdnnTensor(self_),
                                        tpu::TPUGenerateSgdnnTensor(other),
                                        1, // 0 for xor, 1 for and, 2 for or
                                        tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::BITWISE_AND);
      SHOW_TENSOR_OP(self, other, out);
      return out;
    } else {
      TIMING_START;
      auto status = sgdnnElementBitwiseBcast(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          1, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::BITWISE_AND_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_and.Tensor_out", bitwise_and_out_tpu);
}

Tensor &bitwise_or_out_tpu(const Tensor &self, const Tensor &other,
                           Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = bitwise_or(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    auto status = sgdnnElementBitwiseC(tpu::TPUGetDeviceResource(),
                                       tpu::TPUGenerateSgdnnTensor(other),
                                       self.item().toInt(),
                                       2, // 0 for xor, 1 for and, 2 for or
                                       tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::BITWISE_OR_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnElementBitwiseC(tpu::TPUGetDeviceResource(),
                                       tpu::TPUGenerateSgdnnTensor(self),
                                       other.item().toInt(),
                                       2, // 0 for xor, 1 for and, 2 for or
                                       tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::BITWISE_OR_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnElementBitwise(tpu::TPUGetDeviceResource(),
                                        tpu::TPUGenerateSgdnnTensor(self),
                                        tpu::TPUGenerateSgdnnTensor(other),
                                        2, // 0 for xor, 1 for and, 2 for or
                                        tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::BITWISE_OR);
    } else {
      TIMING_START;
      auto status = sgdnnElementBitwiseBcast(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          2, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::BITWISE_OR_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_or.Tensor_out", bitwise_or_out_tpu);
}

Tensor &equal_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = eq(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 0, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::EQUAL_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 0, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::EQUAL_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnComparision(tpu::TPUGetDeviceResource(),
                                     tpu::TPUGenerateSgdnnTensor(self),
                                     tpu::TPUGenerateSgdnnTensor(other), 0,
                                     tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::EQUAL);
    } else {
      TIMING_START;
      auto status = sgdnnComparisionBcast(tpu::TPUGetDeviceResource(),
                                          tpu::TPUGenerateSgdnnTensor(self),
                                          tpu::TPUGenerateSgdnnTensor(other), 0,
                                          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::EQUAL_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("eq.Tensor_out", equal_out_tpu); }

Tensor &greater_or_equal_out_tpu(const Tensor &self, const Tensor &other,
                                 Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = ge(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 3, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::GREATER_OR_EQUAL_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 3, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::GREATER_OR_EQUAL_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnComparision(tpu::TPUGetDeviceResource(),
                                     tpu::TPUGenerateSgdnnTensor(self),
                                     tpu::TPUGenerateSgdnnTensor(other), 3,
                                     tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::GREATER_OR_EQUAL);
    } else {
      TIMING_START;
      auto status = sgdnnComparisionBcast(tpu::TPUGetDeviceResource(),
                                          tpu::TPUGenerateSgdnnTensor(self),
                                          tpu::TPUGenerateSgdnnTensor(other), 3,
                                          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::GREATER_OR_EQUAL_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ge.Tensor_out", greater_or_equal_out_tpu);
}

Tensor &greater_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = gt(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 2, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::GREATER_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 2, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::GREATER_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnComparision(tpu::TPUGetDeviceResource(),
                                     tpu::TPUGenerateSgdnnTensor(self),
                                     tpu::TPUGenerateSgdnnTensor(other), 2,
                                     tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::GREATER);
    } else {
      TIMING_START;
      auto status = sgdnnComparisionBcast(tpu::TPUGetDeviceResource(),
                                          tpu::TPUGenerateSgdnnTensor(self),
                                          tpu::TPUGenerateSgdnnTensor(other), 2,
                                          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::GREATER_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("gt.Tensor_out", greater_out_tpu); }

Tensor &less_than_or_equal_out_tpu(const Tensor &self, const Tensor &other,
                                   Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);
  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = le(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 5, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::LESS_THAN_OR_EQUAL_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 5, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::LESS_THAN_OR_EQUAL_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnComparision(tpu::TPUGetDeviceResource(),
                                     tpu::TPUGenerateSgdnnTensor(self),
                                     tpu::TPUGenerateSgdnnTensor(other), 5,
                                     tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::LESS_THAN_OR_EQUAL);
    } else {
      TIMING_START;
      auto status = sgdnnComparisionBcast(tpu::TPUGetDeviceResource(),
                                          tpu::TPUGenerateSgdnnTensor(self),
                                          tpu::TPUGenerateSgdnnTensor(other), 5,
                                          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::LESS_THAN_OR_EQUAL_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("le.Tensor_out", less_than_or_equal_out_tpu);
}

Tensor &less_than_out_tpu(const Tensor &self, const Tensor &other,
                          Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = lt(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 4, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::LESS_THAN_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 4, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::LESS_THAN_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnComparision(tpu::TPUGetDeviceResource(),
                                     tpu::TPUGenerateSgdnnTensor(self),
                                     tpu::TPUGenerateSgdnnTensor(other), 4,
                                     tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::LESS_THAN);
    } else {
      TIMING_START;
      auto self_  = self.dtype() == torch::kInt64 ? self.to(torch::kInt32) : self;
      auto other_ = other.dtype() == torch::kInt64 ? other.to(torch::kInt32) : other;
      auto out_   = out.dtype() == torch::kInt64 ? out.to(torch::kInt32) : out;
      auto status = sgdnnComparisionBcast(tpu::TPUGetDeviceResource(),
                                          tpu::TPUGenerateSgdnnTensor(self_),
                                          tpu::TPUGenerateSgdnnTensor(other_), 4,
                                          tpu::TPUGenerateSgdnnTensor(out_));
      if(out.dtype() == torch::kInt64)
        out = out_.to(torch::kInt64);
      else
        out = out_;
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::LESS_THAN_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("lt.Tensor_out", less_than_out_tpu); }

Tensor &not_equal_out_tpu(const Tensor &self, const Tensor &other,
                          Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    TIMING_START;
    auto out_cpu = ne(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    TIMING_START;
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 1, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::NOT_EQUAL_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 1, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::NOT_EQUAL_C);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnComparision(tpu::TPUGetDeviceResource(),
                                     tpu::TPUGenerateSgdnnTensor(self),
                                     tpu::TPUGenerateSgdnnTensor(other), 1,
                                     tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::NOT_EQUAL);
    } else {
      TIMING_START;
      auto status = sgdnnComparisionBcast(tpu::TPUGetDeviceResource(),
                                          tpu::TPUGenerateSgdnnTensor(self),
                                          tpu::TPUGenerateSgdnnTensor(other), 1,
                                          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::NOT_EQUAL_BCAST);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("ne.Tensor_out", not_equal_out_tpu); }

Tensor &shift_left_out_tpu(const Tensor &self, const Tensor &other,
                           Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    // not implemented now
    //     auto out_cpu = shift_left(self.cpu(), other.cpu());
    //     tpu::TPUCopyHostToDevice(out.data_ptr(),
    //     out_cpu.contiguous().data_ptr(), out.nbytes());
  } else if (self.dim() == 0) {
    TIMING_START;
    auto status = sgdnnShiftLeftC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toChar(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::SHIFT_LEFT_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnShiftLeftC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toChar(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::SHIFT_LEFT_C);
  } else if (self.dim() == other.dim()) {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnShiftLeft(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::SHIFT_LEFT);
    } else {
      TIMING_START;
      auto status = sgdnnShiftLeftBcast(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::SHIFT_LEFT_BCAST);
    }
  } else {
    TORCH_CHECK(false, "unsupported dims");
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_left_shift.Tensor_out", shift_left_out_tpu);
}

Tensor &shift_right_arithmetic_out_tpu(const Tensor &self, const Tensor &other,
                                       Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0 && other.dim() == 0) {
    // auto out_cpu = shift_right_arithmetic(self.cpu(), other.cpu());
    // tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
    // out.nbytes());
    TORCH_CHECK(false, "unsupported dims");
  } else if (self.dim() == 0) {
    TIMING_START;
    auto status = sgdnnShiftRightArithmeticC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toInt(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::SHIFT_RIGHT_ARITHMETIC_C);
  } else if (other.dim() == 0) {
    TIMING_START;
    auto status = sgdnnShiftRightArithmeticC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toInt(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::SHIFT_RIGHT_ARITHMETIC_C);
  } else if (self.dim() == other.dim()) {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START;
      auto status = sgdnnShiftRightArithmetic(
          tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::SHIFT_RIGHT_ARITHMETIC);
    } else {
      TIMING_START;
      // auto status =
      // sgdnnShiftRightArithmeticBcast(tpu::TPUGetDeviceResource(),
      //                                      tpu:: TPUGenerateSgdnnTensor (
      //                                      self ), tpu::
      //                                      TPUGenerateSgdnnTensor ( other ),
      //                                      tpu:: TPUGenerateSgdnnTensor ( out
      //                                      ) );
      // TORCH_CHECK ( status == SG_SUCCESS );
      TIMING_END(tpu::SHIFT_RIGHT_ARITHMETIC_BCAST);
      TORCH_CHECK(false, "unsupported dims");
    }
  } else {
    TORCH_CHECK(false, "unsupported dims");
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("bitwise_right_shift.Tensor_out", shift_right_arithmetic_out_tpu);
}

Tensor &minimum_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
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
    auto out_cpu = minimum(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
    TIMING_START;
    auto status = sgdnnMinimumC(
        tpu::TPUGetDeviceResource(),
        tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::MINIMUM);
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
    TIMING_START;
    auto status = sgdnnMinimumC(
        tpu::TPUGetDeviceResource(),
        tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::MINIMUM);
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
      TIMING_START;
      auto status =
          sgdnnMinimum(tpu::TPUGetDeviceResource(),
                       tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                       tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                       tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::MINIMUM);
    } else {
      // The shapes of self and other are not the same, need to broadcast
      TIMING_START;
      auto status =
          sgdnnMinimumBcast(tpu::TPUGetDeviceResource(),
                            tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                            tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                            tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::MINIMUM);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("minimum.out", minimum_out_tpu); }

Tensor &maximum_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
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
    auto out_cpu = maximum(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
    TIMING_START;
    auto status = sgdnnMaximumC(
        tpu::TPUGetDeviceResource(),
        tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::MAXIMUM);
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
    TIMING_START;
    auto status = sgdnnMaximumC(
        tpu::TPUGetDeviceResource(),
        tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::MAXIMUM);
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
      TIMING_START;

      auto status =
          sgdnnMaximum(tpu::TPUGetDeviceResource(),
                       tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                       tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                       tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::MAXIMUM);
    } else {
      // The shapes of self and other are not the same, need to broadcast
      TIMING_START;

      auto status =
          sgdnnMaximumBcast(tpu::TPUGetDeviceResource(),
                            tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                            tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                            tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::MAXIMUM);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("maximum.out", maximum_out_tpu); }

Tensor &atan2_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
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
    auto out_cpu = atan2(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
    TIMING_START;

    auto status =
        sgdnnAtan2C(tpu::TPUGetDeviceResource(), *scalar.data_ptr<float>(),
                    tpu::TPUGenerateSgdnnTensor(other.to(torch::kFloat)),
                    tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::ATAN2);
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
    TIMING_START;

    auto status = sgdnnAtan2_C(
        tpu::TPUGetDeviceResource(),
        tpu::TPUGenerateSgdnnTensor(self.to(torch::kFloat)),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::ATAN2);
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
      TIMING_START;

      auto status =
          sgdnnAtan2(tpu::TPUGetDeviceResource(),
                     tpu::TPUGenerateSgdnnTensor(self.to(torch::kFloat)),
                     tpu::TPUGenerateSgdnnTensor(other.to(torch::kFloat)),
                     tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::ATAN2);
    } else {
      // The shapes of self and other are not the same, need to broadcast
      TIMING_START;

      auto status =
          sgdnnAtan2Bcast(tpu::TPUGetDeviceResource(),
                          tpu::TPUGenerateSgdnnTensor(self.to(torch::kFloat)),
                          tpu::TPUGenerateSgdnnTensor(other.to(torch::kFloat)),
                          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::ATAN2);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("atan2.out", atan2_out_tpu); }

Tensor &pow_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);
  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = pow(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    return out;
  }

  if (self.dim() == 0) {
    return pow_outf(self.item(), other, out);
  }
  if (other.dim() == 0) {
    return pow_outf(self, other.item(), out);
  }

  if (tpu::TPUIsSameShape(self, other)) {
    TIMING_START;

    auto status = sgdnnPow(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::POW);
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
    for (int i = 0; i < max_dim; i++) {
      TORCH_CHECK(self_shape[i] == other_shape[i] || self_shape[i] == 1 ||
                      other_shape[i] == 1,
                  "The size of tensor a (%d) must match the size of tensor b "
                  "(%d) at non-signleton dimension %d",
                  self_shape[i], other_shape[i], i)
    }

    SgdnnTensor_t self_t = tpu::TPUGenerateSgdnnTensor(self);
    SgdnnTensor_t other_t = tpu::TPUGenerateSgdnnTensor(other);
    SgdnnTensor_t out_t = tpu::TPUGenerateSgdnnTensor(out);

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
    TIMING_START;

    auto status =
        sgdnnPowBcast(tpu::TPUGetDeviceResource(), self_t, other_t, out_t);
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::POW_BCAST);
  }
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
#if 0
  CPU_IMPL_WARNING();
  auto out_cpu = pow( self.cpu(), exponent );
  out = out_cpu.to(out.device());
#else
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

  auto status =
      sgdnnPowC(tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
                exponent.toDouble(), tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == SG_SUCCESS);
  TIMING_END(tpu::POWC);
#endif
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
  TIMING_START;

  auto status = sgdnnCPow(tpu::TPUGetDeviceResource(),
                          tpu::TPUGenerateSgdnnTensor(exponent),
                          self.toDouble(), tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == SG_SUCCESS);
  TIMING_END(tpu::CPOW);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("pow.Scalar_out", c_pow_out_tpu); }

Tensor &fmax_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
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
    auto out_cpu = fmax(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
    TIMING_START;

    auto status =
        sgdnnFmaxC(tpu::TPUGetDeviceResource(),
                   tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                   *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::FMAX);
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
    TIMING_START;

    auto status =
        sgdnnFmaxC(tpu::TPUGetDeviceResource(),
                   tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                   *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::FMAX);
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
      TIMING_START;

      auto status =
          sgdnnFmax(tpu::TPUGetDeviceResource(),
                    tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                    tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                    tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::FMAX);
    } else {
      // The shapes of self and other are not the same, need to broadcast
      TIMING_START;

      auto status =
          sgdnnFmaxBcast(tpu::TPUGetDeviceResource(),
                         tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::FMAX);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("fmax.out", fmax_out_tpu); }

Tensor &fmin_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
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
    auto out_cpu = fmin(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
    TIMING_START;

    auto status =
        sgdnnFminC(tpu::TPUGetDeviceResource(),
                   tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                   *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::FMIN);
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
    TIMING_START;

    auto status =
        sgdnnFminC(tpu::TPUGetDeviceResource(),
                   tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                   *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::FMIN);
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
      TIMING_START;

      auto status =
          sgdnnFmin(tpu::TPUGetDeviceResource(),
                    tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                    tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                    tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::FMIN);
    } else {
      // The shapes of self and other are not the same, need to broadcast
      TIMING_START;

      auto status =
          sgdnnFminBcast(tpu::TPUGetDeviceResource(),
                         tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == SG_SUCCESS);
      TIMING_END(tpu::FMIN);
    }
  }
  SHOW_TENSOR_OP(self, other, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("fmin.out", fmin_out_tpu); }

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

Tensor &less_than_or_equal_scalar_out_tpu(const Tensor &self,
                                          const Scalar &other, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = le(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else {
    TIMING_START;

    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 5, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::LESS_THAN_OR_EQUAL_C);
  }
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("le.Scalar_out", less_than_or_equal_scalar_out_tpu);
}

Tensor &less_than_scalar_out_tpu(const Tensor &self, const Scalar &other,
                                 Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = lt(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else {
    TIMING_START;

    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 4, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::LESS_THAN_C);
  }
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("lt.Scalar_out", less_than_scalar_out_tpu);
}

Tensor &greater_or_equal_scalar_out_tpu(const Tensor &self, const Scalar &other,
                                        Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = ge(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else {
    TIMING_START;

    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 3, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::GREATER_OR_EQUAL_C);
  }
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ge.Scalar_out", greater_or_equal_scalar_out_tpu);
}

Tensor &greater_scalar_out_tpu(const Tensor &self, const Scalar &other,
                               Tensor &out) {
  auto self_ = self.is_contiguous() ? self : self.contiguous();
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self_);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self_.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = gt(self_.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else {
    TIMING_START;

    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self_),
        other.toFloat(), 2, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::GREATER_C);
  }
  SHOW_TENSOR_OP(self_, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("gt.Scalar_out", greater_scalar_out_tpu);
}

Tensor &not_equal_scalar_out_tpu(const Tensor &self, const Scalar &other,
                                 Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = ne(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else {
    TIMING_START;

    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 1, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::NOT_EQUAL_C);
  }
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ne.Scalar_out", not_equal_scalar_out_tpu);
}

Tensor &equal_scalar_out_tpu(const Tensor &self, const Scalar &other,
                             Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = eq(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    TIMING_END(tpu::CPU_LAYER);
  } else {
    TIMING_START;

    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    auto status = sgdnnComparisionC(
        tpu::TPUGetDeviceResource(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 0, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == SG_SUCCESS);
    TIMING_END(tpu::EQUAL_C);
  }
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("eq.Scalar_out", equal_scalar_out_tpu);
}

} // namespace at
