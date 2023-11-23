#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/quantized/QTensorImpl.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"
#include <cmath>
#include <float.h>

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

  if (self.dim() == 0 || other.dim() == 0) {
    // tensor op scalar
    if (self.dim() == 0) {
      if (self.dtype() == caffe2::TypeMeta::Make<long>() ||
          self.dtype() == caffe2::TypeMeta::Make<int>() ||
          self.dtype() == caffe2::TypeMeta::Make<short>()) {
        Tensor scalar = self.cpu().to(torch::kLong);
        TIMING_START
        bm_status_t status = sgdnnBinaryC(
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
            *scalar.data_ptr<long>() * alpha.toFloat(),
            tpu::TPUGenerateSgdnnTensor(out), binary_type, 1);
        TORCH_CHECK(status == BM_SUCCESS);
        TIMING_END(tpu::BINARYOP_C)
      } else {
        Tensor scalar = self.cpu().to(torch::kFloat);
        TIMING_START
        bm_status_t status = sgdnnBinaryC(
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
            *scalar.data_ptr<float>() * alpha.toFloat(),
            tpu::TPUGenerateSgdnnTensor(out), binary_type, 1);
        TORCH_CHECK(status == BM_SUCCESS);
        TIMING_END(tpu::BINARYOP_C)
      }
    } else {
      if (other.dtype() == caffe2::TypeMeta::Make<long>() ||
          other.dtype() == caffe2::TypeMeta::Make<int>() ||
          other.dtype() == caffe2::TypeMeta::Make<short>()) {
        Tensor scalar = other.cpu().to(torch::kLong);
        TIMING_START
        bm_status_t status = sgdnnBinaryC(
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
            *scalar.data_ptr<long>() * alpha.toFloat(),
            tpu::TPUGenerateSgdnnTensor(out), binary_type, 0);
        TORCH_CHECK(status == BM_SUCCESS);
        TIMING_END(tpu::BINARYOP_C)
      } else {
        Tensor scalar = other.cpu().to(torch::kFloat);
        TIMING_START
        bm_status_t status = sgdnnBinaryC(
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
            *scalar.data_ptr<float>() * alpha.toFloat(),
            tpu::TPUGenerateSgdnnTensor(out), binary_type, 0);
        TORCH_CHECK(status == BM_SUCCESS);
        TIMING_END(tpu::BINARYOP_C)
      }
    }
    return out;
  }

  if (tpu::TPUIsSameShape(self, other)) {
    TIMING_START
    bm_status_t status = sgdnnBinary(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        tpu::TPUGenerateSgdnnTensor(other), alpha.toDouble(),
        tpu::TPUGenerateSgdnnTensor(out), binary_type);
    TORCH_CHECK(status == BM_SUCCESS);
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

    TIMING_START
    bm_status_t status =
        sgdnnBinaryBcast(tpu::TPUGetDeviceHandle(), self_t, other_t,
                         alpha.toDouble(), out_t, binary_type);
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::BINARYOP_BCAST)
  }
  return out;
}

Tensor &add_out_tpu(const Tensor &self, const Tensor &other,
                    const Scalar &alpha, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    LOG(WARNING) << "add_out not contiguous, use stride copy"
                 << " self.is_contiguous : " << self.is_contiguous()
                 << " other.is_contiguous : " << other.is_contiguous()
                 << " out.is_contiguous : " << out.is_contiguous()
                 << " [TODO]  no use strided copy";
    if (out.is_contiguous()) {
      out = add(self.contiguous(), other.contiguous(), alpha);
    } else {
      auto out_ = add(self.contiguous(), other.contiguous(), alpha);
      sgdnnStridedCopy(tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
    }
    return out;
  }
  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = add(self.cpu(), other.cpu(), alpha);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    return out;
  }
  // 0:add, 1:sub, 2:mul, 3:div
  return binary_op_tpu(self, other, alpha, out, 0);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("add.out", add_out_tpu); }

Tensor &sub_out_tpu(const Tensor &self, const Tensor &other,
                    const Scalar &alpha, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    LOG(WARNING) << "add_out not contiguous, use stride copy"
                 << " self.is_contiguous : " << self.is_contiguous()
                 << " other.is_contiguous : " << other.is_contiguous()
                 << " out.is_contiguous : " << out.is_contiguous()
                 << " [TODO]  no use strided copy";
    if (out.is_contiguous()) {
      out = sub(self.contiguous(), other.contiguous(), alpha);
    } else {
      auto out_ = sub(self.contiguous(), other.contiguous(), alpha);
      sgdnnStridedCopy(tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
    }
    return out;
  }
  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = sub(self.cpu(), other.cpu(), alpha);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    return out;
  }
  // 0:add, 1:sub, 2:mul, 3:div
  return binary_op_tpu(self, other, alpha, out, 1);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("sub.out", sub_out_tpu); }

Tensor &mul_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    LOG(WARNING) << "mul_out not contiguous, use stride copy"
                 << " self.is_contiguous : " << self.is_contiguous()
                 << " other.is_contiguous : " << other.is_contiguous()
                 << " out.is_contiguous : " << out.is_contiguous()
                 << " [TODO]  no use strided copy";
    if (out.is_contiguous()) {
      out = mul(self.contiguous(), other.contiguous());
    } else {
      auto out_ = mul(self.contiguous(), other.contiguous());
      sgdnnStridedCopy(tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
    }
    return out;
  }
  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = mul(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    return out;
  }
  // 0:add, 1:sub, 2:mul, 3:div
  return binary_op_tpu(self, other, 1, out, 2);
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("mul.out", mul_out_tpu); }

Tensor &div_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (!self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous()) {
    LOG(WARNING) << "div out use strided copy because of, "
                 << " self_is_contiguous : " << self.is_contiguous()
                 << " other_is_contiguos : " << other.is_contiguous()
                 << " out_is_congiguous : " << out.is_contiguous()
                 << " [TODO] no use strided copy";
    if (out.is_contiguous()) {
      out = div(self.contiguous(), other.contiguous());
    } else {
      auto out_ = div(self.contiguous(), other.contiguous());
      sgdnnStridedCopy(tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor(out_),
                       tpu::TPUGenerateSgdnnTensor(out));
    }
    return out;
  }

  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = div(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
    return out;
  }
  const Tensor self_f = self.to(torch::kFloat);
  const Tensor other_f = other.to(torch::kFloat);
  Tensor out_f = out.to(torch::kFloat);
  // 0:add, 1:sub, 2:mul, 3:div
  binary_op_tpu(self_f, other_f, 1, out_f, 3);
  tpu::TPUCopyDeviceToDevice(out.data_ptr(), out_f.to(out.dtype()).data_ptr(),
                             out.nbytes());
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

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toInt(),
        0, // 0 for xor, 1 for and, 2 for or
        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_XOR_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toInt(),
        0, // 0 for xor, 1 for and, 2 for or
        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_XOR_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwise(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          0, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_XOR, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwiseBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          0, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_XOR_BCAST,
                                       timer.ElapsedUS());
#endif
    }
  }

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
    auto out_cpu = bitwise_and(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toInt(),
        1, // 0 for xor, 1 for and, 2 for or
        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_AND_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toInt(),
        1, // 0 for xor, 1 for and, 2 for or
        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_AND_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwise(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          1, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_AND, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwiseBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          1, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_AND_BCAST,
                                       timer.ElapsedUS());
#endif
    }
  }

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
    auto out_cpu = bitwise_or(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toInt(),
        2, // 0 for xor, 1 for and, 2 for or
        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_OR_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toInt(),
        2, // 0 for xor, 1 for and, 2 for or
        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_OR_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwise(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          2, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_OR, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwiseBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other),
          2, // 0 for xor, 1 for and, 2 for or
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_OR_BCAST,
                                       timer.ElapsedUS());
#endif
    }
  }

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
    auto out_cpu = eq(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 0, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::EQUAL_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 0, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::EQUAL_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparision(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 0,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::EQUAL, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparisionBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 0,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::EQUAL_BCAST, timer.ElapsedUS());
#endif
    }
  }

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
    auto out_cpu = ge(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 3, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::GREATER_OR_EQUAL_C,
                                     timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 3, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::GREATER_OR_EQUAL_C,
                                     timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparision(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 3,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::GREATER_OR_EQUAL,
                                       timer.ElapsedUS());
#endif

    } else {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparisionBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 3,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::GREATER_OR_EQUAL_BCAST,
                                       timer.ElapsedUS());
#endif
    }
  }

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
    auto out_cpu = gt(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 2, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::GREATER_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 2, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::GREATER_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparision(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 2,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::GREATER, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparisionBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 2,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::GREATER_BCAST, timer.ElapsedUS());
#endif
    }
  }

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
    auto out_cpu = le(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 5, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_OR_EQUAL_C,
                                     timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 5, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_OR_EQUAL_C,
                                     timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparision(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 5,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_OR_EQUAL,
                                       timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparisionBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 5,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_OR_EQUAL_BCAST,
                                       timer.ElapsedUS());
#endif
    }
  }

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
    auto out_cpu = lt(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 4, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 4, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparision(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 4,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparisionBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 4,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_BCAST, timer.ElapsedUS());
#endif
    }
  }

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
    auto out_cpu = ne(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), 1, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::NOT_EQUAL_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), 1, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::NOT_EQUAL_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparision(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 1,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::NOT_EQUAL, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparisionBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), 1,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::NOT_EQUAL_BCAST, timer.ElapsedUS());
#endif
    }
  }

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

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnShiftLeftC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toChar(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::SHIFT_LEFT_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnShiftLeftC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toChar(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::SHIFT_LEFT_C, timer.ElapsedUS());
#endif

  } else if (self.dim() == other.dim()) {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnShiftLeft(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::SHIFT_LEFT, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnShiftLeftBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::SHIFT_LEFT_BCAST,
                                       timer.ElapsedUS());
#endif
    }
  } else {
    TORCH_CHECK(false, "unsupported dims");
  }

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

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnShiftRightArithmeticC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toInt(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::SHIFT_RIGHT_ARITHMETIC_C,
                                     timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnShiftRightArithmeticC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toInt(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::SHIFT_RIGHT_ARITHMETIC_C,
                                     timer.ElapsedUS());
#endif

  } else if (self.dim() == other.dim()) {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnShiftRightArithmetic(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::SHIFT_RIGHT_ARITHMETIC,
                                       timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      // bm_status_t status =
      // sgdnnShiftRightArithmeticBcast(tpu::TPUGetDeviceHandle(),
      //                                      tpu:: TPUGenerateSgdnnTensor (
      //                                      self ), tpu::
      //                                      TPUGenerateSgdnnTensor ( other ),
      //                                      tpu:: TPUGenerateSgdnnTensor ( out
      //                                      ) );
      // TORCH_CHECK ( status == BM_SUCCESS );
      TORCH_CHECK(false, "unsupported dims");
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::SHIFT_RIGHT_ARITHMETIC_BCAST,
                                       timer.ElapsedUS());
#endif
      TORCH_CHECK(false, "unsupported dims");
    }
  } else {
    TORCH_CHECK(false, "unsupported dims");
  }

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
    auto out_cpu = minimum(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnMinimumC(
        tpu::TPUGetDeviceHandle(),
        tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::MINIMUM, timer.ElapsedUS());
#endif
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnMinimumC(
        tpu::TPUGetDeviceHandle(),
        tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::MINIMUM, timer.ElapsedUS());
#endif
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnMinimum(tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                       tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                       tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MINIMUM, timer.ElapsedUS());
#endif
    } else {
      // The shapes of self and other are not the same, need to broadcast
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnMinimumBcast(tpu::TPUGetDeviceHandle(),
                            tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                            tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                            tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MINIMUM, timer.ElapsedUS());
#endif
    }
  }
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
    auto out_cpu = maximum(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnMaximumC(
        tpu::TPUGetDeviceHandle(),
        tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::MAXIMUM, timer.ElapsedUS());
#endif
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnMaximumC(
        tpu::TPUGetDeviceHandle(),
        tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::MAXIMUM, timer.ElapsedUS());
#endif
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnMaximum(tpu::TPUGetDeviceHandle(),
                       tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                       tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                       tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MAXIMUM, timer.ElapsedUS());
#endif
    } else {
      // The shapes of self and other are not the same, need to broadcast
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnMaximumBcast(tpu::TPUGetDeviceHandle(),
                            tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                            tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                            tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MAXIMUM, timer.ElapsedUS());
#endif
    }
  }
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
    auto out_cpu = atan2(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnAtan2C(tpu::TPUGetDeviceHandle(), *scalar.data_ptr<float>(),
                    tpu::TPUGenerateSgdnnTensor(other.to(torch::kFloat)),
                    tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::ATAN2, timer.ElapsedUS());
#endif
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnAtan2_C(
        tpu::TPUGetDeviceHandle(),
        tpu::TPUGenerateSgdnnTensor(self.to(torch::kFloat)),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::ATAN2, timer.ElapsedUS());
#endif
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnAtan2(tpu::TPUGetDeviceHandle(),
                     tpu::TPUGenerateSgdnnTensor(self.to(torch::kFloat)),
                     tpu::TPUGenerateSgdnnTensor(other.to(torch::kFloat)),
                     tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::ATAN2, timer.ElapsedUS());
#endif
    } else {
      // The shapes of self and other are not the same, need to broadcast
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnAtan2Bcast(tpu::TPUGetDeviceHandle(),
                          tpu::TPUGenerateSgdnnTensor(self.to(torch::kFloat)),
                          tpu::TPUGenerateSgdnnTensor(other.to(torch::kFloat)),
                          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::ATAN2, timer.ElapsedUS());
#endif
    }
  }
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("atan2.out", atan2_out_tpu); }

Tensor &pow_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(other);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

  auto self_cpu = pow ( self.cpu(),other.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
  tpu::TPUCopyHostToDevice ( other.data_ptr(),other.contiguous().data_ptr(), other.nbytes() );
#else
  if (self.dim() == 0) {
    auto self_cpu = pow(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(),
                             self.nbytes());
    tpu::TPUCopyHostToDevice(other.data_ptr(), other.contiguous().data_ptr(),
                             other.nbytes());
  } else if (IS_TPU_TENSOR(self)) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnPow(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::POW_FORWARD, timer.ElapsedUS());
#endif
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  return out;
}

Tensor &pow_tpu(const Tensor &self, const Tensor &other) {
  auto out = empty(self.sizes(), self.options());
  return pow_out_tpu(self, other, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  // m.impl("pow.out", pow_out_tpu);
  m.impl("pow.Tensor_Tensor_out", pow_out_tpu);
}

Tensor &pow_c_out_tpu(const Tensor &self, const Scalar &exponent, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  LOG( WARNING ) << "pow_out_tpu use cpu impl";
  auto out_cpu = pow( self.cpu(), exponent );
  out = out_cpu.to(out.device());
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
  bm_status_t status =
      sgdnnPowC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                exponent.toDouble(), tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::POWC, timer.ElapsedUS());
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("pow.Tensor_Scalar_out", pow_c_out_tpu);
}

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
    auto out_cpu = fmax(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnFmaxC(tpu::TPUGetDeviceHandle(),
                   tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                   *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::FMAX, timer.ElapsedUS());
#endif
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnFmaxC(tpu::TPUGetDeviceHandle(),
                   tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                   *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::FMAX, timer.ElapsedUS());
#endif
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnFmax(tpu::TPUGetDeviceHandle(),
                    tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                    tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                    tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::FMAX, timer.ElapsedUS());
#endif
    } else {
      // The shapes of self and other are not the same, need to broadcast
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnFmaxBcast(tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::FMAX, timer.ElapsedUS());
#endif
    }
  }
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
    auto out_cpu = fmin(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnFminC(tpu::TPUGetDeviceHandle(),
                   tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                   *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::FMIN, timer.ElapsedUS());
#endif
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnFminC(tpu::TPUGetDeviceHandle(),
                   tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                   *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::FMIN, timer.ElapsedUS());
#endif
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnFmin(tpu::TPUGetDeviceHandle(),
                    tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                    tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                    tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::FMIN, timer.ElapsedUS());
#endif
    } else {
      // The shapes of self and other are not the same, need to broadcast
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnFminBcast(tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::FMIN, timer.ElapsedUS());
#endif
    }
  }
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

#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif

  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = hypot(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {
    bm_status_t status = sgdnnHypotC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
        self.item().toFloat(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
  } else if (other.dim() == 0) {
    bm_status_t status = sgdnnHypotC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.item().toFloat(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
  } else {
    if (tpu::TPUIsSameShape(self, other)) {
      bm_status_t status = sgdnnHypot(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
    } else {
      bm_status_t status = sgdnnHypotBcast(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
    }
  }

#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::HYPOT, timer.ElapsedUS());
#endif

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
    auto out_cpu = nextafter(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (self.dim() == 0) {
    // self is scalar
    Tensor scalar = self.to(torch::kFloat32).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnNextafterC(tpu::TPUGetDeviceHandle(), *scalar.data_ptr<float>(),
                        tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::NEXTAFTER, timer.ElapsedUS());
#endif
  } else if (other.dim() == 0) {
    // other is scalar
    Tensor scalar = other.to(torch::kFloat32).cpu();
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnNextafter_C(
        tpu::TPUGetDeviceHandle(),
        tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
        *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::NEXTAFTER, timer.ElapsedUS());
#endif
  } else {
    // self and other have the same shape
    if (self.sizes() == other.sizes()) {
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status =
          sgdnnNextafter(tpu::TPUGetDeviceHandle(),
                         tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
                         tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::NEXTAFTER, timer.ElapsedUS());
#endif
    } else {
      // The shapes of self and other are not the same, need to broadcast
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnNextafterBcast(
          tpu::TPUGetDeviceHandle(),
          tpu::TPUGenerateSgdnnTensor(self.to(out.dtype())),
          tpu::TPUGenerateSgdnnTensor(other.to(out.dtype())),
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::NEXTAFTER, timer.ElapsedUS());
#endif
    }
  }
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
    auto out_cpu = le(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 5, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_OR_EQUAL_C,
                                     timer.ElapsedUS());
#endif
  }
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
    auto out_cpu = lt(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else {
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 4, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::LESS_THAN_C, timer.ElapsedUS());
#endif
  }
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
    auto out_cpu = ge(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else {
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 3, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::GREATER_OR_EQUAL_C,
                                     timer.ElapsedUS());
#endif
  }
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("ge.Scalar_out", greater_or_equal_scalar_out_tpu);
}

Tensor &greater_scalar_out_tpu(const Tensor &self, const Scalar &other,
                               Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);

  if (self.dim() == 0) {
    auto out_cpu = gt(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 2, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::GREATER_C, timer.ElapsedUS());
#endif
  }
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
    auto out_cpu = ne(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else {
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 1, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::NOT_EQUAL_C, timer.ElapsedUS());
#endif
  }
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
    auto out_cpu = eq(self.cpu(), other);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than,
    // 5 less than or equal pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        other.toFloat(), 0, 1, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::EQUAL_C, timer.ElapsedUS());
#endif
  }
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("eq.Scalar_out", equal_scalar_out_tpu);
}

} // namespace at
