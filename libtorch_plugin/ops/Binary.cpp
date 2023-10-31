#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <cmath>
#include <float.h>
#include "common/config.h"

namespace at
{
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
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = add ( self.cpu(), other.cpu(), alpha );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = add(self.cpu(), other.cpu(), alpha);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (IS_TPU_TENSOR(self) && IS_TPU_TENSOR(other)) {
    if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START
      bm_status_t status =
          sgdnnAdd(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                   tpu::TPUGenerateSgdnnTensor(other), alpha.toDouble(),
                   tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
      TIMING_END(tpu::ADD)
    } else {
      auto self_t = tpu::TPUGenerateSgdnnTensor(self);
      auto other_t = tpu::TPUGenerateSgdnnTensor(other);
      auto out_t = tpu::TPUGenerateSgdnnTensor(out);
      if (self_t.dim != other_t.dim) {

        auto &src_t = other_t.dim != 1 ? other_t : self_t;
        auto &target_t = other_t.dim == 1 ? other_t : self_t;

        TORCH_CHECK(target_t.dim == 1, "only support one of tensor dim = 1")
        target_t.shape[src_t.dim - 1] = target_t.shape[0];
        target_t.stride[src_t.dim - 1] = 1;
        target_t.dim = src_t.dim;
        for (int i = 0; i < src_t.dim - 1; i++) {
          target_t.shape[i] = 1;
          target_t.stride[i] = target_t.shape[0];
        }
      }
      TIMING_START
      bm_status_t status = sgdnnAddBcast(tpu::TPUGetDeviceHandle(), self_t,
                                         other_t, alpha.toDouble(), out_t);
      TORCH_CHECK(status == BM_SUCCESS);
      TIMING_END(tpu::BCAST_ADD)
    }
  } else if ((IS_TPU_TENSOR(self) && IS_CPU_TENSOR(other)) ||
             (IS_CPU_TENSOR(self) && IS_TPU_TENSOR(other))) {
    if (IS_CPU_TENSOR(other)) {
      if (self.dtype() == caffe2::TypeMeta::Make<long>() ||
          self.dtype() == caffe2::TypeMeta::Make<int>()) {
        LOG(WARNING) << "add self's dtype is long or int, use cpu";
        auto out_cpu =
            add(self.to(out.dtype()).cpu(), other.to(out.dtype()).cpu(), alpha);
        out = out_cpu.to(out.device());
        return out;
      }
      TORCH_CHECK(other.dim() == 0, "OTHER must be a scalar");
      Tensor scalar;
      if (other.dtype() == caffe2::TypeMeta::Make<double>()) {
        scalar = other.to(torch::kFloat);
      } else {
        scalar = other;
      }
      TIMING_START
      if (scalar.dtype() == torch::kLong) {
        bm_status_t status = sgdnnAddC(
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
            alpha.toDouble() * (*scalar.data_ptr<long>()),
            tpu::TPUGenerateSgdnnTensor(out));
        TORCH_CHECK(status == BM_SUCCESS);
      } else {
        bm_status_t status = sgdnnAddC(
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
            alpha.toDouble() * (*scalar.data_ptr<float>()),
            tpu::TPUGenerateSgdnnTensor(out));
        TORCH_CHECK(status == BM_SUCCESS);
      }
      TIMING_END(tpu::ADD_C)
    } else {
      TORCH_CHECK(alpha.toDouble() == 1.0);
      TORCH_CHECK(self.dim() == 0, "SELF must be a scalar");
      Tensor scalar;
      if (self.dtype() == caffe2::TypeMeta::Make<double>()) {
        scalar = self.to(torch::kFloat);
      } else {
        scalar = self;
      }
      TIMING_START
      if (scalar.dtype() == torch::kLong) {
        bm_status_t status = sgdnnAddC(
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
            *scalar.data_ptr<long>(), tpu::TPUGenerateSgdnnTensor(out));
        TORCH_CHECK(status == BM_SUCCESS);
      } else {
        bm_status_t status = sgdnnAddC(
            tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
            *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
        TORCH_CHECK(status == BM_SUCCESS);
      }
      TIMING_END(tpu::ADD_C)
    }
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("add.out", add_out_tpu); }

Tensor &sub_out_tpu(const Tensor &self, const Tensor &other,
                    const Scalar &alpha, Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = sub ( self.cpu(), other.cpu(), alpha );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = sub(self.cpu(), other.cpu(), alpha);
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (IS_TPU_TENSOR(self) && IS_TPU_TENSOR(other)) {
      if (tpu::TPUIsSameShape(self, other)) {
      TIMING_START
      bm_status_t status =
          sgdnnAdd(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                   tpu::TPUGenerateSgdnnTensor(other), -1.0f *alpha.toDouble(),
                   tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
      TIMING_END(tpu::ADD)
    } else {
      auto self_t = tpu::TPUGenerateSgdnnTensor(self);
      auto other_t = tpu::TPUGenerateSgdnnTensor(other);
      auto out_t = tpu::TPUGenerateSgdnnTensor(out);
      if (self_t.dim != other_t.dim) {

        auto &src_t = other_t.dim != 1 ? other_t : self_t;
        auto &target_t = other_t.dim == 1 ? other_t : self_t;

        TORCH_CHECK(target_t.dim == 1, "only support one of tensor dim = 1")
        target_t.shape[src_t.dim - 1] = target_t.shape[0];
        target_t.stride[src_t.dim - 1] = 1;
        target_t.dim = src_t.dim;
        for (int i = 0; i < src_t.dim - 1; i++) {
          target_t.shape[i] = 1;
          target_t.stride[i] = target_t.shape[0];
        }
      }
      TIMING_START
      bm_status_t status = sgdnnAddBcast(tpu::TPUGetDeviceHandle(), self_t,
                                         other_t,  -1.0f * alpha.toDouble(), out_t);
      TORCH_CHECK(status == BM_SUCCESS);
      TIMING_END(tpu::BCAST_ADD)
    }
  } else if ((IS_TPU_TENSOR(self) && IS_CPU_TENSOR(other)) ||
             (IS_CPU_TENSOR(self) && IS_TPU_TENSOR(other))) {
    if (IS_CPU_TENSOR(other)) {
      TORCH_CHECK(other.dim() == 0, "OTHER must be a scalar");
      Tensor scalar;
      if (other.dtype() == caffe2::TypeMeta::Make<double>()) {
        scalar = other.to(torch::kFloat);
      } else {
        scalar = other;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      if (scalar.dtype() == torch::kLong) {
            bm_status_t status = sgdnnAddC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           - alpha.toDouble() * ( *scalar.data_ptr<long>() ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
            TORCH_CHECK ( status == BM_SUCCESS );
      } else {
            bm_status_t status = sgdnnAddC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           - alpha.toDouble() * ( *scalar.data_ptr<float>() ),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
            TORCH_CHECK ( status == BM_SUCCESS );
      }
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::ADD_C, timer.ElapsedUS());
#endif
    } else {
      TORCH_CHECK(alpha.toDouble() == 1.0);
      TORCH_CHECK(self.dim() == 0, "SELF must be a scalar");
      Tensor scalar;
      if (self.dtype() == caffe2::TypeMeta::Make<double>()) {
        scalar = self.to(torch::kFloat);
      } else {
        scalar = self;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnCSub(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
          alpha.toDouble() * (*scalar.data_ptr<float>()),
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::C_SUB, timer.ElapsedUS());
#endif
    }
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  return out;
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
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if (other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = mul ( self.cpu(), other.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if (self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = mul(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                             out.nbytes());
  } else if (IS_TPU_TENSOR(self) && IS_TPU_TENSOR(other)) {
    if (other.dim() == 0) {
      Tensor scalar = other.cpu().to(torch::kFloat);
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MUL_C, timer.ElapsedUS());
#endif
    }
    else
    {
      auto self_t  = tpu::TPUGenerateSgdnnTensor ( self );
      auto other_t  = tpu::TPUGenerateSgdnnTensor ( other );
      int maxdim = self_t.dim > other_t.dim ? self_t.dim : other_t.dim;
      if ( self_t.dim != maxdim )
      {
        int stride = 1;
        for ( int i = 0; i < maxdim; i++ )
        {
          if (i < self_t.dim ) { self_t.shape[ maxdim-i-1 ] = self_t.shape[ self_t.dim-i-1 ];
                                self_t.stride[ maxdim-i-1 ] = stride; }
          else { self_t.shape[ maxdim-i-1 ] = 1;
                self_t.stride[ maxdim-i-1 ] =  stride;}
          stride *= self_t.shape[ maxdim-i-1 ];
        }
        self_t.dim = maxdim;
      }
      if ( other_t.dim != maxdim )
      {
        int stride = 1;
        for ( int i = 0; i < maxdim; i++ )
        {
          if (i < other_t.dim ) { other_t.shape[ maxdim-i-1 ] = other_t.shape[ other_t.dim-i-1 ];
                                other_t.stride[ maxdim-i-1 ] = stride; }
          else { other_t.shape[ maxdim-i-1 ] = 1;
                other_t.stride[ maxdim-i-1 ] =  stride;}
          stride *= other_t.shape[ maxdim-i-1 ];
        }
        other_t.dim = maxdim;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMul(
          tpu::TPUGetDeviceHandle(),
          self_t,
          other_t,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MUL, timer.ElapsedUS());
#endif
    }
  } else if ((IS_TPU_TENSOR(self) && IS_CPU_TENSOR(other)) ||
             (IS_CPU_TENSOR(self) && IS_TPU_TENSOR(other))) {
    if (IS_CPU_TENSOR(other)) {
      TORCH_CHECK(other.dim() == 0, "OTHER must be a scalar");
      Tensor scalar = other.to(torch::kFloat);
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MUL_C, timer.ElapsedUS());
#endif
    } else {
      TORCH_CHECK(self.dim() == 0, "SELF must be a scalar");
      Tensor scalar;
      if (self.dtype() == caffe2::TypeMeta::Make<double>()) {
        scalar = self.to(torch::kFloat);
      } else {
        scalar = self;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
          *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MUL_C, timer.ElapsedUS());
#endif
    }
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("mul.out", mul_out_tpu); }

Tensor & div_out_tpu ( const Tensor & self, const Tensor & other, Tensor & out )
{
  if ( !self.is_contiguous() || !other.is_contiguous() || !out.is_contiguous() )
  {
      LOG( WARNING ) << "div out use strided copy because of, "
                        << " self_is_contiguous : " << self.is_contiguous()
                        << " other_is_contiguos : " << other.is_contiguous()
                        << " out_is_congiguous : " << out.is_contiguous()
                        << " [TODO] no use strided copy";
      if (out.is_contiguous()){ out = div(self.contiguous(), other.contiguous()); }
      else {
            auto out_ = div(self.contiguous(), other.contiguous());
            sgdnnStridedCopy(
                  tpu::TPUGetDeviceHandle(),
                  tpu::TPUGenerateSgdnnTensor ( out_ ),
                  tpu::TPUGenerateSgdnnTensor ( out ));
      }
      return out;
  }
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  auto out_cpu = div ( self.cpu(), other.cpu() );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if ( self.dim() == 0 && other.dim() == 0 )
  {
    auto out_cpu = div ( self.cpu(), other.cpu() );
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
  }
  else if ( IS_TPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) )
  {
    if (other.dim() == 0) {
      /* RECIPROCAL */
      Tensor scalar = (1.0 / other.cpu().to(torch::kFloat)).to(torch::kFloat);
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
          *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MUL_C, timer.ElapsedUS());
#endif
    }
    else
    {
      auto self_t  = tpu::TPUGenerateSgdnnTensor ( self );
      auto other_t  = tpu::TPUGenerateSgdnnTensor ( other );
      int maxdim = self_t.dim > other_t.dim ? self_t.dim : other_t.dim;
      if ( self_t.dim != maxdim )
      {
        int stride = 1;
        for ( int i = 0; i < maxdim; i++ )
        {
          if (i < self_t.dim ) { self_t.shape[ maxdim-i-1 ] = self_t.shape[ self_t.dim-i-1 ];
                                self_t.stride[ maxdim-i-1 ] = stride; }
          else { self_t.shape[ maxdim-i-1 ] = 1;
                self_t.stride[ maxdim-i-1 ] =  stride;}
          stride *= self_t.shape[ maxdim-i-1 ];
        }
        self_t.dim = maxdim;
      }
      if ( other_t.dim != maxdim )
      {
        int stride = 1;
        for ( int i = 0; i < maxdim; i++ )
        {
          if (i < other_t.dim ) { other_t.shape[ maxdim-i-1 ] = other_t.shape[ other_t.dim-i-1 ];
                                other_t.stride[ maxdim-i-1 ] = stride; }
          else { other_t.shape[ maxdim-i-1 ] = 1;
                other_t.stride[ maxdim-i-1 ] =  stride;}
          stride *= other_t.shape[ maxdim-i-1 ];
        }
        other_t.dim = maxdim;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnDiv(
          tpu::TPUGetDeviceHandle(),
          self_t,
          other_t,
          tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MUL, timer.ElapsedUS());
#endif
    }
  }
  else if ( ( IS_TPU_TENSOR ( self ) && IS_CPU_TENSOR ( other ) ) ||
            ( IS_CPU_TENSOR ( self ) && IS_TPU_TENSOR ( other ) ) )
  {
    if ( IS_CPU_TENSOR ( other ) )
    {
      TORCH_CHECK ( other.dim() == 0, "OTHER must be a scalar" );
      /* RECIPROCAL */
      Tensor scalar = (1.0 / other.to(torch::kFloat)).to(torch::kFloat);
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnMulC (
                           tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           *scalar.data_ptr<float>(),
                           tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::MUL_C, timer.ElapsedUS());
#endif
    }
    else
    {
      TORCH_CHECK ( self.dim() == 0, "SELF must be a scalar" );
      Tensor scalar;
      if ( self.dtype() == caffe2::TypeMeta::Make<double>() )
      {
        scalar = self.to ( torch::kFloat );
      }
      else
      {
        scalar = self;
      }
#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnCDiv(
          tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
          *scalar.data_ptr<float>(), tpu::TPUGenerateSgdnnTensor(out));
      TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::C_DIV, timer.ElapsedUS());
#endif
    }
  } else {
    TORCH_CHECK(false, "At least one input is required in TPU device");
  }
#endif
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
    bm_status_t status = sgdnnElementBitwiseC(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( other ),
                                              self.item().toInt(),
                                              0,      // 0 for xor, 1 for and, 2 for or
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_XOR_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( self ),
                                              other.item().toInt(),
                                              0,      // 0 for xor, 1 for and, 2 for or
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_XOR_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwise(tpu::TPUGetDeviceHandle(),
                                               tpu:: TPUGenerateSgdnnTensor ( self ),
                                               tpu:: TPUGenerateSgdnnTensor ( other ),
                                               0,      // 0 for xor, 1 for and, 2 for or
                                               tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_XOR, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwiseBcast(tpu::TPUGetDeviceHandle(),
                                                    tpu:: TPUGenerateSgdnnTensor ( self ),
                                                    tpu:: TPUGenerateSgdnnTensor ( other ),
                                                    0,      // 0 for xor, 1 for and, 2 for or
                                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
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
    bm_status_t status = sgdnnElementBitwiseC(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( other ),
                                              self.item().toInt(),
                                              1,      // 0 for xor, 1 for and, 2 for or
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_AND_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( self ),
                                              other.item().toInt(),
                                              1,      // 0 for xor, 1 for and, 2 for or
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_AND_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwise(tpu::TPUGetDeviceHandle(),
                                               tpu:: TPUGenerateSgdnnTensor ( self ),
                                               tpu:: TPUGenerateSgdnnTensor ( other ),
                                               1,      // 0 for xor, 1 for and, 2 for or
                                               tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_AND, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwiseBcast(tpu::TPUGetDeviceHandle(),
                                                    tpu:: TPUGenerateSgdnnTensor ( self ),
                                                    tpu:: TPUGenerateSgdnnTensor ( other ),
                                                    1,      // 0 for xor, 1 for and, 2 for or
                                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
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
    bm_status_t status = sgdnnElementBitwiseC(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( other ),
                                              self.item().toInt(),
                                              2,      // 0 for xor, 1 for and, 2 for or
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_OR_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnElementBitwiseC(tpu::TPUGetDeviceHandle(),
                                              tpu:: TPUGenerateSgdnnTensor ( self ),
                                              other.item().toInt(),
                                              2,      // 0 for xor, 1 for and, 2 for or
                                              tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::BITWISE_OR_C, timer.ElapsedUS());
#endif

  } else {
    if (tpu::TPUIsSameShape(self, other)) {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwise(tpu::TPUGetDeviceHandle(),
                                               tpu:: TPUGenerateSgdnnTensor ( self ),
                                               tpu:: TPUGenerateSgdnnTensor ( other ),
                                               2,      // 0 for xor, 1 for and, 2 for or
                                               tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime(tpu::BITWISE_OR, timer.ElapsedUS());
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnElementBitwiseBcast(tpu::TPUGetDeviceHandle(),
                                                    tpu:: TPUGenerateSgdnnTensor ( self ),
                                                    tpu:: TPUGenerateSgdnnTensor ( other ),
                                                    2,      // 0 for xor, 1 for and, 2 for or
                                                    tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
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
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
    // pos : 0 for self is scalar, 1 for other is scalar
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
          tpu::TPUGenerateSgdnnTensor(other), 0, tpu::TPUGenerateSgdnnTensor(out));
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
          tpu::TPUGenerateSgdnnTensor(other), 0, tpu::TPUGenerateSgdnnTensor(out));
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
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
    // pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status =
        sgdnnComparisionC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
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
    bm_status_t status =
        sgdnnComparisionC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
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
          tpu::TPUGenerateSgdnnTensor(other), 3, tpu::TPUGenerateSgdnnTensor(out));
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
          tpu::TPUGenerateSgdnnTensor(other), 3, tpu::TPUGenerateSgdnnTensor(out));
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
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
    // pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status =
        sgdnnComparisionC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
                 self.item().toFloat(), 2, 0, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime(tpu::GREATER_C, timer.ElapsedUS());
#endif

  } else if (other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status =
        sgdnnComparisionC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
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
          tpu::TPUGenerateSgdnnTensor(other), 2, tpu::TPUGenerateSgdnnTensor(out));
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
          tpu::TPUGenerateSgdnnTensor(other), 2, tpu::TPUGenerateSgdnnTensor(out));
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
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
    // pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status =
        sgdnnComparisionC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
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
    bm_status_t status =
        sgdnnComparisionC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
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
          tpu::TPUGenerateSgdnnTensor(other), 5, tpu::TPUGenerateSgdnnTensor(out));
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
          tpu::TPUGenerateSgdnnTensor(other), 5, tpu::TPUGenerateSgdnnTensor(out));
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

Tensor & less_than_out_tpu( const Tensor &self, const Tensor &other, Tensor &out) {
  if ( self.dim() > 0 )  { CHECK_TENSOR_IN_DEVICE ( self ); }
  if ( other.dim() > 0 ) { CHECK_TENSOR_IN_DEVICE ( other ); }
  CHECK_TENSOR_IN_DEVICE(out);

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = lt(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
    // pos : 0 for self is scalar, 1 for other is scalar
    bm_status_t status = sgdnnComparisionC(tpu::TPUGetDeviceHandle(),
                                           tpu:: TPUGenerateSgdnnTensor ( other ),
                                           self.item().toFloat(),
                                           4,
                                           0,
                                           tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_C, timer.ElapsedUS() );
#endif

  }
  else if(other.dim() == 0) {

#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    bm_status_t status = sgdnnComparisionC(tpu::TPUGetDeviceHandle(),
                                           tpu:: TPUGenerateSgdnnTensor ( self ),
                                           other.item().toFloat(),
                                           4,
                                           1,
                                           tpu:: TPUGenerateSgdnnTensor ( out ) );
    TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_C, timer.ElapsedUS() );
#endif

  }
  else {
    if(tpu::TPUIsSameShape(self, other)){

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparision(tpu::TPUGetDeviceHandle(),
                                            tpu:: TPUGenerateSgdnnTensor ( self ),
                                            tpu:: TPUGenerateSgdnnTensor ( other ),
                                            4,
                                            tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN, timer.ElapsedUS() );
#endif

    } else {

#ifdef TPU_OP_TIMING
      auto timer = tpu::Timer().Start();
#endif
      bm_status_t status = sgdnnComparisionBcast(tpu::TPUGetDeviceHandle(),
                                                 tpu:: TPUGenerateSgdnnTensor ( self ),
                                                 tpu:: TPUGenerateSgdnnTensor ( other ),
                                                 4,
                                                 tpu:: TPUGenerateSgdnnTensor ( out ) );
      TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
      tpu::OpTimer::Instance().AddTime ( tpu::LESS_THAN_BCAST, timer.ElapsedUS() );
#endif

    }
  }

  return out;
}
TORCH_LIBRARY_IMPL( aten, TPU, m) {
  m.impl ( "lt.Tensor_out", less_than_out_tpu );
}

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
    // mode : 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
    // pos : 0 for self is scalar, 1 for other is scalar
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
          tpu::TPUGenerateSgdnnTensor(other), 1, tpu::TPUGenerateSgdnnTensor(out));
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
          tpu::TPUGenerateSgdnnTensor(other), 1, tpu::TPUGenerateSgdnnTensor(out));
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

Tensor &pow_out_tpu(const Tensor &self, const Tensor &other, Tensor &out)
    {
        if (self.dim() > 0)
        {
            CHECK_TENSOR_IN_DEVICE(self);
        }
        CHECK_TENSOR_IN_DEVICE(other);
        CHECK_TENSOR_IN_DEVICE(out);
#if 0

  auto self_cpu = pow ( self.cpu(),other.cpu());
  tpu::TPUCopyHostToDevice ( self.data_ptr(),self.contiguous().data_ptr(), self.nbytes() );
  tpu::TPUCopyHostToDevice ( other.data_ptr(),other.contiguous().data_ptr(), other.nbytes() );
#else
        if (self.dim() == 0)
        {
            auto self_cpu = pow(self.cpu(), other.cpu());
            tpu::TPUCopyHostToDevice(self.data_ptr(), self.contiguous().data_ptr(), self.nbytes());
            tpu::TPUCopyHostToDevice (other.data_ptr(),other.contiguous().data_ptr(), other.nbytes() );
        }
        else if (IS_TPU_TENSOR(self))
        {

#ifdef TPU_OP_TIMING
            auto timer = tpu::Timer().Start();
#endif
            bm_status_t status = sgdnnPow(
                tpu::TPUGetDeviceHandle(),
                tpu::TPUGenerateSgdnnTensor(self),
                tpu::TPUGenerateSgdnnTensor(other),
                tpu::TPUGenerateSgdnnTensor(out));
            TORCH_CHECK(status == BM_SUCCESS);
#ifdef TPU_OP_TIMING
            tpu::OpTimer::Instance().AddTime(tpu::POW_FORWARD, timer.ElapsedUS());
#endif
        }
        else
        {
            TORCH_CHECK(false, "At least one input is required in TPU device");
        }
#endif
        return out;
    }

Tensor &pow_tpu(const Tensor &self, const Tensor &other)
{
      auto out = empty(self.sizes(), self.options());
      return pow_out_tpu(self, other, out);
}

TORCH_LIBRARY_IMPL(aten, TPU, m)
{
      // m.impl("pow.out", pow_out_tpu);
      m.impl("pow.Tensor_Tensor_out", pow_out_tpu);
}

Tensor & pow_c_out_tpu( const Tensor & self, const Scalar & exponent, Tensor & out){
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( out );
#if 0
  LOG( WARNING ) << "pow_out_tpu use cpu impl";
  auto out_cpu = pow( self.cpu(), exponent );
  out = out_cpu.to(out.device());
#else
#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif
bm_status_t status = sgdnnPowC (
                            tpu::TPUGetDeviceHandle(),
                           tpu:: TPUGenerateSgdnnTensor ( self ),
                           exponent.toDouble(),
                           tpu:: TPUGenerateSgdnnTensor ( out )
);
  TORCH_CHECK ( status == BM_SUCCESS );
#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime ( tpu::POWC, timer.ElapsedUS() );
#endif
#endif
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "pow.Tensor_Scalar_out", pow_c_out_tpu );
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

Tensor & hypot_out_tpu(const Tensor &self, const Tensor &other, Tensor &out) {
  if(self.dim() >> 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  if(other.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(other);
  }
  CHECK_TENSOR_IN_DEVICE(out);

#ifdef TPU_OP_TIMING
  auto timer = tpu::Timer().Start();
#endif

  if(self.dim() == 0 && other.dim() == 0) {
    auto out_cpu = hypot(self.cpu(), other.cpu());
    tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes());
  }
  else if(self.dim() == 0) {
    bm_status_t status = sgdnnHypotC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(other),
                                     self.item().toFloat(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK ( status == BM_SUCCESS );
  }
  else if(other.dim() == 0) {
    bm_status_t status = sgdnnHypotC(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                                     other.item().toFloat(), tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK ( status == BM_SUCCESS );
  }
  else {
    if(tpu::TPUIsSameShape(self, other)) {
      bm_status_t status = sgdnnHypot(
                              tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                              tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out ));
      TORCH_CHECK ( status == BM_SUCCESS );
    }
    else {
      bm_status_t status = sgdnnHypotBcast(
                              tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                              tpu::TPUGenerateSgdnnTensor(other), tpu::TPUGenerateSgdnnTensor(out ));
      TORCH_CHECK ( status == BM_SUCCESS );
    }
  }

#ifdef TPU_OP_TIMING
  tpu::OpTimer::Instance().AddTime(tpu::HYPOT, timer.ElapsedUS());
#endif

  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("hypot.out", hypot_out_tpu);
}

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

} // namespace at
