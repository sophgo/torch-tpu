#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/native/ReduceOpsUtils.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

namespace at {
Tensor &mean_out_tpu(const Tensor &self, OptionalIntArrayRef dim_opt,
                     bool keepdim, c10::optional<ScalarType> dtype_opt,
                     Tensor &out) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = mean ( self.cpu(), dim_opt, keepdim, dtype_opt );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else

  auto reduce_dim = dim_opt.value_or(IntArrayRef{});
  std::vector<int> reduction_dim_vec;
  if (reduce_dim.size() > 0) {
    for (auto it : reduce_dim) {
      reduction_dim_vec.push_back(it < 0 ? it + self.dim() : it);
    }
    std::sort(reduction_dim_vec.begin(), reduction_dim_vec.end());
  } else {
    for (auto i = 0; i < self.dim(); ++i) {
      reduction_dim_vec.push_back(i);
    }
  }
  for (size_t i = 0; i < reduction_dim_vec.size() - 1; ++i) {
    TORCH_CHECK(reduction_dim_vec[i] + 1 == reduction_dim_vec[i + 1],
                "Reduction only supports contiguous reduction dimension now");
  }
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status =
      tpudnnReduceAsync(stream, tpu::TPUGenerateTpudnnTensor(stream, self),
                        reduction_dim_vec[0], reduction_dim_vec.back() + 1,
                        keepdim, 0, tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("mean.out", mean_out_tpu); }

Tensor &sum_IntList_out_tpu(const Tensor &self, OptionalIntArrayRef dim_opt,
                            bool keepdim, c10::optional<ScalarType> dtype_opt,
                            Tensor &out) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  CHECK_TENSOR_IN_DEVICE(out);
  auto self_ = self.contiguous();
#if 0
  auto out_cpu = sum ( self.cpu(), dim_opt, keepdim, dtype_opt );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  auto reduce_dim = dim_opt.value_or(IntArrayRef{});
  std::vector<int> reduction_dim_vec;
  if (reduce_dim.size() > 0) {
    for (auto it : reduce_dim) {
      reduction_dim_vec.push_back(it < 0 ? it + self_.dim() : it);
    }
    std::sort(reduction_dim_vec.begin(), reduction_dim_vec.end());
  } else {
    for (auto i = 0; i < self_.dim(); ++i) {
      reduction_dim_vec.push_back(i);
    }
  }
  for (size_t i = 0; i < reduction_dim_vec.size() - 1; ++i) {
    TORCH_CHECK(reduction_dim_vec[i] + 1 == reduction_dim_vec[i + 1],
                "Reduction only supports contiguous reduction dimension now");
  }
  const int start_dim = reduction_dim_vec.front();
  const int end_dim = reduction_dim_vec.back();
  if(start_dim == 0 || end_dim + 1 == self_.dim()){
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnReduceAsync(
        stream, tpu::TPUGenerateTpudnnTensor(stream, self_),
        start_dim, end_dim + 1, keepdim, 1,
        tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
  else{//workaround: matmul to sum
    int64_t M = 1, N = 1, Bsz = 1;
    for(int i = 0; i < start_dim; i++){
      Bsz *= self_.size(i);
    }
    for(int i = start_dim; i <= end_dim; i++){
      M *= self_.size(i);
    }
    for(int i = end_dim + 1; i < self_.dim(); i++){
      N *= self_.size(i);
    }
    Tensor right = self_.reshape({Bsz, M, N});

    TensorOptions opts = self_.options();
    if(dtype_opt.has_value()){
      opts = opts.dtype(dtype_opt.value());
    }
    Tensor left = torch::ones({Bsz, 1, M}, opts);

    Tensor out_mm;
    if(out.numel() != Bsz * N){
      out_mm = at::empty({Bsz, 1, N}, opts);
    } else{
      std::vector<int64_t> new_strides = {N, N, 1};
      out.unsafeGetTensorImpl()->set_sizes_and_strides({Bsz, 1, N}, new_strides);
      out_mm = out;
    }

    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnMatmulAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, left),
    tpu::TPUGenerateTpudnnTensor(stream, right),
    tpudnnUndefinedTensor(),
    tpu::TPUGenerateTpudnnTensor(stream, out_mm));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);

    std::vector<int64_t> size_out;
    for(int d = 0; d < self_.dim(); d++){
      if(d < start_dim || d > end_dim){
        size_out.push_back(self_.size(d));
      } else if(keepdim){
        size_out.push_back(1);
      }
    }
    if(!keepdim && size_out.empty()){
      size_out = {1};
    }
    if(out_mm.data_ptr() == out.data_ptr()){
      out = out.reshape(size_out);
    } else{
      out.resize_(size_out);
      out.copy_(out_mm);
    }
  }
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
Tensor sum_dim_IntList_tpu(const Tensor & self, OptionalIntArrayRef dim, bool keepdim, 
                           c10::optional<ScalarType> dtype) {
    TORCH_CHECK((dtype.has_value() && dtype.value() == self.scalar_type()) || !dtype.has_value(),
                "sum_dim_IntList get error dtype setting");
    Tensor out = native::create_reduction_result(self, dim, keepdim, self.scalar_type());
    out = sum_IntList_out_tpu(self, dim, keepdim, dtype, out);
    return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sum.IntList_out", sum_IntList_out_tpu);
  m.impl("sum.dim_IntList", sum_dim_IntList_tpu);
}

Tensor &prod_int_out_tpu(const Tensor &self, long dim, bool keepdim,
                         c10::optional<ScalarType> dtype_opt, Tensor &out) {
  if (out.numel() == 0) return out;
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);

  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnReduceProdAsync(
      stream, tpu::TPUGenerateTpudnnTensor(stream, self), dim, keepdim,
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("prod.int_out", prod_int_out_tpu); }

Tensor &amax_out_tpu(const Tensor &self, IntArrayRef dim_opt, bool keepdim,
                     Tensor &out) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
#else
  std::vector<int> reduce_dim(dim_opt.begin(), dim_opt.end());
  std::vector<int> reduction_dim_vec;
  if (reduce_dim.size() > 0) {
    for (auto it : reduce_dim) {
      reduction_dim_vec.push_back(it < 0 ? it + self.dim() : it);
    }
    std::sort(reduction_dim_vec.begin(), reduction_dim_vec.end());
  } else {
    for (auto i = 0; i < self.dim(); ++i) {
      reduction_dim_vec.push_back(i);
    }
  }
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnReduceMaxOrMinAsync(
      stream, tpu::TPUGenerateTpudnnTensor(stream, self),
      reduction_dim_vec.data(), reduction_dim_vec.size(), keepdim, 0,
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("amax.out", amax_out_tpu); }

Tensor &amin_out_tpu(const Tensor &self, IntArrayRef dim_opt, bool keepdim,
                     Tensor &out) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
#else
  std::vector<int> reduce_dim(dim_opt.begin(), dim_opt.end());
  std::vector<int> reduction_dim_vec;
  if (reduce_dim.size() > 0) {
    for (auto it : reduce_dim) {
      reduction_dim_vec.push_back(it < 0 ? it + self.dim() : it);
    }
    std::sort(reduction_dim_vec.begin(), reduction_dim_vec.end());
  } else {
    for (auto i = 0; i < self.dim(); ++i) {
      reduction_dim_vec.push_back(i);
    }
  }
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnReduceMaxOrMinAsync(
      stream, tpu::TPUGenerateTpudnnTensor(stream, self),
      reduction_dim_vec.data(), reduction_dim_vec.size(), keepdim, 1,
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("amin.out", amin_out_tpu); }

Tensor var_correction_tpu(const Tensor &self, OptionalIntArrayRef dims,
                          optional<long> conrrection, bool keepdim) {
  TIMING_START;
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  auto reduce_list = dims.value_or(IntArrayRef{});
  TORCH_CHECK(reduce_list.size() <= (size_t)self.dim());

  Tensor out;
  torch::Device device(torch::kPrivateUse1);
  std::vector<int> reduce_vec;
  if (reduce_list.empty()) {
    out = torch::tensor(0.).to(device);
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnReduceVarAllAsync(
        stream, tpu::TPUGenerateTpudnnTensor(stream, self), conrrection.value(),
        keepdim, tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  } else {
    std::map<int, int> reduce_map;
    std::vector<int64_t> size_vec;
    for (auto it : reduce_list) {
      reduce_map[it] = 1;
      reduce_vec.push_back(it);
    }
    for (int i = 0; i < self.dim(); ++i) {
      if (reduce_map.find(i) == reduce_map.end()) {
        size_vec.push_back(self.size(i));
      }
    }
    IntArrayRef size(size_vec);
    out = torch::empty(size, self.dtype()).to(device);

    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnReduceVarAsync(
        stream, tpu::TPUGenerateTpudnnTensor(stream, self), reduce_vec.data(),
        reduce_vec.size(), conrrection.value(), keepdim,
        tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}
// TORCH_LIBRARY_IMPL(aten, TPU, m) {
//   m.impl("var.correction", var_correction_tpu);
// }

Tensor max_tpu(const Tensor &self) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  std::vector<int> reduction_dim_vec;
  TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  Tensor out = torch::scalar_tensor(0, options);

  for (auto i = 0; i < self.dim(); ++i) {
      reduction_dim_vec.push_back(i);
  }
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnReduceMaxOrMinAsync(
      stream, tpu::TPUGenerateTpudnnTensor(stream, self),
      reduction_dim_vec.data(), reduction_dim_vec.size(), false, 0,
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("max", max_tpu); }

Tensor min_tpu(const Tensor &self) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(self);
  std::vector<int> reduction_dim_vec;
  TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  Tensor out = torch::scalar_tensor(0, options);
  for (auto i = 0; i < self.dim(); ++i) {
      reduction_dim_vec.push_back(i);
  }
  torch::Device device(torch::kPrivateUse1);
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnReduceMaxOrMinAsync(
      stream, tpu::TPUGenerateTpudnnTensor(stream, self),
      reduction_dim_vec.data(), reduction_dim_vec.size(), false, 1,
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  TIMING_END;
  SHOW_TENSOR_OP(self, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("min", min_tpu); }
} // namespace at
