#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

namespace at {
Tensor &mean_out_tpu(const Tensor &self, OptionalIntArrayRef dim_opt,
                     bool keepdim, c10::optional<ScalarType> dtype_opt,
                     Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
  auto out_cpu = mean ( self.cpu(), dim_opt, keepdim, dtype_opt );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if (self.dim() == 0) {
    CPU_IMPL_WARNING();
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), self.data_ptr(), out.nbytes());
    return out;
  }

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
  for (auto i = 0; i < reduction_dim_vec.size() - 1; ++i) {
    TORCH_CHECK(reduction_dim_vec[i] + 1 == reduction_dim_vec[i + 1],
                "Reduction only supports contiguous reduction dimension now");
  }
  TIMING_START;
  bm_status_t status =
      sgdnnReduce(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
                  reduction_dim_vec[0], reduction_dim_vec.back() + 1, keepdim,
                  0, tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::REDUCE_MEAN);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("mean.out", mean_out_tpu); }

Tensor &sum_IntList_out_tpu(const Tensor &self, OptionalIntArrayRef dim_opt,
                            bool keepdim, c10::optional<ScalarType> dtype_opt,
                            Tensor &out) {
  CHECK_TENSOR_IN_DEVICE_NO_CONTIGUOUS(self);
  CHECK_TENSOR_IN_DEVICE(out);
  auto self_ = self.contiguous();
#if 0
  auto out_cpu = sum ( self.cpu(), dim_opt, keepdim, dtype_opt );
  tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
#else
  if (self_.dim() == 0) // corner case; use cpu impl.
  {
    CPU_IMPL_WARNING();
    TIMING_START;
    auto out_cpu = sum ( self_.cpu(), dim_opt, keepdim, dtype_opt );
    tpu::TPUCopyHostToDevice ( out.data_ptr(), out_cpu.contiguous().data_ptr(), out.nbytes() );
    TIMING_END(tpu::CPU_LAYER);
  }
  else 
  {
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
  for (auto i = 0; i < reduction_dim_vec.size() - 1; ++i) {
    TORCH_CHECK(reduction_dim_vec[i] + 1 == reduction_dim_vec[i + 1],
                "Reduction only supports contiguous reduction dimension now");
  }

  TIMING_START;
  bm_status_t status =
      sgdnnReduce(tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self_),
                  reduction_dim_vec[0], reduction_dim_vec.back() + 1, keepdim,
                  1, tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::REDUCE_SUM);
  }
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("sum.IntList_out", sum_IntList_out_tpu);
}

Tensor &prod_int_out_tpu(const Tensor &self, long dim, bool keepdim,
                         c10::optional<ScalarType> dtype_opt, Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);

  TIMING_START;
  bm_status_t status = sgdnnReduceProd(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self), dim,
      keepdim, tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(BM_SUCCESS == status);
  TIMING_END(tpu::REDUCE_PROD);
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("prod.int_out", prod_int_out_tpu); }

Tensor &amax_out_tpu(const Tensor &self, IntArrayRef dim_opt, bool keepdim,
                     Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0
#else
  if (self.dim() == 0) {
    TIMING_START;
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), self.data_ptr(), out.nbytes());
    TIMING_END(tpu::COPY);
    return out;
  }
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
  TIMING_START;
  bm_status_t status = sgdnnReduceMaxOrMin(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
      reduction_dim_vec.data(), reduction_dim_vec.size(), keepdim, 0,
      tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::REDUCE_MAX);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("amax.out", amax_out_tpu); }

Tensor &amin_out_tpu(const Tensor &self, IntArrayRef dim_opt, bool keepdim,
                     Tensor &out) {
  CHECK_TENSOR_IN_DEVICE(self);
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

#else
  if (self.dim() == 0) {
    TIMING_START;
    tpu::TPUCopyDeviceToDevice(out.data_ptr(), self.data_ptr(), out.nbytes());
    TIMING_END(tpu::COPY); 
    return out;
  }
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
  TIMING_START;
  bm_status_t status = sgdnnReduceMaxOrMin(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
      reduction_dim_vec.data(), reduction_dim_vec.size(), keepdim, 1,
      tpu::TPUGenerateSgdnnTensor(out));
  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::REDUCE_MIN);
#endif
  SHOW_TENSOR_OP(self, out);
  return out;
}
TORCH_LIBRARY_IMPL(aten, TPU, m) { m.impl("amin.out", amin_out_tpu); }

Tensor var_correction_tpu(const Tensor &self, OptionalIntArrayRef dims,
                          optional<long> conrrection, bool keepdim) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  auto reduce_list = dims.value_or(IntArrayRef{});
  TORCH_CHECK(reduce_list.size() <= self.dim());

  Tensor out;
  torch::Device device(torch::kPrivateUse1);
  std::vector<int> reduce_vec;
  if (reduce_list.empty()) {
    out = torch::tensor(0.).to(device);
    TIMING_START;
    bm_status_t status = sgdnnReduceVarAll(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        conrrection.value(), keepdim, tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::REDUCE_VAR);
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

    TIMING_START;
    bm_status_t status = sgdnnReduceVar(
        tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
        reduce_vec.data(), reduce_vec.size(), conrrection.value(), keepdim,
        tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::REDUCE_VAR);
  }

  SHOW_TENSOR_OP(self, out);
  return out;
}
// TORCH_LIBRARY_IMPL(aten, TPU, m) {
//   m.impl("var.correction", var_correction_tpu);
// }

}  // namespace at
