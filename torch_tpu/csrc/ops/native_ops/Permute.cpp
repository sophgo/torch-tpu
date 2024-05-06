// #include <ATen/EmptyTensor.h>
// #include <ATen/core/TensorBase.h>
// #include <torch/library.h>
// #include <torch/torch.h>


// #include "common/config.h"
// #include "sgdnn_api.h"
// #include "TPUTorchUtils.h"

// namespace at {

// Tensor permute_tpu(const Tensor &self, IntArrayRef dim_order) {
//     if(self.dim() > 0) {
//         TORCH_CHECK(self.device().type() == DeviceType::TPU, "self is not in TPU device");
//     }
//     Tensor out = empty_like(self);
// #if 0
// #else
//     if(self.dim() == 0 || self.dim() == 1) {
//         CPU_IMPL_WARNING();
//         TIMING_START;
//         tpu::TPUCopyHostToDevice(out.data_ptr(), self.contiguous().data_ptr(), out.nbytes());
//         TIMING_END(tpu::CPU_LAYER);
//     }
//     else {
//         std::vector<int> dim_order_vec;
//         std::vector<int64_t> out_shape_vec;
//         int idx = 0;
//         for(auto it : dim_order) {
//             dim_order_vec.push_back(it);
//             out_shape_vec.push_back(self.size(it));
//         }
//         IntArrayRef out_shape_arr(out_shape_vec);
//         out = out.reshape(out_shape_arr);
//         TIMING_START;
//         #if defined BACKEND_1684X
//         auto status = sgdnnPermute( tpu::TPUGetDeviceHandle(),
//                                            tpu::TPUGenerateSgdnnTensor(self),
//                                            dim_order_vec.data(),
//                                            tpu::TPUGenerateSgdnnTensor(out));
//         TORCH_CHECK(status == BM_SUCCESS);
//         #elif defined BACKEND_SG2260
//         #endif
//         TIMING_END(tpu::PERMUTE);
//     }
// #endif
//     SHOW_TENSOR_OP(self);
//     return out;
// }
// TORCH_LIBRARY_IMPL(aten, TPU, m) {
//     m.impl("permute", permute_tpu);
// }

// }   // namespace at