#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBase.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

#if defined BACKEND_SG2260
#define NPU_NUM 64
#define data_size_w 1
#define data_size_i 2
#define core_num 8
#define l2_mem_size 128 * 1024 * 1024
#elif defined BACKEND_SG2260E
#define NPU_NUM 64
#define data_size_w 1
#define data_size_i 2
#define core_num 4
#define l2_mem_size 16 * 1024 * 1024
#endif

#ifdef USING_PPL
#include "MlpW8A16Dq.h"
static void MlpW8A16Dq_impl(uint64_t output_addr, uint64_t input_addr,
                            uint64_t gate_weight_addr, uint64_t up_weight_addr,
                            uint64_t down_weight_addr, uint64_t gate_scale_addr,
                            uint64_t up_scale_addr, uint64_t down_scale_addr,
                            int blocksize, int batch, int input_w, int middle_w,
                            at::ScalarType dtype, at::ScalarType weight_dtype) {
  int32_t addrs_ib_ob[25];
  auto mlp_split_inner_batch_outer_batch_check_mem = [&](int tile_size) -> int {
    return kernel_mlp_split_inner_batch_outer_batch_check_mem(
        output_addr, input_addr, gate_weight_addr, up_weight_addr,
        down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
        blocksize, batch, input_w, middle_w, tile_size, addrs_ib_ob);
  };

  int32_t addrs_ib_om[26];
  auto mlp_split_inner_batch_outer_middle_w_check_mem =
      [&](int batch_slice, int middle_slice) -> int {
    return kernel_mlp_split_inner_batch_outer_middle_w_check_mem(
        output_addr, input_addr, gate_weight_addr, up_weight_addr,
        down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
        blocksize, batch, input_w, middle_w, batch_slice, middle_slice,
        addrs_ib_om);
  };

  int32_t addrs_ibm_om[26];
  auto mlp_split_inner_bm_outer_middle_w_check_mem =
      [&](int batch_slice, int middle_slice, int batch_outer) -> int {
    return kernel_mlp_split_inner_bm_outer_middle_w_check_mem(
        output_addr, input_addr, gate_weight_addr, up_weight_addr,
        down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
        blocksize, batch, input_w, middle_w, batch_slice, middle_slice,
        batch_outer, addrs_ibm_om);
  };

  auto mlp_split_inner_batch_outer_batch = [&](TPUStream stream,
                                               tpuKernelModule_t ppl_module,
                                               int tile_size) -> int {
    return kernel_mlp_split_inner_batch_outer_batch(
        stream, 
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, gate_weight_addr,
        up_weight_addr, down_weight_addr, gate_scale_addr, up_scale_addr,
        down_scale_addr, blocksize, batch, input_w, middle_w, tile_size);
  };

  auto mlp_split_inner_batch_outer_middle_w =
      [&](TPUStream stream, tpuKernelModule_t ppl_module, int batch_slice,
          int middle_slice) -> int {
    return kernel_mlp_split_inner_batch_outer_middle_w(
        stream, 
#ifndef BACKEND_SG2260
        ppl_module,
#endif
        output_addr, input_addr, gate_weight_addr,
        up_weight_addr, down_weight_addr, gate_scale_addr, up_scale_addr,
        down_scale_addr, blocksize, batch, input_w, middle_w, batch_slice,
        middle_slice);
  };

  auto mlp_split_inner_bm_outer_middle_w =
      [&](TPUStream stream, tpuKernelModule_t ppl_module, int batch_slice,
          int middle_slice, int batch_outer) -> int {
    return kernel_mlp_split_inner_bm_outer_middle_w(
        stream,
#ifndef BACKEND_SG2260
        ppl_module,
#endif        
        output_addr, input_addr, gate_weight_addr,
        up_weight_addr, down_weight_addr, gate_scale_addr, up_scale_addr,
        down_scale_addr, blocksize, batch, input_w, middle_w, batch_slice,
        middle_slice, batch_outer);
  };

  auto stream = c10_tpu::getCurrentTPUStream();
  tpuKernelModule_t ppl_module = getPplModule();

  // int NPU_NUM = 64;
  // int data_size_w = 1;
  // int data_size_i = 2;
  // int core_num = 4;
  // int l2_mem_size = 16 * 1024 * 1024;

  int ret_ib_ob = mlp_split_inner_batch_outer_batch_check_mem(NPU_NUM);
  int overload_ib_ob = middle_w * 3 * data_size_w;
  int overload_im_ob = batch * data_size_i;

  if (ret_ib_ob == 0) {
    if (overload_ib_ob < overload_im_ob) {
      uint32_t tile_size = (batch + core_num - 1) / core_num;
      while (tile_size >= 1) {
        int ret =
            mlp_split_inner_batch_outer_batch(stream, ppl_module, tile_size);
        if (ret == 0) {
          return;
        } else {
          tile_size = tile_size / 2;
          continue;
        }
      }
    } else {
      int middle_core = (middle_w + core_num - 1) / core_num;
      middle_core = (middle_core + blocksize - 1) / blocksize * blocksize;
      uint32_t tile_size = batch;
      while (tile_size >= 1) {
        int ret = mlp_split_inner_batch_outer_middle_w(stream, ppl_module,
                                                       tile_size, middle_core);
        if (ret == 0) {
          return;
        } else {
          tile_size = tile_size / 2;
          continue;
        }
      }
    }
  } else {
    int middle_core = (middle_w + core_num - 1) / core_num;
    middle_core = (middle_core + blocksize - 1) / blocksize * blocksize;
    int ret_ib_om =
        mlp_split_inner_batch_outer_middle_w_check_mem(NPU_NUM, middle_core);
    if (ret_ib_om == 0) {
      uint32_t tile_size = batch;
      while (tile_size >= 1) {
        int ret = mlp_split_inner_batch_outer_middle_w(stream, ppl_module,
                                                       tile_size, middle_core);
        if (ret == 0) {
          return;
        } else {
          tile_size = tile_size / 2;
          continue;
        }
      }
    } else {
      int batch_outer = batch;
      while (batch_outer * input_w * data_size_i > l2_mem_size) {
        batch_outer = batch_outer / 2;
      }
      int middle_slice = middle_core;
      while (middle_slice >= blocksize) {
        int ret = mlp_split_inner_bm_outer_middle_w_check_mem(
            NPU_NUM, middle_slice, batch_outer);
        if (ret == 0) {
          break;
        } else {
          middle_slice = middle_slice - blocksize;
        }
      }
      int batch_slice = batch_outer;
      while (batch_slice >= 1) {
        int ret = mlp_split_inner_bm_outer_middle_w(
            stream, ppl_module, batch_slice, middle_slice, batch_outer);
        if (ret == 0) {
          return;
        } else {
          batch_slice = batch_slice / 2;
          continue;
        }
      }
    }
  }
}
#endif
namespace at {
Tensor mlp_w8a16_dq_forward(Tensor &input, Tensor &gate_weight,
                            Tensor &up_weight, Tensor &down_weight,
                            Tensor &gate_scale, Tensor &up_scale,
                            Tensor &down_scale, Tensor &output,
                            int64_t blocksize) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE(input);
  CHECK_TENSOR_IN_DEVICE(gate_weight);
  CHECK_TENSOR_IN_DEVICE(up_weight);
  CHECK_TENSOR_IN_DEVICE(down_weight);
  CHECK_TENSOR_IN_DEVICE(gate_scale);
  CHECK_TENSOR_IN_DEVICE(up_scale);
  CHECK_TENSOR_IN_DEVICE(down_scale);
  CHECK_TENSOR_IN_DEVICE(output);

#ifdef USING_PPL
  if (usePPLKernels()) {
  at::ScalarType dtype = input.scalar_type();
  at::ScalarType weight_dtype = gate_weight.scalar_type();
  int batch = input.size(0);
  int input_w = input.size(1);
  int middle_w = gate_weight.size(0);
  MlpW8A16Dq_impl(reinterpret_cast<uint64_t>(output.data_ptr()),
                  reinterpret_cast<uint64_t>(input.data_ptr()),
                  reinterpret_cast<uint64_t>(gate_weight.data_ptr()),
                  reinterpret_cast<uint64_t>(up_weight.data_ptr()),
                  reinterpret_cast<uint64_t>(down_weight.data_ptr()),
                  reinterpret_cast<uint64_t>(gate_scale.data_ptr()),
                  reinterpret_cast<uint64_t>(up_scale.data_ptr()),
                  reinterpret_cast<uint64_t>(down_scale.data_ptr()), blocksize,
                  batch, input_w, middle_w, dtype, weight_dtype);
  } else 
#endif
{
#if defined BACKEND_SG2260
    auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnMlpW8A16DqAsync(
        stream, tpu::TPUGenerateTpudnnTensor(stream, input),
        tpu::TPUGenerateTpudnnTensor(stream, gate_weight),
        tpu::TPUGenerateTpudnnTensor(stream, up_weight),
        tpu::TPUGenerateTpudnnTensor(stream, down_weight),
        tpu::TPUGenerateTpudnnTensor(stream, gate_scale),
        tpu::TPUGenerateTpudnnTensor(stream, up_scale),
        tpu::TPUGenerateTpudnnTensor(stream, down_scale),
        tpu::TPUGenerateTpudnnTensor(stream, output), blocksize);
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#elif defined BACKEND_1684X
    TORCH_CHECK(false);
#endif
}
  TIMING_END;
  return output;
}

}  // namespace at
