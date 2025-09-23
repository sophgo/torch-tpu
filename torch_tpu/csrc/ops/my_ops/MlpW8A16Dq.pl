#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

#ifdef __bm1690__
#define CORE_NUM 8
#elif defined __sg2260e__
#define CORE_NUM 4
#endif

template <typename TYPE, typename W_TYPE>
void weight_dequant_bc(tensor<TYPE> &dequant_tensor,
                       tensor<W_TYPE> &quantized_tensor,
                       tensor<TYPE> &scale_tensor, tensor<TYPE> &buffer_tensor,
                       int row, int col, int blocksize) {
  int dtype_eu_num = ppl::get_eu_num<TYPE>();
  dim4 weight_shape = {1, row, 1, align(col, blocksize)};
  dim4 weight_r_shape = {1, row, 1, col};
  dim4 weight_stride = {1, align(align(col, blocksize), dtype_eu_num),
                        align(col, blocksize), 1};
  dim4 offset_zero = {0, 0, 0, 0};
  ppl::tiu::cast(buffer_tensor.view(weight_r_shape, weight_stride),
                 quantized_tensor);
  dim4 scale_shape = {div_up(row, blocksize), min(row, blocksize),
                      div_up(col, blocksize), blocksize};

  dim4 scale_stride = {align(div_up(col, blocksize), dtype_eu_num), 0, 1, 0};
  dim4 dst_stride = {align(col, dtype_eu_num) *
                         (div_up(scale_shape.c, NPU_NUM)),
                     align(col, dtype_eu_num), blocksize, 1};
  ppl::tiu::fmul(dequant_tensor.view(scale_shape, dst_stride),
                 scale_tensor.view(scale_shape, scale_stride),
                 buffer_tensor.view(scale_shape));
}

template <typename TYPE, typename W_TYPE>
void mlp_split_inner_batch_outer_batch(
    TYPE *output_addr, TYPE *input_addr, W_TYPE *gate_weight_addr,
    W_TYPE *up_weight_addr, W_TYPE *down_weight_addr, TYPE *gate_scale_addr,
    TYPE *up_scale_addr, TYPE *down_scale_addr, const int batch_slice_mem,
    const int block_size, const int batch, const int input_w,
    const int middle_w) {
  // weight is loaded only once
  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();

  int scale_n = div_up(input_w, block_size);
  int scale_m = div_up(middle_w, block_size);
  dim4 input_shape = {1, batch_slice_mem, 1, input_w};
  dim4 weight_shape = {1, middle_w, 1, input_w};
  dim4 scale_shape = {scale_m, min(middle_w, NPU_NUM), 1, scale_n};
  dim4 scale_gshape = {scale_m, 1, 1, scale_n};
  dim4 buffer1_shape = {1, align(middle_w, block_size), 1,
                        align(input_w, block_size)};
  dim4 buffer2_shape = {1, batch_slice_mem, 1, middle_w};

  auto input_tensor = make_tensor<TYPE>(input_shape, input_shape);
  auto weight0_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto weight1_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto weight2_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto scale0_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto scale1_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto scale2_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto buffer1_tensor = make_tensor<TYPE>(buffer1_shape, buffer1_shape);
  auto buffer2_tensor = make_tensor<TYPE>(buffer2_shape, buffer2_shape);
  auto buffer3_tensor = make_tensor<FP32>(buffer2_shape, buffer2_shape);
  auto buffer4_tensor = make_tensor<FP32>(buffer2_shape, buffer2_shape);
  auto buffer5_tensor = make_tensor<TYPE>(buffer2_shape, buffer2_shape);
  auto outputkk_tensor = make_tensor<TYPE>(input_shape, input_shape);

  auto weight0_gtensor =
      gtensor<W_TYPE>(weight_shape, GLOBAL, gate_weight_addr);
  auto weight1_gtensor = gtensor<W_TYPE>(weight_shape, GLOBAL, up_weight_addr);
  auto weight2_gtensor =
      gtensor<W_TYPE>(weight_shape, GLOBAL, down_weight_addr);
  auto scale0_gtensor = gtensor<TYPE>(scale_gshape, GLOBAL, gate_scale_addr);
  auto scale1_gtensor = gtensor<TYPE>(scale_gshape, GLOBAL, up_scale_addr);
  auto scale2_gtensor = gtensor<TYPE>(scale_gshape, GLOBAL, down_scale_addr);

  dim4 input_gshape = {1, batch, 1, input_w};
  dim4 output_gshape = {1, batch, 1, input_w};
  auto input_gtensor = gtensor<TYPE>(input_gshape, GLOBAL, input_addr);
  auto output_gtensor = gtensor<TYPE>(output_gshape, GLOBAL, output_addr);

  dim4 offset_zero = {0, 0, 0, 0};
  int batch_core = div_up(batch, core_num);
  int batch_offset = batch_core * core_idx;
  batch_core = min(batch - batch_offset, batch_core);
  if (batch_core <= 0)
    return;
  int batch_slice = min(batch_slice_mem, batch_core);
  int slice_num = div_up(batch_core, batch_slice);

  int idx = 0;
  while (idx < slice_num + 2) {
    ppl::parallel_start();
    if (idx > 1) {
      int load_batch = min(batch_core - batch_slice * (idx - 2), batch_slice);
      dim4 output_real_shape = {1, load_batch, 1, input_w};
      dim4 store_offset = {0, batch_offset + batch_slice * (idx - 2), 0, 0};
      ppl::dma::store(output_gtensor.sub_view(output_real_shape, store_offset),
                      outputkk_tensor.view(output_real_shape));
    }
    if (idx > 0 && idx < slice_num + 1) {
      // calc and load weight1
      if (idx == 1) {
        ppl::dma::load(weight1_tensor, weight1_gtensor);
        ppl::dma::load(scale1_tensor.sub_view(scale_gshape, offset_zero),
                       scale1_gtensor);
      }
      dim4 calc_in_shape = {
          1, min(batch_slice, batch_core - batch_slice * (idx - 1)), 1,
          input_w};
      dim4 calc_mid_shape = {
          1, min(batch_slice, batch_core - batch_slice * (idx - 1)), 1,
          middle_w};
      ppl::tiu::broadcast(scale0_tensor, scale0_tensor.view(scale_gshape));
      weight_dequant_bc(buffer1_tensor, weight0_tensor, scale0_tensor,
                        buffer1_tensor, middle_w, input_w, block_size);
      ppl::tiu::fmm2_nt(buffer2_tensor.view(calc_mid_shape),
                        input_tensor.view(calc_in_shape),
                        buffer1_tensor.view(weight_shape), false);
      ppl::tiu::cast(buffer3_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape));
      ppl::tiu::fmul(buffer4_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape), -1.0);
      exp_no_overflow(buffer3_tensor.view(calc_mid_shape),
                      buffer4_tensor.view(calc_mid_shape), &buffer2_shape,
                      &calc_mid_shape);
      ppl::tiu::fadd(buffer4_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape), 1.0);
      ppl::tiu::fdiv(buffer3_tensor.view(calc_mid_shape), 1.0,
                     buffer4_tensor.view(calc_mid_shape), 4);
      ppl::tiu::cast(buffer5_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape));
      ppl::tiu::fmul(buffer5_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape),
                     buffer5_tensor.view(calc_mid_shape));

      ppl::parallel_end();
      ppl::parallel_start();

      if (idx == 1) {
        ppl::dma::load(weight2_tensor, weight2_gtensor);
        ppl::dma::load(scale2_tensor.sub_view(scale_gshape, offset_zero),
                       scale2_gtensor);
      }

      ppl::tiu::broadcast(scale1_tensor, scale1_tensor.view(scale_gshape));
      weight_dequant_bc(buffer1_tensor, weight1_tensor, scale1_tensor,
                        buffer1_tensor, middle_w, input_w, block_size);
      ppl::tiu::fmm2_nt(buffer2_tensor.view(calc_mid_shape),
                        input_tensor.view(calc_in_shape),
                        buffer1_tensor.view(weight_shape), false);
      ppl::tiu::fmul(buffer5_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape),
                     buffer5_tensor.view(calc_mid_shape));

      ppl::parallel_end();
      ppl::parallel_start();

      ppl::tiu::broadcast(scale2_tensor, scale2_tensor.view(scale_gshape));
      weight_dequant_bc(buffer1_tensor, weight2_tensor, scale2_tensor,
                        buffer1_tensor, middle_w, input_w, block_size);
      ppl::tiu::fmm2_nn(outputkk_tensor.view(calc_in_shape),
                        buffer5_tensor.view(calc_mid_shape),
                        buffer1_tensor.view(weight_shape));
    }

    if (idx == 0) {
      // load weight0
      ppl::dma::load(weight0_tensor, weight0_gtensor);
      ppl::dma::load(scale0_tensor.sub_view(scale_gshape, offset_zero),
                     scale0_gtensor);
    }
    if (idx < slice_num) {
      // load input
      int load_batch = min(batch_core - batch_slice * idx, batch_slice);
      dim4 input_shape = {1, load_batch, 1, input_w};
      dim4 input_offset = {0, batch_offset + batch_slice * idx, 0, 0};
      ppl::dma::load(input_tensor.view(input_shape),
                     input_gtensor.sub_view(input_shape, input_offset));
    }
    ppl::parallel_end();
    idx++;
  }
}

template <typename TYPE, typename W_TYPE>
void mlp_split_inner_batch_outer_middle_w(
    TYPE *output_addr, TYPE *input_addr, W_TYPE *gate_weight_addr,
    W_TYPE *up_weight_addr, W_TYPE *down_weight_addr, TYPE *gate_scale_addr,
    TYPE *up_scale_addr, TYPE *down_scale_addr, int batch_slice_mem,
    int middle_slice_mem, int block_size, int batch, int input_w, int middle_w,
    gtensor<TYPE> &output_shared) {
  // weight is loaded only once
  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();

  int scale_n = div_up(input_w, block_size);
  int scale_m = div_up(middle_slice_mem, block_size);
  dim4 input_shape = {1, batch_slice_mem, 1, input_w};
  dim4 weight_shape = {1, middle_slice_mem, 1, input_w};
  dim4 scale_shape = {scale_m, min(middle_slice_mem, NPU_NUM), 1, scale_n};
  dim4 scale_gshape = {scale_m, 1, 1, scale_n};
  dim4 buffer1_shape = {1, middle_slice_mem, 1, align(input_w, block_size)};
  dim4 buffer2_shape = {1, batch_slice_mem, 1, middle_slice_mem};

  auto input_tensor = make_tensor<TYPE>(input_shape, input_shape);
  auto weight0_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto weight1_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto weight2_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto scale0_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto scale1_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto scale2_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto buffer1_tensor = make_tensor<TYPE>(buffer1_shape, buffer1_shape);
  auto buffer2_tensor = make_tensor<TYPE>(buffer2_shape, buffer2_shape);
  auto buffer3_tensor = make_tensor<FP32>(buffer2_shape, buffer2_shape);
  auto buffer4_tensor = make_tensor<FP32>(buffer2_shape, buffer2_shape);
  auto buffer5_tensor = make_tensor<TYPE>(buffer2_shape, buffer2_shape);
  auto outputkk_tensor = make_tensor<TYPE>(input_shape, input_shape);

  dim4 weight_gl_shape = {1, middle_w, 1, input_w};
  dim4 scale_gl_shape = {div_up(middle_w, block_size), 1, 1, scale_n};

  auto weight0_gtensor =
      gtensor<W_TYPE>(weight_gl_shape, GLOBAL, gate_weight_addr);
  auto weight1_gtensor =
      gtensor<W_TYPE>(weight_gl_shape, GLOBAL, up_weight_addr);
  auto weight2_gtensor =
      gtensor<W_TYPE>(weight_gl_shape, GLOBAL, down_weight_addr);
  auto scale0_gtensor = gtensor<TYPE>(scale_gl_shape, GLOBAL, gate_scale_addr);
  auto scale1_gtensor = gtensor<TYPE>(scale_gl_shape, GLOBAL, up_scale_addr);
  auto scale2_gtensor = gtensor<TYPE>(scale_gl_shape, GLOBAL, down_scale_addr);

  dim4 input_gshape = {1, batch, 1, input_w};
  // dim4 output_gshape = {1, batch, 1, input_w};
  auto input_gtensor = gtensor<TYPE>(input_gshape, GLOBAL, input_addr);
  // auto output_gtensor = gtensor<TYPE>(output_gshape, GLOBAL, output_addr);
  // auto output_shared = gtensor<TYPE>(output_gshape, L2);

  dim4 offset_zero = {0, 0, 0, 0};
  int batch_core = batch;
  int batch_slice = min(batch_slice_mem, batch_core);
  int slice_num = div_up(batch_core, batch_slice);

  int middle_core = div_up(middle_w, core_num);
  // middle_core = middle_core > NPU_NUM ? align(middle_core, NPU_NUM)
  //                                     : NPU_NUM / (NPU_NUM / middle_core);

  // load scale only once:
  // middle_core % block_size == 0 || block_size % middle_core == 0
  if (middle_core < block_size)
    middle_core = block_size / (block_size / middle_core);

  int num_up = 0;
  int middle_offset = 0;

  if (middle_core > block_size) {
    middle_core = middle_core / block_size * block_size;
    num_up = div_up(middle_w - middle_core * core_num, block_size);
  }
  if (core_idx < num_up) {
    middle_core = middle_core + block_size;
    middle_offset = middle_core * core_idx;
  }
  if (core_idx >= num_up) {
    middle_offset =
        (middle_core + block_size) * num_up + middle_core * (core_idx - num_up);
  }
  if (middle_core + middle_offset > middle_w) {
    middle_core = middle_w - middle_offset;
  }
  if (middle_core <= 0)
    return;

  assert(middle_core <= middle_slice_mem);
  dim4 weight_offset = {0, middle_offset, 0, 0};
  dim4 weight_r_shape = {1, middle_core, 1, input_w};
  dim4 scale_offset = {0, middle_offset / block_size, 0, 0};
  dim4 scale_r_shape = {div_up(middle_core, block_size), 1, 1, scale_n};
  dim4 scale_bc_shape = {div_up(middle_core, block_size),
                         min(middle_core, NPU_NUM), 1, scale_n};
  dim4 buffer1_r_shape = {1, middle_core, 1, align(input_w, block_size)};

  dim4 view_shape = {1, 2, 1, 10};

  int idx = 0;
  while (idx < slice_num + 2) {
    ppl::parallel_start();
    if (idx > 1) {
      int load_batch = min(batch_core - batch_slice * (idx - 2), batch_slice);
      dim4 output_real_shape = {1, load_batch, 1, input_w};
      dim4 store_offset = {0, batch_slice * (idx - 2), 0, 0};
      ppl::dma::reduce(output_shared.sub_view(output_real_shape, store_offset),
                       outputkk_tensor.view(output_real_shape),
                       ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
    }
    if (idx > 0 && idx < slice_num + 1) {
      // calc and load weight1
      if (idx == 1) {
        ppl::dma::load(weight1_tensor.view(weight_r_shape),
                       weight1_gtensor.sub_view(weight_r_shape, weight_offset));
        ppl::dma::load(scale1_tensor.sub_view(scale_r_shape, offset_zero),
                       scale1_gtensor.sub_view(scale_r_shape, scale_offset));
      }
      dim4 calc_in_shape = {
          1, min(batch_slice, batch_core - batch_slice * (idx - 1)), 1,
          input_w};
      dim4 calc_mid_shape = {
          1, min(batch_slice, batch_core - batch_slice * (idx - 1)), 1,
          middle_core};
      ppl::tiu::broadcast(scale0_tensor.view(scale_bc_shape),
                          scale0_tensor.view(scale_r_shape));
      weight_dequant_bc(buffer1_tensor.view(buffer1_r_shape),
                        weight0_tensor.view(weight_r_shape),
                        scale0_tensor.view(scale_bc_shape),
                        buffer1_tensor.view(buffer1_r_shape), middle_core,
                        input_w, block_size);
      ppl::tiu::fmm2_nt(buffer2_tensor.view(calc_mid_shape),
                        input_tensor.view(calc_in_shape),
                        buffer1_tensor.view(weight_r_shape), false);
      ppl::tiu::cast(buffer3_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape));
      ppl::tiu::fmul(buffer4_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape), -1.0);
      exp_no_overflow(buffer3_tensor.view(calc_mid_shape),
                      buffer4_tensor.view(calc_mid_shape), &buffer2_shape,
                      &calc_mid_shape);
      ppl::tiu::fadd(buffer4_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape), 1.0);
      ppl::tiu::fdiv(buffer3_tensor.view(calc_mid_shape), 1.0,
                     buffer4_tensor.view(calc_mid_shape), 4);
      ppl::tiu::cast(buffer5_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape));
      ppl::tiu::fmul(buffer5_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape),
                     buffer5_tensor.view(calc_mid_shape));

      ppl::parallel_end();
      ppl::parallel_start();

      if (idx == 1) {
        ppl::dma::load(weight2_tensor.view(weight_r_shape),
                       weight2_gtensor.sub_view(weight_r_shape, weight_offset));
        ppl::dma::load(scale2_tensor.sub_view(scale_r_shape, offset_zero),
                       scale2_gtensor.sub_view(scale_r_shape, scale_offset));
      }

      ppl::tiu::broadcast(scale1_tensor.view(scale_bc_shape),
                          scale1_tensor.view(scale_r_shape));
      weight_dequant_bc(buffer1_tensor.view(buffer1_r_shape),
                        weight1_tensor.view(weight_r_shape),
                        scale1_tensor.view(scale_bc_shape),
                        buffer1_tensor.view(buffer1_r_shape), middle_core,
                        input_w, block_size);
      ppl::tiu::fmm2_nt(buffer2_tensor.view(calc_mid_shape),
                        input_tensor.view(calc_in_shape),
                        buffer1_tensor.view(weight_r_shape), false);
      ppl::tiu::fmul(buffer5_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape),
                     buffer5_tensor.view(calc_mid_shape));

      ppl::parallel_end();
      ppl::parallel_start();

      ppl::tiu::broadcast(scale2_tensor.view(scale_bc_shape),
                          scale2_tensor.view(scale_r_shape));
      weight_dequant_bc(buffer1_tensor.view(buffer1_r_shape),
                        weight2_tensor.view(weight_r_shape),
                        scale2_tensor.view(scale_bc_shape),
                        buffer1_tensor.view(buffer1_r_shape), middle_core,
                        input_w, block_size);
      ppl::tiu::fmm2_nn(outputkk_tensor.view(calc_in_shape),
                        buffer5_tensor.view(calc_mid_shape),
                        buffer1_tensor.view(weight_r_shape));
    }

    if (idx == 0) {
      // load weight0
      ppl::dma::load(weight0_tensor.view(weight_r_shape),
                     weight0_gtensor.sub_view(weight_r_shape, weight_offset));
      ppl::dma::load(scale0_tensor.sub_view(scale_r_shape, offset_zero),
                     scale0_gtensor.sub_view(scale_r_shape, scale_offset));
    }
    if (idx < slice_num) {
      // load input
      int load_batch = min(batch_core - batch_slice * idx, batch_slice);
      dim4 load_shape = {1, load_batch, 1, input_w};
      dim4 load_offset = {0, batch_slice * idx, 0, 0};
      ppl::dma::load(input_tensor.view(load_shape),
                     input_gtensor.sub_view(load_shape, load_offset));
    }
    ppl::parallel_end();
    idx++;
  }
}

template <typename TYPE, typename W_TYPE>
void mlp_split_inner_bm_outer_middle_w(
    TYPE *output_addr, TYPE *input_addr, W_TYPE *gate_weight_addr,
    W_TYPE *up_weight_addr, W_TYPE *down_weight_addr, TYPE *gate_scale_addr,
    TYPE *up_scale_addr, TYPE *down_scale_addr, int batch_slice_mem,
    int middle_slice_mem, int block_size, int batch_num, int batch_offset,
    int batch, int input_w, int middle_w, gtensor<TYPE> &output_shared) {
  // inner spilt middle_w and batch
  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();

  int scale_n = div_up(input_w, block_size);
  int scale_m = div_up(middle_slice_mem, block_size);
  dim4 input_shape = {1, batch_slice_mem, 1, input_w};
  dim4 weight_shape = {1, middle_slice_mem, 1, input_w};
  dim4 scale_shape = {scale_m, min(middle_slice_mem, NPU_NUM), 1, scale_n};
  dim4 scale_gshape = {scale_m, 1, 1, scale_n};
  dim4 buffer1_shape = {1, align(middle_slice_mem, block_size), 1,
                        align(input_w, block_size)};
  dim4 buffer2_shape = {1, batch_slice_mem, 1, middle_slice_mem};

  auto input_tensor = make_tensor<TYPE>(input_shape, input_shape);
  auto weight0_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto weight1_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto weight2_tensor = make_tensor<W_TYPE>(weight_shape, weight_shape);
  auto scale0_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto scale1_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto scale2_tensor = make_tensor<TYPE>(scale_shape, scale_shape);
  auto buffer1_tensor = make_tensor<TYPE>(buffer1_shape, buffer1_shape);
  auto buffer2_tensor = make_tensor<TYPE>(buffer2_shape, buffer2_shape);
  auto buffer3_tensor = make_tensor<FP32>(buffer2_shape, buffer2_shape);
  auto buffer4_tensor = make_tensor<FP32>(buffer2_shape, buffer2_shape);
  auto buffer5_tensor = make_tensor<TYPE>(buffer2_shape, buffer2_shape);
  auto outputkk_tensor = make_tensor<TYPE>(input_shape, input_shape);

  dim4 weight_gl_shape = {1, middle_w, 1, input_w};
  dim4 scale_gl_shape = {div_up(middle_w, block_size), 1, 1, scale_n};

  auto weight0_gtensor =
      gtensor<W_TYPE>(weight_gl_shape, GLOBAL, gate_weight_addr);
  auto weight1_gtensor =
      gtensor<W_TYPE>(weight_gl_shape, GLOBAL, up_weight_addr);
  auto weight2_gtensor =
      gtensor<W_TYPE>(weight_gl_shape, GLOBAL, down_weight_addr);
  auto scale0_gtensor = gtensor<TYPE>(scale_gl_shape, GLOBAL, gate_scale_addr);
  auto scale1_gtensor = gtensor<TYPE>(scale_gl_shape, GLOBAL, up_scale_addr);
  auto scale2_gtensor = gtensor<TYPE>(scale_gl_shape, GLOBAL, down_scale_addr);

  dim4 input_init_gshape = {1, batch, 1, input_w};
  dim4 input_gshape = {1, batch_num, 1, input_w};
  dim4 input_goffset = {0, batch_offset, 0, 0};
  auto input_init_gtensor =
      gtensor<TYPE>(input_init_gshape, GLOBAL, input_addr);
  auto input_gtensor = input_init_gtensor.sub_view(input_gshape, input_goffset);

  dim4 offset_zero = {0, 0, 0, 0};
  int batch_core = batch_num;
  int batch_slice = min(batch_slice_mem, batch_core);
  int b_slice_num = div_up(batch_core, batch_slice);

  int middle_core = div_up(middle_w, core_num);
  // middle_core = middle_core > NPU_NUM ? align(middle_core, NPU_NUM)
  //                                     : NPU_NUM / (NPU_NUM / middle_core);

  // load scale only once:
  // middle_core % block_size == 0 || block_size % middle_core == 0
  if (middle_core < block_size)
    middle_core = block_size / (block_size / middle_core);

  int num_up = 0;
  int middle_offset = 0;

  if (middle_core > block_size) {
    middle_core = middle_core / block_size * block_size;
    num_up = div_up(middle_w - middle_core * core_num, block_size);
  }
  if (core_idx < num_up) {
    middle_core = middle_core + block_size;
    middle_offset = middle_core * core_idx;
  }
  if (core_idx >= num_up) {
    middle_offset =
        (middle_core + block_size) * num_up + middle_core * (core_idx - num_up);
  }
  if (middle_core + middle_offset > middle_w) {
    middle_core = middle_w - middle_offset;
  }
  if (middle_core <= 0)
    return;

  int middle_slice = min(middle_core, middle_slice_mem);

  int m_slice_num = div_up(middle_core, middle_slice);
  int slice_num = m_slice_num * b_slice_num;

  int idx = 0;
  while (idx < slice_num + 2) {
    ppl::parallel_start();
    if (idx > 1) {
      int bidx = (idx - 2) % b_slice_num;
      int load_batch = min(batch_core - batch_slice * bidx, batch_slice);
      dim4 output_real_shape = {1, load_batch, 1, input_w};
      dim4 store_offset = {0, batch_slice * bidx, 0, 0};
      ppl::dma::reduce(output_shared.sub_view(output_real_shape, store_offset),
                       outputkk_tensor.view(output_real_shape),
                       ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
    }
    if (idx > 0 && idx < slice_num + 1) {
      // calc and load weight1
      int bidx = (idx - 1) % b_slice_num;
      int midx = (idx - 1) / b_slice_num;
      int calc_middle = min(middle_core - middle_slice * midx, middle_slice);
      int calc_batch = min(batch_core - batch_slice * bidx, batch_slice);

      dim4 weight_offset = {0, middle_offset + middle_slice * midx, 0, 0};
      dim4 weight_r_shape = {1, calc_middle, 1, input_w};
      dim4 scale_offset = {
          0, (middle_offset + middle_slice * midx) / block_size, 0, 0};
      dim4 scale_r_shape = {div_up(calc_middle, block_size), 1, 1, scale_n};
      dim4 scale_bc_shape = {div_up(calc_middle, block_size),
                             min(calc_middle, NPU_NUM), 1, scale_n};

      if (bidx == 0) {
        ppl::dma::load(weight1_tensor.view(weight_r_shape),
                       weight1_gtensor.sub_view(weight_r_shape, weight_offset));
        ppl::dma::load(scale1_tensor.sub_view(scale_r_shape, offset_zero),
                       scale1_gtensor.sub_view(scale_r_shape, scale_offset));
      }

      dim4 calc_in_shape = {1, calc_batch, 1, input_w};
      dim4 calc_mid_shape = {1, calc_batch, 1, calc_middle};
      dim4 buffer1_r_shape = {1, align(calc_middle, block_size), 1,
                              align(input_w, block_size)};

      ppl::tiu::broadcast(scale0_tensor.view(scale_bc_shape),
                          scale0_tensor.view(scale_r_shape));
      weight_dequant_bc(buffer1_tensor.view(buffer1_r_shape),
                        weight0_tensor.view(weight_r_shape),
                        scale0_tensor.view(scale_bc_shape),
                        buffer1_tensor.view(buffer1_r_shape), calc_middle,
                        input_w, block_size);
      ppl::tiu::fmm2_nt(buffer2_tensor.view(calc_mid_shape),
                        input_tensor.view(calc_in_shape),
                        buffer1_tensor.view(weight_r_shape), false);
      ppl::tiu::cast(buffer3_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape));
      ppl::tiu::fmul(buffer4_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape), -1.0);
      exp_no_overflow(buffer3_tensor.view(calc_mid_shape),
                      buffer4_tensor.view(calc_mid_shape), &buffer2_shape,
                      &calc_mid_shape);
      ppl::tiu::fadd(buffer4_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape), 1.0);
      ppl::tiu::fdiv(buffer3_tensor.view(calc_mid_shape), 1.0,
                     buffer4_tensor.view(calc_mid_shape), 4);
      ppl::tiu::cast(buffer5_tensor.view(calc_mid_shape),
                     buffer3_tensor.view(calc_mid_shape));
      ppl::tiu::fmul(buffer5_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape),
                     buffer5_tensor.view(calc_mid_shape));

      ppl::parallel_end();
      ppl::parallel_start();

      if (bidx == 0) {
        ppl::dma::load(weight2_tensor.view(weight_r_shape),
                       weight2_gtensor.sub_view(weight_r_shape, weight_offset));
        ppl::dma::load(scale2_tensor.sub_view(scale_r_shape, offset_zero),
                       scale2_gtensor.sub_view(scale_r_shape, scale_offset));
      }

      ppl::tiu::broadcast(scale1_tensor.view(scale_bc_shape),
                          scale1_tensor.view(scale_r_shape));
      weight_dequant_bc(buffer1_tensor.view(buffer1_r_shape),
                        weight1_tensor.view(weight_r_shape),
                        scale1_tensor.view(scale_bc_shape),
                        buffer1_tensor.view(buffer1_r_shape), calc_middle,
                        input_w, block_size);
      ppl::tiu::fmm2_nt(buffer2_tensor.view(calc_mid_shape),
                        input_tensor.view(calc_in_shape),
                        buffer1_tensor.view(weight_r_shape), false);
      ppl::tiu::fmul(buffer5_tensor.view(calc_mid_shape),
                     buffer2_tensor.view(calc_mid_shape),
                     buffer5_tensor.view(calc_mid_shape));

      ppl::parallel_end();
      ppl::parallel_start();

      ppl::tiu::broadcast(scale2_tensor.view(scale_bc_shape),
                          scale2_tensor.view(scale_r_shape));
      weight_dequant_bc(buffer1_tensor.view(buffer1_r_shape),
                        weight2_tensor.view(weight_r_shape),
                        scale2_tensor.view(scale_bc_shape),
                        buffer1_tensor.view(buffer1_r_shape), calc_middle,
                        input_w, block_size);
      ppl::tiu::fmm2_nn(outputkk_tensor.view(calc_in_shape),
                        buffer5_tensor.view(calc_mid_shape),
                        buffer1_tensor.view(weight_r_shape));
    }

    if (idx % b_slice_num == 0 && idx / b_slice_num < m_slice_num) {
      // load weight0
      int midx = idx / b_slice_num;
      int load_middle = min(middle_core - middle_slice * midx, middle_slice);

      dim4 weight_offset = {0, middle_offset + middle_slice * midx, 0, 0};
      dim4 weight_r_shape = {1, load_middle, 1, input_w};
      dim4 scale_offset = {
          0, (middle_offset + middle_slice * midx) / block_size, 0, 0};
      dim4 scale_r_shape = {div_up(load_middle, block_size), 1, 1, scale_n};
      dim4 scale_bc_shape = {div_up(load_middle, block_size),
                             min(load_middle, NPU_NUM), 1, scale_n};

      ppl::dma::load(weight0_tensor.view(weight_r_shape),
                     weight0_gtensor.sub_view(weight_r_shape, weight_offset));
      ppl::dma::load(scale0_tensor.sub_view(scale_r_shape, offset_zero),
                     scale0_gtensor.sub_view(scale_r_shape, scale_offset));
    }
    if (idx < slice_num) {
      // load input
      int bidx = idx % b_slice_num;
      int load_batch = min(batch_core - batch_slice * bidx, batch_slice);
      dim4 load_shape = {1, load_batch, 1, input_w};
      dim4 load_offset = {0, batch_slice * bidx, 0, 0};
      ppl::dma::load(input_tensor.view(load_shape),
                     input_gtensor.sub_view(load_shape, load_offset));
    }
    ppl::parallel_end();
    idx++;
  }
}

__KERNEL__ void kernel_mlp_split_inner_batch_outer_batch(
    bf16 *output_addr, bf16 *input_addr, fp8e4m3 *gate_weight_addr,
    fp8e4m3 *up_weight_addr, fp8e4m3 *down_weight_addr, bf16 *gate_scale_addr,
    bf16 *up_scale_addr, bf16 *down_scale_addr, const int blocksize,
    const int batch, const int input_w, const int middle_w,
    const int batch_slice) {
  ppl::set_core_num(CORE_NUM);

  mlp_split_inner_batch_outer_batch(
      output_addr, input_addr, gate_weight_addr, up_weight_addr,
      down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
      batch_slice, blocksize, batch, input_w, middle_w);
}

__KERNEL__ void kernel_mlp_split_inner_batch_outer_middle_w(
    bf16 *output_addr, bf16 *input_addr, fp8e4m3 *gate_weight_addr,
    fp8e4m3 *up_weight_addr, fp8e4m3 *down_weight_addr, bf16 *gate_scale_addr,
    bf16 *up_scale_addr, bf16 *down_scale_addr, const int blocksize,
    const int batch, const int input_w, const int middle_w,
    const int batch_slice, const int middle_slice) {
  ppl::set_core_num(CORE_NUM);

  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();
  int ele_num = batch * input_w;
  int ele_slice = div_up(ele_num, core_num);
  int offset = ele_slice * core_idx;
  ele_slice = min(ele_slice, ele_num - offset);
  dim4 core_shape = {1, 1, 1, ele_slice};
  dim4 core_offset = {0, 0, 0, offset};
  dim4 output_shape = {1, 1, 1, ele_num};
  dim4 src_shape = {1, batch, 1, input_w};

  auto src_gtensor = gtensor<bf16>(output_shape, L2);
  ppl::sdma::zero(src_gtensor.sub_view(core_shape, core_offset));
  ppl::sync();

  mlp_split_inner_batch_outer_middle_w(
      output_addr, input_addr, gate_weight_addr, up_weight_addr,
      down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
      batch_slice, middle_slice, blocksize, batch, input_w, middle_w,
      src_gtensor.view(src_shape));

  ppl::sync();

  auto output_gtensor = gtensor<bf16>(output_shape, GLOBAL, output_addr);
  ppl::sdma::move(output_gtensor.sub_view(core_shape, core_offset),
                  src_gtensor.sub_view(core_shape, core_offset));
}

__KERNEL__ void kernel_mlp_split_inner_bm_outer_middle_w(
    bf16 *output_addr, bf16 *input_addr, fp8e4m3 *gate_weight_addr,
    fp8e4m3 *up_weight_addr, fp8e4m3 *down_weight_addr, bf16 *gate_scale_addr,
    bf16 *up_scale_addr, bf16 *down_scale_addr, const int blocksize,
    const int batch, const int input_w, const int middle_w,
    const int batch_slice, const int middle_slice, const int batch_outer) {
  ppl::set_core_num(CORE_NUM);

  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();

  dim4 fake_shape = {1, 1, 1, batch_outer * input_w};
  dim4 output_shape = {1, 1, 1, batch * input_w};

  auto src_gtensor = gtensor<bf16>(fake_shape, L2);
  auto output_gtensor = gtensor<bf16>(output_shape, GLOBAL, output_addr);

  int batch_slice_outer = min(batch, batch_outer);
  int batch_iters = div_up(batch, batch_slice_outer);
  int batch_idx = 0;
  while (batch_idx < batch_iters) {
    int batch_offset = batch_slice_outer * batch_idx;
    int batch_num = min(batch_slice_outer, batch - batch_offset);
    int ele_num = batch_num * input_w;
    int ele_slice = div_up(ele_num, core_num);
    int offset = ele_slice * core_idx;
    ele_slice = min(ele_slice, ele_num - offset);
    dim4 core_shape = {1, 1, 1, ele_slice};
    dim4 core_offset = {0, 0, 0, offset};
    dim4 output_shape = {1, 1, 1, ele_num};
    dim4 src_shape = {1, batch_num, 1, input_w};

    ppl::sdma::zero(src_gtensor.sub_view(core_shape, core_offset));
    ppl::sync();

    mlp_split_inner_bm_outer_middle_w(
        output_addr, input_addr, gate_weight_addr, up_weight_addr,
        down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
        batch_slice, middle_slice, blocksize, batch_num, batch_offset, batch,
        input_w, middle_w, src_gtensor.view(src_shape));

    ppl::sync();

    dim4 output_offset = {0, 0, 0,
                          batch_slice_outer * batch_idx * input_w + offset};
    ppl::sdma::move(output_gtensor.sub_view(core_shape, output_offset),
                    src_gtensor.sub_view(core_shape, core_offset));
    batch_idx++;
  }
}