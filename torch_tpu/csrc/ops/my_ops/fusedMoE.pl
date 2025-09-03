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

template <typename TYPE>
void gather_load_sparse_input_v3(tensor<TYPE> &out_gather_tensor,
                                 gtensor<TYPE> &input_gtensor,
                                 gtensor<uint32> &index_gtensor,
                                 int gather_batch, int input_batch,
                                 int input_w) {
  dim4 param_shape = {1, gather_batch, input_batch, input_w};
  dim4 index_shape = {1, gather_batch, 1, 1};
  dim4 output_shape = {1, gather_batch, 1, input_w};

  dim4 param_stride = {1, 0, input_w, 1};
  dim4 index_stride = {1, 1, 1, 1};
  dim4 output_stride = {1, input_w, input_w, 1};

  ppl::dma::gather_h(out_gather_tensor.view(output_shape, output_stride),
                     input_gtensor.view(param_shape, param_stride),
                     index_gtensor.view(index_shape, index_stride), 0);
}

template <typename TYPE>
void scatter_store_sparse_output_v3(gtensor<TYPE> &out_scatter_gtensor,
                                    tensor<TYPE> &input_tensor,
                                    gtensor<uint32> &index_gtensor,
                                    int scatter_batch, int batch, int input_w,
                                    int num_exper_per_topk) {
  assert(batch * num_exper_per_topk <= 65535);

  dim4 output_shape = {1, scatter_batch, batch * num_exper_per_topk, input_w};
  dim4 index_shape = {1, scatter_batch, 1, 1};
  dim4 input_shape = {1, scatter_batch, 1, input_w};

  dim4 output_stride = {1, 0, input_w, 1};
  dim4 index_stride = {1, 1, 1, 1};
  dim4 input_stride = {1, input_w, input_w, 1};
  ppl::dma::scatter_h(out_scatter_gtensor.view(output_shape, output_stride),
                      input_tensor.view(input_shape, input_stride),
                      index_gtensor.view(index_shape, index_stride));
}

template <typename TYPE>
void scatter_store_sparse_output_v3_sdma(gtensor<TYPE> &out_scatter_gtensor,
                                         tensor<TYPE> &input_tensor,
                                         gtensor<uint32> &index_gtensor,
                                         gtensor<TYPE> &buffer_gtensor,
                                         int scatter_batch, int batch,
                                         int input_w, int num_exper_per_topk) {
  //   assert(batch * num_exper_per_topk > 65535);

  dim4 output_shape = {1, scatter_batch, batch * num_exper_per_topk, input_w};
  dim4 index_shape = {1, scatter_batch, 1, 1};
  dim4 input_shape = {1, scatter_batch, 1, input_w};

  dim4 output_stride = {1, 0, input_w, 1};
  dim4 index_stride = {1, 1, 1, 1};
  dim4 input_stride = {1, input_w, input_w, 1};

  ppl::dma::store(buffer_gtensor, input_tensor);
  ppl::sdma::scatter_h(out_scatter_gtensor.view(output_shape, output_stride),
                       buffer_gtensor.view(input_shape, input_stride),
                       index_gtensor.view(index_shape, index_stride));
}

template <typename TYPE, typename W_TYPE>
void parallel_expert_split_batch(
    TYPE *output_addr, TYPE *input_addr, W_TYPE *gate_weight_addr,
    W_TYPE *up_weight_addr, W_TYPE *down_weight_addr, TYPE *gate_scale_addr,
    TYPE *up_scale_addr, TYPE *down_scale_addr, uint32 *gather_index,
    uint32 *scatter_index, gtensor<uint> &num_per_expert,
    /*tensor<int8> &token_mask_expert,*/ const int num_experts,
    const int batch_slice_mem, const int block_size, const int batch,
    const int input_w, const int middle_w, const int num_experts_per_topk) {

  int64 num_token_expert = get_gmem_addr(num_per_expert);
  //   int *num_token_expert = (int *)get_gmem_addr(num_per_expert);
  invalid_cache(num_per_expert, num_experts);

  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();

  int scale_n = div_up(input_w, block_size);
  int scale_m = div_up(middle_w, block_size);
  dim4 input_shape = {1, batch_slice_mem, 1, input_w};
  dim4 weight_shape = {1, middle_w, 1, input_w};
  dim4 scale_shape = {scale_m, min(middle_w, NPU_NUM), 1, scale_n};
  dim4 scale_gshape = {scale_m, 1, 1, scale_n};
  dim4 buffer1_shape = {1, middle_w, 1, align(input_w, block_size)};
  dim4 buffer2_shape = {1, batch_slice_mem, 1, middle_w};
  dim4 gather_shape = {1, batch_slice_mem, 1, 1};

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

  dim4 weight_gl_shape = {num_experts, middle_w, 1, input_w};
  dim4 scale_gl_shape = {num_experts, scale_m, 1, scale_n};
  dim4 scale_iter_shape = {1, scale_m, 1, scale_n};

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
  dim4 output_gshape = {1, 1, batch * num_experts_per_topk, input_w};
  dim4 gather_gshape = {num_experts, 1, 1, batch};
  dim4 gather_single_gshape = {1, 1, 1, batch};
  auto input_gtensor = gtensor<TYPE>(input_gshape, GLOBAL, input_addr);
  auto output_gtensor = gtensor<TYPE>(output_gshape, GLOBAL, output_addr);
  auto gather_gtensor = gtensor<uint32>(gather_gshape, GLOBAL, gather_index);
  auto scatter_gtensor = gtensor<uint32>(gather_gshape, GLOBAL, scatter_index);

  dim4 offset_zero = {0, 0, 0, 0};
  int experts[3] = {0};
  int dp_thr = 2 * NPU_NUM * core_num;
  int experts_used = 0;
  int is_first_expert = 1;
  for (int e = 0; e < num_experts; e++) {
    uint token_num = get_value<uint>(num_token_expert + sizeof(uint) * e);
    if (token_num > 0)
      experts_used += 1;
    bool dp_para = token_num >= dp_thr;
    if (dp_para ||
        (token_num > 0 && (experts_used - 1) % core_num == core_idx)) {
      int expert_idx = e;
      int batch_core = dp_para ? div_up(token_num, core_num) : token_num;
      int batch_offset = dp_para ? batch_core * core_idx : 0;
      batch_core =
          dp_para ? min(token_num - batch_offset, batch_core) : token_num;
      if (batch_core <= 0)
        continue;
      int batch_slice = min(batch_slice_mem, batch_core);
      int slice_num = div_up(batch_core, batch_slice);
      int idx = 0;
      dim4 cur_offset = {expert_idx, 0, 0, 0};
      auto cur_gather_gtensor =
          gather_gtensor.sub_view(gather_single_gshape, cur_offset);
      while (idx < slice_num + 2) {
        ppl::parallel_start();
        if (idx > 1) {
          int load_batch =
              min(batch_core - batch_slice * (idx - 2), batch_slice);
          dim4 scatter_real_shape = {1, 1, 1, load_batch};
          dim4 output_real_shape = {1, load_batch, 1, input_w};
          dim4 store_offset = {expert_idx, 0, 0,
                               batch_offset + batch_slice * (idx - 2)};
          scatter_store_sparse_output_v3(
              output_gtensor,
              outputkk_tensor.sub_view(output_real_shape, offset_zero),
              scatter_gtensor.sub_view(scatter_real_shape, store_offset),
              load_batch, batch, input_w, num_experts_per_topk);
        }
        if (idx > 0 && idx < slice_num + 1) {
          // calc and load weight1
          if (idx == 1) {
            ppl::dma::load(weight1_tensor,
                           weight1_gtensor.sub_view(weight_shape, cur_offset));
            ppl::dma::load(
                scale1_tensor.sub_view(scale_gshape, offset_zero),
                scale1_gtensor.sub_view(scale_iter_shape, cur_offset));
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
            ppl::dma::load(weight2_tensor,
                           weight2_gtensor.sub_view(weight_shape, cur_offset));
            ppl::dma::load(
                scale2_tensor.sub_view(scale_gshape, offset_zero),
                scale2_gtensor.sub_view(scale_iter_shape, cur_offset));
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

        if (is_first_expert && idx == 0) {
          // load weight0
          ppl::dma::load(weight0_tensor,
                         weight0_gtensor.sub_view(weight_shape, cur_offset));
          ppl::dma::load(scale0_tensor.sub_view(scale_gshape, offset_zero),
                         scale0_gtensor.sub_view(scale_iter_shape, cur_offset));
          // load input
          int load_batch = min(batch_core - batch_slice * idx, batch_slice);
          dim4 gather_offset = {0, 0, 0, batch_offset + batch_slice * idx};
          dim4 gather_slice_shape = {1, 1, 1, load_batch};
          gather_load_sparse_input_v3(
              input_tensor, input_gtensor,
              cur_gather_gtensor.sub_view(gather_slice_shape, gather_offset),
              load_batch, batch, input_w);
        }
        if (idx && idx < slice_num) {
          // load input
          int load_batch = min(batch_core - batch_slice * idx, batch_slice);
          dim4 gather_offset = {0, 0, 0, batch_offset + batch_slice * idx};
          dim4 gather_slice_shape = {1, 1, 1, load_batch};
          gather_load_sparse_input_v3(
              input_tensor, input_gtensor,
              cur_gather_gtensor.sub_view(gather_slice_shape, gather_offset),
              load_batch, batch, input_w);
        }

        if (idx == slice_num) {
          // load weight0,input of next expert
          int ne = e + 1;
          bool getNext = ne < num_experts;
          while (getNext && ne < num_experts) {
            int n_token_num =
                get_value<int>(num_token_expert + sizeof(int32) * ne);
            if (n_token_num > 0)
              experts_used += 1;
            if (n_token_num >= dp_thr ||
                n_token_num > 0 &&
                    ((experts_used - 1) % core_num == core_idx)) {
              int n_dp_para = n_token_num >= dp_thr;
              int n_batch_core =
                  n_dp_para ? div_up(n_token_num, core_num) : n_token_num;
              int n_batch_offset = n_dp_para ? n_batch_core * core_idx : 0;
              n_batch_core =
                  n_dp_para
                      ? min(n_token_num - n_batch_core * core_idx, n_batch_core)
                      : n_token_num;
              if (n_batch_core > 0) {
                dim4 n_offset = {ne, 0, 0, 0};
                auto n_gather_gtensor =
                    gather_gtensor.sub_view(gather_single_gshape, n_offset);

                ppl::dma::load(weight0_tensor, weight0_gtensor.sub_view(
                                                   weight_shape, n_offset));
                ppl::dma::load(
                    scale0_tensor.sub_view(scale_gshape, offset_zero),
                    scale0_gtensor.sub_view(scale_iter_shape, n_offset));

                int load_batch = min(n_batch_core, batch_slice_mem);
                dim4 gather_slice_shape = {1, 1, 1, load_batch};
                dim4 gather_offset = {0, 0, 0, n_batch_offset};
                gather_load_sparse_input_v3(
                    input_tensor, input_gtensor,
                    n_gather_gtensor.sub_view(gather_slice_shape,
                                              gather_offset),
                    load_batch, batch, input_w);
                e = ne - 1;
                getNext = false;
                experts_used -= 1;
              }
            }
            ne++;
          }
          if (getNext) {
            e = ne - 1;
          }
        }
        ppl::parallel_end();
        idx++;
      }
      if (is_first_expert)
        is_first_expert = 0;
    }
  }
}

template <typename TYPE, typename W_TYPE>
void parallel_expert_split_batch_sdma(
    TYPE *output_addr, TYPE *input_addr, W_TYPE *gate_weight_addr,
    W_TYPE *up_weight_addr, W_TYPE *down_weight_addr, TYPE *gate_scale_addr,
    TYPE *up_scale_addr, TYPE *down_scale_addr, uint32 *gather_index,
    uint32 *scatter_index, gtensor<uint> &num_per_expert,
    /*tensor<int8> &token_mask_expert,*/ const int num_experts,
    const int batch_slice_mem, const int block_size, const int batch,
    const int input_w, const int middle_w, const int num_experts_per_topk) {
  int64 num_token_expert = get_gmem_addr(num_per_expert);
  //   int *num_token_expert = (int *)get_gmem_addr(num_per_expert);
  invalid_cache(num_per_expert, num_experts);

  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();

  int scale_n = div_up(input_w, block_size);
  int scale_m = div_up(middle_w, block_size);
  dim4 input_shape = {1, batch_slice_mem, 1, input_w};
  dim4 weight_shape = {1, middle_w, 1, input_w};
  dim4 scale_shape = {scale_m, min(middle_w, NPU_NUM), 1, scale_n};
  dim4 scale_gshape = {scale_m, 1, 1, scale_n};
  dim4 buffer1_shape = {1, middle_w, 1, align(input_w, block_size)};
  dim4 buffer2_shape = {1, batch_slice_mem, 1, middle_w};
  dim4 gather_shape = {1, batch_slice_mem, 1, 1};

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

  dim4 weight_gl_shape = {num_experts, middle_w, 1, input_w};
  dim4 scale_gl_shape = {num_experts, scale_m, 1, scale_n};
  dim4 scale_iter_shape = {1, scale_m, 1, scale_n};

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
  dim4 output_gshape = {1, 1, batch * num_experts_per_topk, input_w};
  dim4 gather_gshape = {num_experts, 1, 1, batch};
  dim4 gather_single_gshape = {1, 1, 1, batch};
  auto input_gtensor = gtensor<TYPE>(input_gshape, GLOBAL, input_addr);
  auto output_gtensor = gtensor<TYPE>(output_gshape, GLOBAL, output_addr);
  auto gather_gtensor = gtensor<uint32>(gather_gshape, GLOBAL, gather_index);
  auto scatter_gtensor = gtensor<uint32>(gather_gshape, GLOBAL, scatter_index);

  dim4 output_muti_shape = {core_num, batch_slice_mem, 1, input_w};
  dim4 output_muti_offset = {core_idx, 0, 0, 0};
  //   auto output_muti_l2 = gtensor<TYPE>(
  //       output_muti_shape, L2, num_token_expert + num_experts * sizeof(int));
  auto output_muti_l2 = gtensor<TYPE>(output_muti_shape, L2);
  auto output_l2 = output_muti_l2.sub_view(input_shape, output_muti_offset);

  dim4 offset_zero = {0, 0, 0, 0};
  int experts[3] = {0};
  int dp_thr = 2 * NPU_NUM * core_num;
  int experts_used = 0;
  int is_first_expert = 1;

  for (int e = 0; e < num_experts; e++) {
    uint token_num = get_value<uint>(num_token_expert + sizeof(uint) * e);
    if (token_num > 0)
      experts_used += 1;
    bool dp_para = token_num >= dp_thr;
    if (dp_para ||
        (token_num > 0 && (experts_used - 1) % core_num == core_idx)) {
      int expert_idx = e;
      int batch_core = dp_para ? div_up(token_num, core_num) : token_num;
      int batch_offset = dp_para ? batch_core * core_idx : 0;
      batch_core =
          dp_para ? min(token_num - batch_offset, batch_core) : token_num;
      if (batch_core <= 0)
        continue;
      int batch_slice = min(batch_slice_mem, batch_core);
      int slice_num = div_up(batch_core, batch_slice);
      int idx = 0;
      dim4 cur_offset = {expert_idx, 0, 0, 0};
      auto cur_gather_gtensor =
          gather_gtensor.sub_view(gather_single_gshape, cur_offset);
      while (idx < slice_num + 2) {
        ppl::parallel_start();
        if (idx > 1) {
          int load_batch =
              min(batch_core - batch_slice * (idx - 2), batch_slice);
          dim4 scatter_real_shape = {1, 1, 1, load_batch};
          dim4 output_real_shape = {1, load_batch, 1, input_w};
          dim4 store_offset = {expert_idx, 0, 0,
                               batch_offset + batch_slice * (idx - 2)};
          scatter_store_sparse_output_v3_sdma(
              output_gtensor,
              outputkk_tensor.sub_view(output_real_shape, offset_zero),
              scatter_gtensor.sub_view(scatter_real_shape, store_offset),
              output_l2.sub_view(output_real_shape, offset_zero), load_batch,
              batch, input_w, num_experts_per_topk);
        }
        if (idx > 0 && idx < slice_num + 1) {
          // calc and load weight1
          if (idx == 1) {
            ppl::dma::load(weight1_tensor,
                           weight1_gtensor.sub_view(weight_shape, cur_offset));
            ppl::dma::load(
                scale1_tensor.sub_view(scale_gshape, offset_zero),
                scale1_gtensor.sub_view(scale_iter_shape, cur_offset));
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
            ppl::dma::load(weight2_tensor,
                           weight2_gtensor.sub_view(weight_shape, cur_offset));
            ppl::dma::load(
                scale2_tensor.sub_view(scale_gshape, offset_zero),
                scale2_gtensor.sub_view(scale_iter_shape, cur_offset));
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

        if (is_first_expert && idx == 0) {
          // load weight0
          ppl::dma::load(weight0_tensor,
                         weight0_gtensor.sub_view(weight_shape, cur_offset));
          ppl::dma::load(scale0_tensor.sub_view(scale_gshape, offset_zero),
                         scale0_gtensor.sub_view(scale_iter_shape, cur_offset));
          // load input
          int load_batch = min(batch_core - batch_slice * idx, batch_slice);
          dim4 gather_offset = {0, 0, 0, batch_offset + batch_slice * idx};
          dim4 gather_slice_shape = {1, 1, 1, load_batch};
          gather_load_sparse_input_v3(
              input_tensor, input_gtensor,
              cur_gather_gtensor.sub_view(gather_slice_shape, gather_offset),
              load_batch, batch, input_w);
        }
        if (idx && idx < slice_num) {
          // load input
          int load_batch = min(batch_core - batch_slice * idx, batch_slice);
          dim4 gather_offset = {0, 0, 0, batch_offset + batch_slice * idx};
          dim4 gather_slice_shape = {1, 1, 1, load_batch};
          gather_load_sparse_input_v3(
              input_tensor, input_gtensor,
              cur_gather_gtensor.sub_view(gather_slice_shape, gather_offset),
              load_batch, batch, input_w);
        }

        if (idx == slice_num) {
          // load weight0,input of next expert
          int ne = e + 1;
          bool getNext = ne < num_experts;
          while (getNext && ne < num_experts) {
            int n_token_num =
                get_value<int>(num_token_expert + sizeof(int32) * ne);
            if (n_token_num > 0)
              experts_used += 1;
            if (n_token_num >= dp_thr ||
                n_token_num > 0 &&
                    ((experts_used - 1) % core_num == core_idx)) {
              int n_dp_para = n_token_num >= dp_thr;
              int n_batch_core =
                  n_dp_para ? div_up(n_token_num, core_num) : n_token_num;
              int n_batch_offset = n_dp_para ? n_batch_core * core_idx : 0;
              n_batch_core =
                  n_dp_para
                      ? min(n_token_num - n_batch_core * core_idx, n_batch_core)
                      : n_token_num;
              if (n_batch_core > 0) {
                dim4 n_offset = {ne, 0, 0, 0};
                auto n_gather_gtensor =
                    gather_gtensor.sub_view(gather_single_gshape, n_offset);

                ppl::dma::load(weight0_tensor, weight0_gtensor.sub_view(
                                                   weight_shape, n_offset));
                ppl::dma::load(
                    scale0_tensor.sub_view(scale_gshape, offset_zero),
                    scale0_gtensor.sub_view(scale_iter_shape, n_offset));

                int load_batch = min(n_batch_core, batch_slice_mem);
                dim4 gather_slice_shape = {1, 1, 1, load_batch};
                dim4 gather_offset = {0, 0, 0, n_batch_offset};
                gather_load_sparse_input_v3(
                    input_tensor, input_gtensor,
                    n_gather_gtensor.sub_view(gather_slice_shape,
                                              gather_offset),
                    load_batch, batch, input_w);
                e = ne - 1;
                getNext = false;
                experts_used -= 1;
              }
            }
            ne++;
          }
          if (getNext) {
            e = ne - 1;
          }
        }
        ppl::parallel_end();
        idx++;
      }
      if (is_first_expert)
        is_first_expert = 0;
    }
  }
}

template <typename TYPE, typename W_TYPE>
void parallel_expert_multi_core(
    TYPE *output_addr, TYPE *input_addr, W_TYPE *gate_weight_addr,
    W_TYPE *up_weight_addr, W_TYPE *down_weight_addr, TYPE *gate_scale_addr,
    TYPE *up_scale_addr, TYPE *down_scale_addr, int32_t *select_experts_addr,
    bf16 *routing_weights_addr, uint32 *gather_index, uint32 *scatter_index,
    int8 *gather_buffer, int8 *scatter_buffer, const int blocksize,
    const int num_experts, const int num_experts_per_topk, const int batch,
    const int batch_slice, const int index_batch_, const int input_w,
    const int middle_w, const int quantized) {
  const int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();

  int num_experts_core = div_up(num_experts, core_num);
  int index_batch = index_batch_;

  dim4 expert_shape = {1, 1, 1, num_experts};
  dim4 select_expert_shape = {1, batch, num_experts_per_topk, 1};
  dim4 select_slice_shape = {1, index_batch, num_experts_per_topk, 1};
  dim4 mask_shape = {1, index_batch, num_experts_per_topk, num_experts};
  dim4 token_mask_shape = {1, index_batch, 1, num_experts};
  dim4 token_trans_shape = {1, num_experts, 1, index_batch};
  dim4 scatter_mid_shape = {1, num_experts, num_experts_per_topk, index_batch};

  dim4 token_trans_gshape = {1, num_experts, 1, batch};
  dim4 scatter_mask_shape = {1, num_experts, batch, num_experts_per_topk};

  auto select_expert_gtensor =
      gtensor<int32_t>(select_expert_shape, GLOBAL, select_experts_addr);
  auto gather_index_gtensor =
      gtensor<uint32>(token_trans_gshape, GLOBAL, gather_index);
  auto num_per_expert = make_l2tensor<uint>(expert_shape, L2, expert_shape);

  //   auto num_per_expert = gtensor<uint>(expert_shape, L2);
  auto scatter_index_gtensor =
      gtensor<uint32>(token_trans_gshape, GLOBAL, scatter_index);
  auto gather_buffer_gtensor =
      gtensor<int8>(token_trans_gshape, GLOBAL, gather_buffer);
  auto scatter_buffer_gtensor =
      gtensor<int8>(scatter_mask_shape, GLOBAL, scatter_buffer);

  auto experts_idx = make_tensor<int32_t>(expert_shape, expert_shape);
  auto select_expert_tensor =
      make_tensor<int32_t>(select_expert_shape, select_expert_shape);
  auto mask_tensor = make_tensor<int8>(mask_shape, mask_shape);
  auto token_mask_tensor =
      make_tensor<int8>(token_mask_shape, token_mask_shape);
  auto token_trans_tensor =
      make_tensor<int8>(token_trans_shape, token_trans_shape);
  auto scatter_mid_tensor =
      make_tensor<int8>(scatter_mid_shape, scatter_mid_shape);

  dim2 kernel = {num_experts_per_topk, 1};
  dim2 stride = {1, 1};
  padding_t pad = {0, 0, 0, 0};
  dim2 dilation = {1, 1};

  int batch_core = div_up(batch, core_num);
  int batch_offset = batch_core * core_idx;
  batch_core = min(batch_core, batch - batch_offset);
  index_batch = min(index_batch_, batch_core);
  int iters = div_up(batch_core, index_batch_);

  arange_broadcast(experts_idx, 1, 0, 1, num_experts);

  if (batch_core > 0) {
    for (int i = 0; i < iters; ++i) {
      dim4 load_offset = {0, batch_offset + index_batch * i, 0, 0};
      int load_batch = min(index_batch, batch_core - index_batch * i);
      dim4 load_shape = {1, load_batch, num_experts_per_topk, 1};
      ppl::dma::load(select_expert_tensor.view(load_shape),
                     select_expert_gtensor.sub_view(load_shape, load_offset));

      int calc_batch = min(index_batch, batch_core - index_batch * i);
      dim4 mask_real_shape = {1, calc_batch, num_experts_per_topk, num_experts};
      dim4 select_real_shape = {1, calc_batch, num_experts_per_topk, 1};
      dim4 token_mask_rshape = {1, calc_batch, 1, num_experts};
      dim4 token_trans_rshape = {1, num_experts, 1, calc_batch};
      dim4 scatter_mid_rshape = {1, num_experts, num_experts_per_topk,
                                 calc_batch};
      ppl::tiu::eq(mask_tensor.view(mask_real_shape),
                   select_expert_tensor.view(select_real_shape), experts_idx,
                   1);

      ppl::tiu::pool_max(token_mask_tensor.view(token_mask_rshape),
                         mask_tensor.view(mask_real_shape), &kernel, &pad,
                         &stride, &dilation);
      ppl::tiu::transpose_wc(token_trans_tensor.view(token_trans_rshape),
                             token_mask_tensor.view(token_mask_rshape));
      ppl::tiu::transpose_wc(scatter_mid_tensor.view(scatter_mid_rshape),
                             mask_tensor.view(mask_real_shape));

      int store_batch = min(index_batch, batch_core - index_batch * i);
      dim4 store_gather_shape = {1, num_experts, 1, store_batch};
      dim4 store_scatter_shape = {1, num_experts, store_batch,
                                  num_experts_per_topk};
      dim4 gather_offset = {0, 0, 0, batch_offset + index_batch * i};
      dim4 scatter_offset = {0, 0, batch_offset + index_batch * i, 0};

      dim4 scatter_src_stride = {
          1, align(store_batch * num_experts_per_topk, get_eu_num<int8>()), 1,
          store_batch};

      ppl::dma::store(
          gather_buffer_gtensor.sub_view(store_gather_shape, gather_offset),
          token_trans_tensor.view(store_gather_shape));
      ppl::dma::store(
          scatter_buffer_gtensor.sub_view(store_scatter_shape, scatter_offset),
          scatter_mid_tensor.view(store_scatter_shape, scatter_src_stride));
    }
  }
  ppl::sync_all();

  dim4 nonzero_shape = {1, 1, 1, batch};
  dim4 st_shape = {1, 1, 1, 1};
  dim4 scatter_single_shape = {1, 1, batch, num_experts_per_topk};
  for (int i = num_experts_core * core_idx;
       i < min(num_experts_core * (core_idx + 1), num_experts); ++i) {
    dim4 cur_offset = {0, i, 0, 0};
    dim4 st_offset = {0, 0, 0, i};
    // input_gather_index
    uint tokens = ppl::dma::nonzero(
        gather_index_gtensor.sub_view(nonzero_shape, cur_offset),
        gather_buffer_gtensor.sub_view(nonzero_shape, cur_offset));
    ppl::dma::fill(num_per_expert.sub_view(st_shape, st_offset), tokens);

    // scatter_index
    uint tokens_ = ppl::dma::nonzero(
        scatter_index_gtensor.sub_view(nonzero_shape, cur_offset),
        scatter_buffer_gtensor.sub_view(scatter_single_shape, cur_offset));
    assert(tokens == tokens_);
  }
  ppl::sync_all();

  if (num_experts_per_topk * batch > 65535) {
    parallel_expert_split_batch_sdma(
        output_addr, input_addr, gate_weight_addr, up_weight_addr,
        down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
        gather_index, scatter_index, num_per_expert, num_experts, batch_slice,
        blocksize, batch, input_w, middle_w, num_experts_per_topk);
  } else {
    parallel_expert_split_batch(
        output_addr, input_addr, gate_weight_addr, up_weight_addr,
        down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
        gather_index, scatter_index, num_per_expert, num_experts, batch_slice,
        blocksize, batch, input_w, middle_w, num_experts_per_topk);
  }
}

template <typename TYPE, typename W_TYPE>
void moe_multi_core(TYPE *output_addr, TYPE *input_addr,
                    W_TYPE *gate_weight_addr, W_TYPE *up_weight_addr,
                    W_TYPE *down_weight_addr, TYPE *gate_scale_addr,
                    TYPE *up_scale_addr, TYPE *down_scale_addr,
                    int32_t *select_experts_addr, bf16 *routing_weights_addr,
                    uint32 *gather_index, uint32 *scatter_index,
                    int8 *gather_buffer, int8 *scatter_buffer,
                    const int blocksize, const int num_expert,
                    const int num_expert_per_topk, const int batch,
                    const int batch_slice, const int index_batch,
                    const int input_w, const int middle_w, int quantized) {
  ppl::set_core_num(CORE_NUM);

  parallel_expert_multi_core<TYPE, W_TYPE>(
      output_addr, input_addr, gate_weight_addr, up_weight_addr,
      down_weight_addr, gate_scale_addr, up_scale_addr, down_scale_addr,
      select_experts_addr, routing_weights_addr, gather_index, scatter_index,
      gather_buffer, scatter_buffer, blocksize, num_expert, num_expert_per_topk,
      batch, batch_slice, index_batch, input_w, middle_w, quantized);
}

__KERNEL__ void moe_multi_core_bf16_fp8e4m3(
    bf16 *output_addr, bf16 *input_addr, fp8e4m3 *gate_weight_addr,
    fp8e4m3 *up_weight_addr, fp8e4m3 *down_weight_addr, bf16 *gate_scale_addr,
    bf16 *up_scale_addr, bf16 *down_scale_addr, int32_t *select_experts_addr,
    bf16 *routing_weights_addr, uint32 *gather_index, uint32 *scatter_index,
    int8 *gather_buffer, int8 *scatter_buffer, const int blocksize,
    const int num_expert, const int num_expert_per_topk, const int batch,
    const int batch_slice, const int index_batch, const int input_w,
    const int middle_w, int quantized) {
  moe_multi_core(output_addr, input_addr, gate_weight_addr, up_weight_addr,
                 down_weight_addr, gate_scale_addr, up_scale_addr,
                 down_scale_addr, select_experts_addr, routing_weights_addr,
                 gather_index, scatter_index, gather_buffer, scatter_buffer,
                 blocksize, num_expert, num_expert_per_topk, batch, batch_slice,
                 index_batch, input_w, middle_w, quantized);
}
