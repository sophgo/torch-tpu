#include "ppl.h"
#include "ppl_wrapper_func.h"
#include <vector>

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void topk_kernel_blocked(
        uint32_t* ptr_index_out,
        const T* ptr_input,
        T* ptr_value_out,
        int outer_size,
        int axis_size,
        int inner_size,
        int k,
        bool largest,
        const int block_size
        ) {
// if axis_size is too large (>64), we need two stages to deal with it.
// 1. for (1,...,iter_w) topk(1, C, 1, 64)=(1, C, 1, k)
// 2. topk(1, C, 1, k*iter_w)=(1, C, 1, k)

    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    int C = outer_size;
    int W = axis_size;

    dim4 in_gshape = {1, C, 1, W};
    dim4 out_gshape = {1, C, 1, k};

    auto in_g  = gtensor<T>(in_gshape, GLOBAL, const_cast<T*>(ptr_input));
    auto outv_g = gtensor<T>(out_gshape, GLOBAL, ptr_value_out);
    auto outi_g = gtensor<uint32_t>(out_gshape, GLOBAL, ptr_index_out);

    int block_c = LANE_NUM;

    // const int block_size = 64;
    int iter_w = W / block_size;
    dim4 block_shape = {1, block_c, block_size, block_size};
    dim4 block_k_shape = {1, block_c, block_size, k};
    dim4 block_topk_shape = {1, block_c, 1, k*iter_w};
    dim4 block_x_shape = {1, block_c, 1, block_size};

    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        padding_t pad = {0, 0, 0, 0};
        dim2 dil = {1,1}, ins = {0,0};
        dim2 stride = {1, 1};
        dim4 pool_buffer_shape_block = {1, block_c, block_size, 1};
        dim4 scatter_out_shape_block = {1, block_c, 1, block_size};
        auto outv_block = make_tensor<fp32>(block_topk_shape, block_topk_shape);
        auto outi_block = make_tensor<uint32_t>(block_topk_shape, block_topk_shape);
        dim4 in_lshape_block = {1, block_c, 1, block_size};
        dim4 out_lshape = {1, c, 1, k};
        dim4 in_offset_ = {0,core_offset+c_idx,0,0};

        for (auto w_idx = 0; w_idx < W; w_idx += block_size) {
            enable_pipeline();
            int w = min(block_size, W - w_idx);
            dim2 kernel = {1, w};
            dim4 pool_buffer_shape = {1, c, w, 1};
            dim4 scatter_out_shape = {1, c, 1, w};
            dim4 scatter_index_shape = {1, c, 1, w};
            dim4 in_lshape = {1, c, 1, w};
            dim4 in_offset = {0,core_offset+c_idx,0,w_idx};
            auto x_fp = make_tensor<T>(in_lshape_block, in_lshape);
            dma::load(x_fp, in_g.sub_view(in_lshape, in_offset));
            auto x_fp32 = make_tensor<fp32>(in_lshape_block, in_lshape);
            tiu::cast(x_fp32, x_fp);

    // 1.1) input + tiny_bias*i to avoid the same input value
            auto idx_u32 = make_tensor<uint32_t>(in_lshape_block, in_lshape);
            arange_broadcast(idx_u32, c, w_idx, 1, w);

            auto idx_fp32 = make_tensor<fp32>(in_lshape_block, in_lshape);
            tiu::cast(idx_fp32, idx_u32);

            // bias = idx * tiny_bias
            auto bias = make_tensor<fp32>(in_lshape_block, in_lshape);

            float tiny_bias;
            if (std::is_same<T, fp16>::value || std::is_same<T, bf16>::value) {
                tiny_bias = 1e-4f;
            } else {
                tiny_bias = 1e-6f;
            }

            // for largest: later index should be larger -> add +tiny_bias*idx
            // for smallest: later index should be considered smaller -> add -tiny_bias*idx
            float signed_tiny = largest ? tiny_bias : -tiny_bias;
            tiu::fmul(bias, idx_fp32, signed_tiny);

            // x_fp32 = x_fp32 + signed tiny bias
            auto x = make_tensor<fp32>(in_lshape_block, in_lshape);
            tiu::fadd(x, x_fp32, bias);

    // 1.2) mask via gt_select
            dim4 actual_mask_shape = {1, block_c, w, w};
            auto ones_block = make_tensor<fp32>(block_shape, actual_mask_shape);
            auto zeros_block = make_tensor<fp32>(block_shape, actual_mask_shape);
            tiu::fill(ones_block, 1.0f);
            tiu::fill(zeros_block, 0.0f);
            auto Xcol = make_tensor<fp32>(block_shape, actual_mask_shape);
            auto Xrow = make_tensor<fp32>(block_shape, actual_mask_shape);
            tiu::fmul(Xcol, ones_block, x); // x: [1,1,1,block_size]

            dim4 x_trans_shape = {1, c, w, 1};
            dim4 x_trans_stride = {c * w, w, 1, w};
            auto x_row_view = x.view(x_trans_shape, x_trans_stride);
            tiu::fmul(Xrow, ones_block, x_row_view);

            auto mask = make_tensor<fp32>(block_shape, actual_mask_shape);
            if (largest) {
                tiu::gt_select(mask, Xcol, Xrow, ones_block, zeros_block);
            } else {
                tiu::lt_select(mask, Xcol, Xrow, ones_block, zeros_block);
            }

    // 1.3) avgpool
            auto pool_buffer = make_tensor<fp32>(pool_buffer_shape_block, pool_buffer_shape);
            tiu::fpool_avg(pool_buffer, mask, &kernel, &pad, &stride, &dil, &ins, 1.0f);

            auto pool_int =  make_tensor<uint16_t>(pool_buffer_shape_block, pool_buffer_shape);
            tiu::cast(pool_int, pool_buffer, RM_HALF_AWAY_FROM_ZERO);

    // 1.4) scatter

            auto scatter_out = make_tensor<fp32>(scatter_out_shape_block, scatter_out_shape); // {1,C,1,W}

            tiu::scatter_w(scatter_out, x, pool_int.view(scatter_index_shape), 0);// x -> {1,C,1,W} ; scatter_index_w -> {1,W,1,1}

            auto scatter_index_index_out = make_tensor<uint32_t>(scatter_out_shape_block, scatter_out_shape);
            tiu::scatter_w(scatter_index_index_out, idx_u32, pool_int.view(scatter_index_shape), 0);
            dim4 outk_offset = {0, 0, 0, w_idx*k/block_size};

            tiu::move(outv_block.sub_view(out_lshape, outk_offset), scatter_out.view(out_lshape));
            tiu::move(outi_block.sub_view(out_lshape, outk_offset), scatter_index_index_out.view(out_lshape));
        }

    if (iter_w == 1){  // one stage
        auto outv_block_T = make_tensor<T>(block_topk_shape, block_topk_shape);
        tiu::cast(outv_block_T, outv_block);
        dma::store(outv_g.sub_view(out_lshape, in_offset_), outv_block_T.view(out_lshape));
        dma::store(outi_g.sub_view(out_lshape, in_offset_), outi_block.view(out_lshape));
    } else {           // two stage (1,1,1,k*iter_w) -> (1,1,1,k)

    // 2.1) mask via gt_select
        dim4 mask_shape = {1, c, k*iter_w, k*iter_w};
        dim4 mask_shape_block = {1, block_c, k*iter_w, k*iter_w};
        dim4 block_topk_trans_shape = {1, c, k*iter_w, 1};

        auto out_ones_block = make_tensor<fp32>(mask_shape_block, mask_shape);
        auto out_zeros_block = make_tensor<fp32>(mask_shape_block, mask_shape);
        tiu::fill(out_ones_block, 1.0f);
        tiu::fill(out_zeros_block, 0.0f);

        auto out_Xcol = make_tensor<fp32>(mask_shape_block, mask_shape);
        auto out_Xrow = make_tensor<fp32>(mask_shape_block, mask_shape);
        auto Xrow = make_tensor<fp32>(mask_shape_block, mask_shape);
        tiu::fmul(out_Xcol, out_ones_block, outv_block); // x: [1,1,1,k*iter_w]

        dim4 block_topk_trans_stride = {c*k*iter_w, k*iter_w, 1, k * iter_w};
        auto out_x_row_view = outv_block.view(block_topk_trans_shape, block_topk_trans_stride);
        tiu::fmul(out_Xrow, out_ones_block, out_x_row_view);

       auto out_mask = make_tensor<fp32>(mask_shape_block, mask_shape);
       if (largest) {
        tiu::gt_select(out_mask, out_Xcol, out_Xrow, out_ones_block, out_zeros_block);
       } else {
        tiu::lt_select(out_mask, out_Xcol, out_Xrow, out_ones_block, out_zeros_block);
       }

    //  2.2) avgpool
       dim4 out_pool_buffer_shape = {1, c, k*iter_w, 1};
       dim4 out_pool_buffer_shape_block = {1, block_c, k*iter_w, 1};
       dim2 out_kernel = {1, k*iter_w};

       dim4 out_scatter_index_shape = {1, c, 1, k*iter_w};
       dim4 out_scatter_index_shape_block = {1, block_c, 1, k*iter_w};
       dim4 out_scatter_out_shape = {1, c, 1, k*iter_w};
       dim4 out_scatter_out_shape_block = {1, block_c, 1, k*iter_w};

       auto out_pool_buffer = make_tensor<fp32>(out_pool_buffer_shape_block, out_pool_buffer_shape);
       tiu::fpool_avg(out_pool_buffer, out_mask, &out_kernel, &pad, &stride, &dil, &ins, 1.0f);

       auto out_pool_int =  make_tensor<uint16_t>(out_pool_buffer_shape_block, out_pool_buffer_shape);
       tiu::cast(out_pool_int, out_pool_buffer);

       auto out_scatter_index = make_tensor<uint16_t>(out_scatter_index_shape_block, out_scatter_index_shape);

    // 2.3) scatter
       auto out_scatter_out = make_tensor<fp32>(out_scatter_out_shape_block, out_scatter_out_shape); // {1,C,1,W}

       tiu::scatter_w(out_scatter_out, outv_block, out_pool_int.view(out_scatter_index_shape), 0);

       auto out_scatter_index_index_out = make_tensor<uint32_t>(out_scatter_out_shape_block, out_scatter_out_shape);
       tiu::scatter_w(out_scatter_index_index_out, outi_block, out_pool_int.view(out_scatter_index_shape), 0);

       auto out_scatter_out_T = make_tensor<T>(out_scatter_out_shape_block, out_scatter_out_shape);
       tiu::cast(out_scatter_out_T, out_scatter_out);
       dma::store(outv_g.sub_view(out_lshape, in_offset_), out_scatter_out_T.view(out_lshape));
       dma::store(outi_g.sub_view(out_lshape, in_offset_), out_scatter_index_index_out.view(out_lshape));
    }
  }
}

__KERNEL__ void topk_fp32(uint32_t *ptr_index, fp32 *ptr_input, fp32 *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         int k, bool largest, const int tile_size) {
    topk_kernel_blocked<fp32>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, k, largest, tile_size);

}

__KERNEL__ void topk_fp16(uint32_t *ptr_index, fp16 *ptr_input, fp16 *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         int k, bool largest, const int tile_size) {
    topk_kernel_blocked<fp16>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, k, largest, tile_size);

}

__KERNEL__ void topk_bf16(uint32_t *ptr_index, bf16 *ptr_input, bf16 *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         int k, bool largest, const int tile_size) {
    topk_kernel_blocked<bf16>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, k, largest, tile_size);

}
