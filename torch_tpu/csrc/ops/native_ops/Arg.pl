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
void arg_kernel_blocked(
        uint32_t* ptr_index_out,
        const T* ptr_input,
        T* ptr_value_out,
        int outer_size,
        int axis_size,
        int inner_size,
        bool ismin,
        const int block_w,
        bool select_last_index = false
        ) {
    int C = outer_size;
    int H = axis_size;
    int W = inner_size;

    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(C, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, C - core_offset);

    dim4 in_gshape = {1, C, H, W};
    dim4 out_gshape = {1, C, 1, W};

    auto in_g  = gtensor<T>(in_gshape, GLOBAL, const_cast<T*>(ptr_input));
    auto outv_g = gtensor<T>(out_gshape, GLOBAL, ptr_value_out);
    auto outi_g = gtensor<uint32_t>(out_gshape, GLOBAL, ptr_index_out);

    int block_c = LANE_NUM;
    int block_h = min(256, H);
    int iter_h = (H + block_h - 1) / block_h;

    // Define block shapes
    dim4 in_lshape_block = {1, block_c, block_h, block_w};
    dim4 outv_lshape_block = {1, block_c, 1, block_w};
    dim4 outi_lshape_block = {1, block_c, 1, block_w};
    dim4 idx_shape_block = {1, block_c, block_h, 1};
    dim4 temp_v_shape = {1, block_c, iter_h, W};
    dim4 temp_i_shape = {1, block_c, iter_h, W};

    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        dim2 kernel = {block_h, 1};
        dim2 stride = {1, 1};
        dim2 dilation = {1, 1};
        padding_t pad = {0, 0, 0, 0};

        // Define local shapes
        dim4 outv_shape_ = {1, c, iter_h, W};
        dim4 input_global_offset_ = {0, core_offset + c_idx, 0, 0};

        auto x_maxmin_block = make_tensor<T>(temp_v_shape, outv_shape_);
        auto out_idx_u32_block = make_tensor<uint32_t>(temp_i_shape, outv_shape_);

        for (auto h_idx = 0; h_idx < H; h_idx += block_h) {
            int h = min(block_h, H - h_idx);
            int h_block_idx = h_idx / block_h;

            for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
                enable_pipeline();
                int w = min(block_w, W - w_idx);

                // Define local shapes for this iteration
                dim4 in_lshape = {1, c, h, w};
                dim4 outv_lshape = {1, c, 1, w};
                dim4 outi_lshape = {1, c, 1, w};
                dim4 idx_shape = {1, c, h, 1};

                auto x = make_tensor<T>(in_lshape_block, in_lshape);
                dim4 input_global_offset = {0, core_offset + c_idx, h_idx, w_idx};
                dma::load(x, in_g.sub_view(in_lshape, input_global_offset));

                // argmax/argmin compute
                auto x_maxmin_fp32 = make_tensor<fp32>(outv_lshape_block, outv_lshape);
                auto x_fp32 = make_tensor<fp32>(in_lshape_block, in_lshape);
                tiu::cast(x_fp32, x);

                if (ismin) {
                    tiu::fpool_min(x_maxmin_fp32, x_fp32, &kernel, &pad, &stride, &dilation);
                } else {
                    tiu::fpool_max(x_maxmin_fp32, x_fp32, &kernel, &pad, &stride, &dilation);
                }

                // Store to temporary buffer
                dim4 v_offset = {0, 0, h_block_idx, w_idx};
                auto temp_v_view = x_maxmin_block.sub_view(outv_lshape, v_offset);
                tiu::cast(temp_v_view, x_maxmin_fp32);

                // x equal x_maxmin = mask
                auto mask_eq = make_tensor<int32_t>(in_lshape_block, in_lshape);
                tiu::eq(mask_eq, x, temp_v_view, 1);

                auto local_idx = make_tensor<uint32_t>(idx_shape_block, idx_shape);
                arange_broadcast(local_idx, c, 0, 1, h);

                auto weight = make_tensor<uint32_t>(in_lshape_block, in_lshape);
                if (select_last_index) {
                    // last index: 使用正向权重 (0, 1, 2, ..., h-1)
                    tiu::cast(weight, mask_eq);
                    tiu::mul(weight, weight, local_idx);
                } else {
                    // first index: 使用反向权重 (h-1, h-2, ..., 0)
                    auto reversed_idx = make_tensor<uint32_t>(idx_shape_block, idx_shape);
                    tiu::fill(reversed_idx, static_cast<uint32_t>(h - 1));
                    tiu::sub(reversed_idx, reversed_idx, local_idx);

                    tiu::cast(weight, mask_eq);
                    tiu::mul(weight, weight, reversed_idx);
                }

                auto weight_fp = make_tensor<fp32>(in_lshape_block, in_lshape);
                tiu::cast(weight_fp, weight);
                // reduce to get argmax/argmin index
                auto max_weight_fp = make_tensor<fp32>(outi_lshape_block, outi_lshape);
                tiu::fill(max_weight_fp, 0.0f);
                tiu::fpool_max(max_weight_fp, weight_fp, &kernel, &pad, &stride, &dilation);

                auto max_weight = make_tensor<uint32_t>(outi_lshape_block, outi_lshape);
                tiu::cast(max_weight, max_weight_fp);

                auto selected_local_idx = make_tensor<uint32_t>(outi_lshape_block, outi_lshape);
                if (select_last_index) {
                    tiu::move(selected_local_idx, max_weight);
                } else {
                    // local_idx = h - 1 - weight
                    auto h_minus_1 = make_tensor<uint32_t>(outi_lshape_block, outi_lshape);
                    tiu::fill(h_minus_1, static_cast<uint32_t>(h - 1));
                    tiu::sub(selected_local_idx, h_minus_1, max_weight);
                }
                tiu::add(selected_local_idx, selected_local_idx, h_idx);

                // Store index to temporary buffer
                dim4 i_offset = {0, 0, h_block_idx, w_idx};
                auto temp_i_view = out_idx_u32_block.sub_view(outi_lshape, i_offset);
                tiu::move(temp_i_view, selected_local_idx);
            } // w
        } // h

        if (iter_h == 1) { // one stage done
            dim4 out_final_shape = {1, c, 1, W};
            dma::store(outv_g.sub_view(out_final_shape, input_global_offset_),
                      x_maxmin_block.view(out_final_shape));
            dma::store(outi_g.sub_view(out_final_shape, input_global_offset_),
                      out_idx_u32_block.view(out_final_shape));
        } else { // two stage
            dim2 stage2_kernel = {iter_h, 1};
            dim2 stage2_stride = {1, 1};
            dim2 stage2_dilation = {1, 1};
            padding_t stage2_pad = {0, 0, 0, 0};
            dim4 out_final_shape = {1, c, 1, W};

            auto stage2_v = x_maxmin_block.view(outv_shape_); // {1, c, iter_h, W}
            auto stage2_i = out_idx_u32_block.view(outv_shape_);

            dim4 stage2_full_lshape = {1, block_c, iter_h, W};
            dim4 stage2_out_lshape  = {1, block_c, 1, W};

            // 1. result_v : {1, c, iter_h, W} -> {1, c, 1, W}
            auto stage2_v_fp32 = make_tensor<fp32>(stage2_full_lshape, outv_shape_);
            auto result_v_fp32 = make_tensor<fp32>(stage2_out_lshape, out_final_shape);
            tiu::cast(stage2_v_fp32, stage2_v);

            if (ismin) {
                tiu::fpool_min(result_v_fp32, stage2_v_fp32, &stage2_kernel, &stage2_pad, &stage2_stride, &stage2_dilation);
            } else {
                tiu::fpool_max(result_v_fp32, stage2_v_fp32, &stage2_kernel, &stage2_pad, &stage2_stride, &stage2_dilation);
            }

            auto result_v = make_tensor<T>(stage2_out_lshape, out_final_shape);
            tiu::cast(result_v, result_v_fp32);

            // 2. result_i : find the correct index
            auto mask_eq = make_tensor<int32_t>(stage2_full_lshape, outv_shape_);
            tiu::eq(mask_eq, stage2_v, result_v, 1);

            dim4 stage2_idx_lshape = {1, block_c, iter_h, 1};
            dim4 stage2_idx_shape  = {1, c,     iter_h, 1};
            auto stage2_local_idx = make_tensor<uint32_t>(stage2_idx_lshape, stage2_idx_shape);
            arange_broadcast(stage2_local_idx, c, 0, 1, iter_h);

            auto weight = make_tensor<uint32_t>(stage2_full_lshape, outv_shape_);
            if (select_last_index) {
                // last index: 使用正向权重
                tiu::cast(weight, mask_eq);
                // broadcast stage2_local_idx over W when multiplying (last dim 1 -> W)
                tiu::mul(weight, weight, stage2_local_idx);
            } else {
                // first index: 使用反向权重
                // reversed_idx should have last dim = 1 so it broadcasts correctly to W
                dim4 reversed_lshape = {1, block_c, iter_h, 1};
                dim4 reversed_shape  = {1, c,     iter_h, 1};
                auto reversed_idx = make_tensor<uint32_t>(reversed_lshape, reversed_shape);
                tiu::fill(reversed_idx, static_cast<uint32_t>(iter_h - 1));
                tiu::sub(reversed_idx, reversed_idx, stage2_local_idx);

                tiu::cast(weight, mask_eq);
                tiu::mul(weight, weight, reversed_idx); // reversed_idx broadcasts to W
            }

            auto weight_fp = make_tensor<fp32>(stage2_full_lshape, outv_shape_);
            tiu::cast(weight_fp, weight);

            auto max_weight_fp = make_tensor<fp32>(stage2_out_lshape, out_final_shape);
            tiu::fill(max_weight_fp, 0.0f);
            tiu::fpool_max(max_weight_fp, weight_fp, &stage2_kernel, &stage2_pad, &stage2_stride, &stage2_dilation);

            auto max_weight = make_tensor<uint32_t>(stage2_out_lshape, out_final_shape);
            tiu::cast(max_weight, max_weight_fp);

            auto stage2_local_result = make_tensor<uint32_t>(stage2_out_lshape, out_final_shape);
            if (select_last_index) {
                tiu::move(stage2_local_result, max_weight);
            } else {
                auto iter_h_minus_1 = make_tensor<uint32_t>(stage2_out_lshape, out_final_shape);
                tiu::fill(iter_h_minus_1, static_cast<uint32_t>(iter_h - 1));
                tiu::sub(stage2_local_result, iter_h_minus_1, max_weight);
            }

            // 使用局部索引选择全局索引
            auto idx_range = make_tensor<uint32_t>(stage2_idx_lshape, stage2_idx_shape);
            arange_broadcast(idx_range, c, 0, 1, iter_h);

            auto final_mask = make_tensor<int32_t>(stage2_full_lshape, outv_shape_);
            tiu::eq(final_mask, idx_range, stage2_local_result, 1);

            auto final_mask_u32 = make_tensor<uint32_t>(stage2_full_lshape, outv_shape_);
            tiu::cast(final_mask_u32, final_mask);

            auto selected_global_idx = make_tensor<uint32_t>(stage2_full_lshape, outv_shape_);
            tiu::mul(selected_global_idx, stage2_i, final_mask_u32);

            // 使用max reduce得到最终索引
            auto selected_global_idx_fp = make_tensor<fp32>(stage2_full_lshape, outv_shape_);
            tiu::cast(selected_global_idx_fp, selected_global_idx);

            auto final_idx_fp = make_tensor<fp32>(stage2_out_lshape, out_final_shape);
            tiu::fill(final_idx_fp, 0.0f);
            tiu::fpool_max(final_idx_fp, selected_global_idx_fp, &stage2_kernel, &stage2_pad, &stage2_stride, &stage2_dilation);

            auto final_idx = make_tensor<uint32_t>(stage2_out_lshape, out_final_shape);
            tiu::cast(final_idx, final_idx_fp);

            dma::store(outv_g.sub_view(out_final_shape, input_global_offset_), result_v);
            dma::store(outi_g.sub_view(out_final_shape, input_global_offset_), final_idx);
        }
    }
}

__KERNEL__ void arg_fp32(uint32_t *ptr_index, fp32 *ptr_input, fp32 *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         bool ismin, const int tile_size) {
    arg_kernel_blocked<fp32>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, ismin, tile_size);

}

__KERNEL__ void arg_fp16(uint32_t *ptr_index, fp16 *ptr_input, fp16 *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         bool ismin, const int tile_size) {
    arg_kernel_blocked<fp16>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, ismin, tile_size);

}

__KERNEL__ void arg_bf16(uint32_t *ptr_index, bf16 *ptr_input, bf16 *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         bool ismin, const int tile_size) {
    arg_kernel_blocked<bf16>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, ismin, tile_size);

}

__KERNEL__ void arg_int32(uint32_t *ptr_index, int32_t *ptr_input, int32_t *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         bool ismin, const int tile_size) {
    arg_kernel_blocked<int32_t>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, ismin, tile_size);

}

__KERNEL__ void arg_int16(uint32_t *ptr_index, int16_t *ptr_input, int16_t *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         bool ismin, const int tile_size) {
    arg_kernel_blocked<int16_t>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, ismin, tile_size);

}

__KERNEL__ void arg_int8(uint32_t *ptr_index, int8_t *ptr_input, int8_t *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         bool ismin, const int tile_size) {
    arg_kernel_blocked<int8_t>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, ismin, tile_size);

}

__KERNEL__ void arg_uint8(uint32_t *ptr_index, uint8_t *ptr_input, uint8_t *ptr_value,
                         int outer_size, int axis_size, int inner_size,
                         bool ismin, const int tile_size) {
    arg_kernel_blocked<uint8_t>(ptr_index, ptr_input, ptr_value,
                        outer_size, axis_size, inner_size, ismin, tile_size);

}
