#include "ppl.h"

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void reduce_kernel_w(T *ptr_output, T *ptr_input, int row, int column, int axis, int reduction, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int slice_per_core = div_up(column, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, column - core_offset);

    dim4 input_shape = {1, 1, row, column};
    dim4 output_shape = {1, 1, 1, column};
    auto in_gtensor = gtensor<T>(input_shape, GLOBAL, ptr_input);
    auto out_gtensor = gtensor<T>(output_shape, GLOBAL, ptr_output);

    int block_h = LANE_NUM;
    dim4 local_in_block_shape = {1, 1, block_h, block_w};
    dim4 intermediate_shape = {1, 1, 1, block_w};
    dim4 local_out_block_shape = {1, 1, 1, block_w};

    padding_t pad = {0, 0, 0, 0};
    dim2 stride = {1, 1};
    dim2 dilation = {1, 1};
    dim2 ins = {0, 0};

    for (auto w_idx = 0; w_idx < slice_size_for_core; w_idx += block_w) {
        int w = min(block_w, slice_size_for_core - w_idx);
        dim4 output_offset = {0, 0, 0, core_offset + w_idx};
        dim4 local_out_shape = {1, 1, 1, w};

        auto intermediate = make_tensor<T>(intermediate_shape, local_out_shape);
        auto intermediate_fp32 = make_tensor<fp32>(intermediate_shape, local_out_shape);
        auto res_add = make_tensor<T>(intermediate_shape, local_out_shape);
        auto res_add_fp32 = make_tensor<fp32>(intermediate_shape, local_out_shape);
        tiu::zero(intermediate);
        tiu::zero(intermediate_fp32);
        tiu::zero(res_add);
        tiu::zero(res_add_fp32);

        for (auto h_idx = 0; h_idx < row; h_idx += block_h) {
            enable_pipeline();
            int h = min(block_h, row - h_idx);
            dim4 local_in_shape = {1, 1, h, w};
            dim4 input_offset = {0, 0, h_idx, core_offset + w_idx};

            auto local_in = make_tensor<T>(local_in_block_shape, local_in_shape);
            dma::load(local_in, in_gtensor.sub_view(local_in_shape, input_offset));
            dim2 kernel1 = {h, 1};
            if constexpr (std::is_same<T, fp32>::value) {
                tiu::fpool_avg(intermediate, local_in, &kernel1, &pad, &stride, &dilation, &ins, 1.0f);
                tiu::fadd(res_add, res_add, intermediate);
            } else if constexpr (std::is_same<T, fp16>::value || std::is_same<T, bf16>::value) {
                auto local_in_fp32 = make_tensor<fp32>(local_in_block_shape, local_in_shape);
                tiu::cast(local_in_fp32, local_in);
                tiu::fpool_avg(intermediate_fp32, local_in_fp32, &kernel1, &pad, &stride, &dilation, &ins, 1.0f);
                tiu::fadd(res_add_fp32, res_add_fp32, intermediate_fp32);
            }
        }

        auto res = make_tensor<T>(local_out_block_shape, local_out_shape);
        dim2 kernel2 = {1, 1};
        float scale = (reduction == 0) ? (1.0f / row) : 1.0f;
        if constexpr (std::is_same<T, fp32>::value) {
            tiu::fpool_avg(res, res_add, &kernel2, &pad, &stride, &dilation, &ins, scale);
        } else if constexpr (std::is_same<T, fp16>::value || std::is_same<T, bf16>::value) {
            auto res_fp32 = make_tensor<fp32>(local_out_block_shape, local_out_shape);
            tiu::fpool_avg(res_fp32, res_add_fp32, &kernel2, &pad, &stride, &dilation, &ins, scale);
            tiu::cast(res, res_fp32);
        }
        dma::store(out_gtensor.sub_view(local_out_shape, output_offset), res);
    }
}


template <typename T>
void reduce_kernel_c(T *ptr_output, T *ptr_input, int row, int column, int axis, int reduction, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int slice_per_core = div_up(row, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, row - core_offset);

    dim4 input_shape = {1, row, 1, column};
    dim4 output_shape = {1, row, 1, 1};
    auto in_gtensor = gtensor<T>(input_shape, GLOBAL, ptr_input);
    auto out_gtensor = gtensor<T>(output_shape, GLOBAL, ptr_output);

    int block_c = LANE_NUM;
    dim4 local_in_block_shape = {1, block_c, 1, block_w};
    dim4 intermediate_shape = {1, block_c, 1, 1};
    dim4 local_out_block_shape = {1, block_c, 1, 1};

    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        dim4 output_offset = {0, core_offset + c_idx, 0, 0};
        dim4 local_out_shape = {1, c, 1, 1};

        auto intermediate = make_tensor<T>(intermediate_shape, local_out_shape);
        auto intermediate_fp32 = make_tensor<fp32>(intermediate_shape, local_out_shape);
        auto res_add = make_tensor<T>(intermediate_shape, local_out_shape);
        auto res_add_fp32 = make_tensor<fp32>(intermediate_shape, local_out_shape);
        tiu::zero(intermediate);
        tiu::zero(intermediate_fp32);
        tiu::zero(res_add);
        tiu::zero(res_add_fp32);
        padding_t pad = {0, 0, 0, 0};
        dim2 stride = {1, 1};
        dim2 dilation = {1, 1};
        dim2 ins = {0, 0};

        for (auto w_idx = 0; w_idx < column; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, column - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};

            auto local_in = make_tensor<T>(local_in_block_shape, local_in_shape);
            dma::load(local_in, in_gtensor.sub_view(local_in_shape, input_offset));

            dim2 kernel1 = {1, w};
            if constexpr (std::is_same<T, fp32>::value) {
                tiu::fpool_avg(intermediate, local_in, &kernel1, &pad, &stride, &dilation, &ins, 1.0f);
                tiu::fadd(res_add, res_add, intermediate);
            } else if constexpr (std::is_same<T, fp16>::value || std::is_same<T, bf16>::value) {
                auto local_in_fp32 = make_tensor<fp32>(local_in_block_shape, local_in_shape);
                tiu::cast(local_in_fp32, local_in);
                tiu::fpool_avg(intermediate_fp32, local_in_fp32, &kernel1, &pad, &stride, &dilation, &ins, 1.0f);
                tiu::fadd(res_add_fp32, res_add_fp32, intermediate_fp32);
            }
        }

        auto res = make_tensor<T>(local_out_block_shape, local_out_shape);
        dim2 kernel2 = {1, 1};
        float scale = (reduction == 0 && column > 0) ? (1.0f / column) : 1.0f;
        if constexpr (std::is_same<T, fp32>::value) {
            tiu::fpool_avg(res, res_add, &kernel2, &pad, &stride, &dilation, &ins, scale);
        } else if constexpr (std::is_same<T, fp16>::value || std::is_same<T, bf16>::value) {
            auto res_fp32 = make_tensor<fp32>(local_out_block_shape, local_out_shape);
            tiu::fpool_avg(res_fp32, res_add_fp32, &kernel2, &pad, &stride, &dilation, &ins, scale);
            tiu::cast(res, res_fp32);
        }
        dma::store(out_gtensor.sub_view(local_out_shape, output_offset), res);
    }
}

__KERNEL__ void reduce_fp32(fp32 *ptr_output, fp32 *ptr_input,
                             int row, int column, int axis, int reduction, const int block_w) {
    if (axis == 0)
        reduce_kernel_w<fp32>(ptr_output, ptr_input, row, column, axis, reduction, block_w);
    else
        reduce_kernel_c<fp32>(ptr_output, ptr_input, row, column, axis, reduction, block_w);
}

__KERNEL__ void reduce_fp16(fp16 *ptr_output, fp16 *ptr_input,
                             int row, int column, int axis, int reduction, const int block_w) {
    if (axis == 0)
        reduce_kernel_w<fp16>(ptr_output, ptr_input, row, column, axis, reduction, block_w);
    else
        reduce_kernel_c<fp16>(ptr_output, ptr_input, row, column, axis, reduction, block_w);
}

__KERNEL__ void reduce_bf16(bf16 *ptr_output, bf16 *ptr_input,
                             int row, int column, int axis, int reduction, const int block_w) {
    if (axis == 0)
        reduce_kernel_w<bf16>(ptr_output, ptr_input, row, column, axis, reduction, block_w);
    else
        reduce_kernel_c<bf16>(ptr_output, ptr_input, row, column, axis, reduction, block_w);
}

