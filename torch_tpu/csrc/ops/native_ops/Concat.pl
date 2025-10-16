#include "ppl.h"

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif
#define TPU_MAX_CONCAT_NUM 16

template <typename T>
void concat_kernel(T *ptr_output,
                    T *ptr_input0, T *ptr_input1, T *ptr_input2, T *ptr_input3,
                    T *ptr_input4, T *ptr_input5, T *ptr_input6, T *ptr_input7,
                    T *ptr_input8, T *ptr_input9, T *ptr_input10, T *ptr_input11,
                    T *ptr_input12, T *ptr_input13, T *ptr_input14, T *ptr_input15,
                    int input_num, int outer_num, int out_inner_num,
                    int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7,
                    int num8, int num9, int num10, int num11, int num12, int num13, int num14, int num15) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int slice_per_core = div_up(input_num, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, input_num - core_offset);

    dim4 out_glb_shape = {1, 1, outer_num, out_inner_num};
    auto out_gtensor = gtensor<T>(out_glb_shape, GLOBAL, ptr_output);

    int inner_nums[TPU_MAX_CONCAT_NUM] = {num0, num1, num2, num3, num4, num5, num6, num7,
                         num8, num9, num10, num11, num12, num13, num14, num15};
    int cum_offsets[TPU_MAX_CONCAT_NUM] = {0};
    for (int i = 1; i < TPU_MAX_CONCAT_NUM; ++i) {
        cum_offsets[i] = cum_offsets[i-1] + inner_nums[i-1];
    }

    for (int i = core_offset; i < core_offset + slice_size_for_core && i < input_num; ++i) {
        enable_pipeline();

        #define PROCESS_INPUT(idx) \
        if (i == idx) { \
            dim4 in_glb_shape = {1, 1, outer_num, num##idx}; \
            auto in_gtensor = gtensor<T>(in_glb_shape, GLOBAL, ptr_input##idx); \
            dim4 src_stride; \
            get_stride<T>(&src_stride, &in_glb_shape, TPU_ALIGN); \
            dim4 in_offset = {0, 0, 0, cum_offsets[idx]}; \
            auto dst_view = out_gtensor.sub_view(in_glb_shape, in_offset); \
            dma::move(dst_view, in_gtensor.view(in_glb_shape, src_stride)); \
        }

        PROCESS_INPUT(0)  else PROCESS_INPUT(1)  else PROCESS_INPUT(2)  else PROCESS_INPUT(3)
        else PROCESS_INPUT(4)  else PROCESS_INPUT(5)  else PROCESS_INPUT(6)  else PROCESS_INPUT(7)
        else PROCESS_INPUT(8)  else PROCESS_INPUT(9)  else PROCESS_INPUT(10) else PROCESS_INPUT(11)
        else PROCESS_INPUT(12) else PROCESS_INPUT(13) else PROCESS_INPUT(14) else PROCESS_INPUT(15)

        #undef PROCESS_INPUT
    }
}


template <typename T>
void concat_batch_kernel(T *ptr_output,
                    T *ptr_input0, T *ptr_input1, T *ptr_input2, T *ptr_input3,
                    T *ptr_input4, T *ptr_input5, T *ptr_input6, T *ptr_input7,
                    T *ptr_input8, T *ptr_input9, T *ptr_input10, T *ptr_input11,
                    T *ptr_input12, T *ptr_input13, T *ptr_input14, T *ptr_input15,
                    int input_num, int outer_num, int out_inner_num,
                    int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7,
                    int num8, int num9, int num10, int num11, int num12, int num13, int num14, int num15) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(input_num, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, input_num - core_offset);
    int block_h = LANE_NUM;
    dim4 out_glb_shape = {1, 1, out_inner_num, outer_num};

    auto out_gtensor = gtensor<T>(out_glb_shape, GLOBAL, ptr_output);
    int inner_nums[TPU_MAX_CONCAT_NUM] = {num0, num1, num2, num3, num4, num5, num6, num7,
                         num8, num9, num10, num11, num12, num13, num14, num15};
    int cum_offsets[TPU_MAX_CONCAT_NUM] = {0};
    for (int i = 1; i < TPU_MAX_CONCAT_NUM; ++i) {
        cum_offsets[i] = cum_offsets[i-1] + inner_nums[i-1];
    }

    for (int i = core_offset; i < core_offset + slice_size_for_core && i < input_num; ++i) {
        enable_pipeline();

        #define PROCESS_INPUT(idx) \
        if (i == idx) { \
            dim4 in_glb_shape = {1, 1, num##idx, outer_num}; \
            auto in_gtensor = gtensor<T>(in_glb_shape, GLOBAL, ptr_input##idx); \
            dim4 src_stride; \
            get_stride<T>(&src_stride, &in_glb_shape, TPU_ALIGN); \
            dim4 in_offset = {0, 0, cum_offsets[idx], 0}; \
            auto dst_view = out_gtensor.sub_view(in_glb_shape, in_offset); \
            dma::move(dst_view, in_gtensor.view(in_glb_shape, src_stride)); \
        }

        PROCESS_INPUT(0)  else PROCESS_INPUT(1)  else PROCESS_INPUT(2)  else PROCESS_INPUT(3)
        else PROCESS_INPUT(4)  else PROCESS_INPUT(5)  else PROCESS_INPUT(6)  else PROCESS_INPUT(7)
        else PROCESS_INPUT(8)  else PROCESS_INPUT(9)  else PROCESS_INPUT(10) else PROCESS_INPUT(11)
        else PROCESS_INPUT(12) else PROCESS_INPUT(13) else PROCESS_INPUT(14) else PROCESS_INPUT(15)

        #undef PROCESS_INPUT
    }

}

__KERNEL__ void concat_fp32(fp32 *ptr_output, fp32 *ptr_input0, fp32 *ptr_input1, fp32 *ptr_input2, fp32 *ptr_input3,
                            fp32 *ptr_input4, fp32 *ptr_input5, fp32 *ptr_input6, fp32 *ptr_input7,
                            fp32 *ptr_input8, fp32 *ptr_input9, fp32 *ptr_input10, fp32 *ptr_input11,
                            fp32 *ptr_input12, fp32 *ptr_input13, fp32 *ptr_input14, fp32 *ptr_input15,
                            int input_num, int outer_num, int out_inner_num, int dim,
                            int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7,
                            int num8, int num9, int num10, int num11, int num12, int num13, int num14, int num15) {
    if (dim == 0) {
        concat_batch_kernel<fp32>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                            ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                            ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                            num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    } else {
        concat_kernel<fp32>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                                ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                                ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                                num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    }
}

__KERNEL__ void concat_fp16(fp16 *ptr_output, fp16 *ptr_input0, fp16 *ptr_input1, fp16 *ptr_input2, fp16 *ptr_input3,
                            fp16 *ptr_input4, fp16 *ptr_input5, fp16 *ptr_input6, fp16 *ptr_input7,
                            fp16 *ptr_input8, fp16 *ptr_input9, fp16 *ptr_input10, fp16 *ptr_input11,
                            fp16 *ptr_input12, fp16 *ptr_input13, fp16 *ptr_input14, fp16 *ptr_input15,
                            int input_num, int outer_num, int out_inner_num, int dim,
                            int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7,
                            int num8, int num9, int num10, int num11, int num12, int num13, int num14, int num15) {
    if (dim == 0) {
        concat_batch_kernel<fp16>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                            ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                            ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                            num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    } else {
        concat_kernel<fp16>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                                ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                                ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                                num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    }
}

__KERNEL__ void concat_bf16(bf16 *ptr_output, bf16 *ptr_input0, bf16 *ptr_input1, bf16 *ptr_input2, bf16 *ptr_input3,
                            bf16 *ptr_input4, bf16 *ptr_input5, bf16 *ptr_input6, bf16 *ptr_input7,
                            bf16 *ptr_input8, bf16 *ptr_input9, bf16 *ptr_input10, bf16 *ptr_input11,
                            bf16 *ptr_input12, bf16 *ptr_input13, bf16 *ptr_input14, bf16 *ptr_input15,
                            int input_num, int outer_num, int out_inner_num, int dim,
                            int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7,
                            int num8, int num9, int num10, int num11, int num12, int num13, int num14, int num15) {
    if (dim == 0) {
        concat_batch_kernel<bf16>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                            ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                            ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                            num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    } else {
        concat_kernel<bf16>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                                ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                                ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                                num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    }
}

__KERNEL__ void concat_int32(int32 *ptr_output, int32 *ptr_input0, int32 *ptr_input1, int32 *ptr_input2, int32 *ptr_input3,
                            int32 *ptr_input4, int32 *ptr_input5, int32 *ptr_input6, int32 *ptr_input7,
                            int32 *ptr_input8, int32 *ptr_input9, int32 *ptr_input10, int32 *ptr_input11,
                            int32 *ptr_input12, int32 *ptr_input13, int32 *ptr_input14, int32 *ptr_input15,
                            int input_num, int outer_num, int out_inner_num, int dim,
                            int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7,
                            int num8, int num9, int num10, int num11, int num12, int num13, int num14, int num15) {
    if (dim == 0) {
        concat_batch_kernel<int32>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                            ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                            ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                            num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    } else {
        concat_kernel<int32>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                                ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                                ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                                num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    }
}

__KERNEL__ void concat_int16(int16 *ptr_output, int16 *ptr_input0, int16 *ptr_input1, int16 *ptr_input2, int16 *ptr_input3,
                            int16 *ptr_input4, int16 *ptr_input5, int16 *ptr_input6, int16 *ptr_input7,
                            int16 *ptr_input8, int16 *ptr_input9, int16 *ptr_input10, int16 *ptr_input11,
                            int16 *ptr_input12, int16 *ptr_input13, int16 *ptr_input14, int16 *ptr_input15,
                            int input_num, int outer_num, int out_inner_num, int dim,
                            int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7,
                            int num8, int num9, int num10, int num11, int num12, int num13, int num14, int num15) {
    if (dim == 0) {
        concat_batch_kernel<int16>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                            ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                            ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                            num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    } else {
        concat_kernel<int16>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                                ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                                ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                                num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    }
}

__KERNEL__ void concat_int8(int8 *ptr_output, int8 *ptr_input0, int8 *ptr_input1, int8 *ptr_input2, int8 *ptr_input3,
                            int8 *ptr_input4, int8 *ptr_input5, int8 *ptr_input6, int8 *ptr_input7,
                            int8 *ptr_input8, int8 *ptr_input9, int8 *ptr_input10, int8 *ptr_input11,
                            int8 *ptr_input12, int8 *ptr_input13, int8 *ptr_input14, int8 *ptr_input15,
                            int input_num, int outer_num, int out_inner_num, int dim,
                            int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7,
                            int num8, int num9, int num10, int num11, int num12, int num13, int num14, int num15) {
    if (dim == 0) {
        concat_batch_kernel<int8>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                            ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                            ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                            num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    } else {
        concat_kernel<int8>(ptr_output, ptr_input0, ptr_input1, ptr_input2, ptr_input3, ptr_input4, ptr_input5, ptr_input6,
                                ptr_input7, ptr_input8, ptr_input9, ptr_input10, ptr_input11, ptr_input12, ptr_input13,
                                ptr_input14, ptr_input15, input_num, outer_num, out_inner_num,
                                num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    }
}