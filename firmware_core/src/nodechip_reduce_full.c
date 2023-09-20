#include <math.h>
#include <string.h>

#include "sg_api_struct.h"
#include "tpu_kernel.h"

#define MAX_SHAPE 8
#define CW_LIMIT 65535
typedef struct {
    int nsecs;
    int csecs;
    int hsecs;
    int wsecs;
} secs_info_t;

inline static void pipeline_move(int *array, int num) {
    for (int i = num - 1; i > 0; i--) {
        array[i] = array[i - 1];
    }
}

inline static int update_tensor_size(int* shape) {
    return shape[0] * shape[1] * shape[2] * shape[3] * sizeof(int);
}

inline static int find_smaller_factor(int num) {
    int factor, sqrt_integer = (int)sqrt(num);
    if (num % sqrt_integer == 0) return sqrt_integer;
    for (factor = sqrt_integer - 1; factor > 0; factor--) {
        if (num % factor == 0) break;
    }
    return factor;
}

extern void nodechip_active(global_addr_t in_global_addr,
                            global_addr_t out_global_addr,
                            const int *shape,
                            int shape_dim,
                            data_type_t dtype,
                            sg_active_type_t active_type,
                            float *coef);

extern void nodechip_transpose(global_addr_t         input_global_addr,
                              global_addr_t         output_global_addr,
                              int*                  input_shape,
                              int*                  order,
                              int                   dims,
                              global_addr_t         buffer_global_addr,
                              unsigned long long*   buffer_size,
                              data_type_t           dtype);

static void nodechip_reduce_onedim_local(
local_addr_t input_local_addr,
local_addr_t buffer_local_addr,
local_addr_t output_local_addr,
const dim4 *ishape,
int method,
bool is_reduce_h, // true: reduce h, false: reduce w
data_type_t dtype) {
    const int eu_num = tpu_eu_num(dtype);

    if (method == REDUCE_PROD) {
        int c_per_npu = DIV_UP(ishape->c, tpu_npu_num());
        dim4 cpy_dim = {ishape->n, ishape->c, 1, ishape->w};
        dim4 src_str = {.n = ALIGN(ishape->h * ishape->w, eu_num) * c_per_npu,
                        .c = ALIGN(ishape->h, ishape->w),
                        .h = ishape->w,
                        .w = 1};
        dim4 dst_str = {.n = ALIGN(ishape->w, eu_num) * c_per_npu,
                        .c = ALIGN(ishape->w, eu_num),
                        .h = ishape->w,
                        .w = 1};
        if (is_reduce_h) {
            for(int i = 0; i<ishape->h;++i) {
                if (i == 0) {
                    tpu_bdc_cpy(output_local_addr,
                                input_local_addr,
                                &cpy_dim,
                                &dst_str,
                                &src_str,
                                dtype);
                }
                else {
                    tpu_bdc_fp_mul(output_local_addr,
                                   input_local_addr + i * ishape->w * tpu_data_type_size(dtype),
                                   output_local_addr,
                                   &cpy_dim,
                                   &dst_str,
                                   &src_str,
                                   &dst_str,
                                   dtype);
                }
            }
        }
        else {
            // 确保reduce两两相乘，缩短一半
            int head = pow(2, (int)log2(ishape->w)); // 将w向下取整至距离最近的2的某个幂次方
            int tail = ishape->w - head;             // 减去2的幂次方后，额外需要处理的部分
            TPUKERNEL_ASSERT(head > tail);
            if (tail > 0) {     // 存在额外需要处理的部分
                cpy_dim.h = ishape->h;
                cpy_dim.w = tail;
                dst_str.h = ishape->w;
                dst_str.c = ALIGN(dst_str.h * cpy_dim.h, eu_num);
                dst_str.n = c_per_npu * dst_str.c;
                tpu_bdc_fp_mul(buffer_local_addr,
                               input_local_addr,
                               input_local_addr + head * tpu_data_type_size(dtype),
                               &cpy_dim,
                               &dst_str,
                               &src_str,
                               &src_str,
                               dtype);
                cpy_dim.w = head - tail;
                tpu_bdc_cpy(buffer_local_addr + tail * tpu_data_type_size(dtype),
                            input_local_addr + tail * tpu_data_type_size(dtype),
                            &cpy_dim,
                            &dst_str,
                            &src_str,
                            dtype);
                input_local_addr = buffer_local_addr;
            }

            // process head, and we keep dst stride == src stride, and then copy to output addr
            src_str.h = ishape->w;
            src_str.c = ALIGN(src_str.h * ishape->h, eu_num);
            src_str.n = c_per_npu * src_str.c;
            int left = head;
            while(left > 1) {
                dim4 prod_dim = { ishape->n, ishape->c, ishape->h, left >> 1 };
                tpu_bdc_fp_mul(buffer_local_addr,
                            head == left ? input_local_addr : buffer_local_addr,
                            (head == left ? input_local_addr : buffer_local_addr) + (left >> 1) * tpu_data_type_size(dtype),
                            &prod_dim,
                            &src_str,
                            &src_str,
                            &src_str,
                            dtype);
                left = left >> 1;
            }
            // copy data from buffer to output
            cpy_dim.h = ishape->h;
            cpy_dim.w = 1;
            dst_str.w = 1;
            dst_str.h = 1;
            dst_str.c = ALIGN(cpy_dim.h * dst_str.h, eu_num);
            dst_str.n = c_per_npu * dst_str.c;
            tpu_bdc_cpy(output_local_addr,
                        head == 1 ? input_local_addr : buffer_local_addr,
                        &cpy_dim,
                        &dst_str,
                        &src_str,
                        dtype);
        }
    }
}

/**
 * process_shape: eliminate dim = 1
 *    1. if dims < 4, fill 1 in the front dim to make dims=4, e.g. (N, c, 1, W) --> (1, N, c, W)
 *    2. if dims > 4, merge dims that are not reduced, e.g. (A, B, c, d, e) --> (AB, c, d, e)
 * the following is not implemented yet:
 *    3. if still dims > 4, do transpose, e.g. (A, b, C, d, E) --> (ACE, bd) or (a, B, c, D, e) --> (BD, ace)
 *    then (ACD, bd) --> (1, ACD, b, d) or (BD, ace) --> (1, BD, a, ce) to do reduce_hw
 */
static int process_shape(
int *input_shape,
int *axis_list,
int *axis_num,
const int *input_shape_orig,
int shape_dim_orig) {
    bool is_reduce_orig[MAX_SHAPE] = {0};
    bool is_reduce[MAX_SHAPE] = {0};
    for(int i = 0; i < MAX_SHAPE; ++i) {
        input_shape[i] = 1;
        is_reduce[i] = is_reduce_orig[i] = false;
    }
    for(int i = 0; i < *axis_num; ++i) {
        is_reduce_orig[axis_list[i]] = true;
    }


    int pos = 0, reduce_pos = 0;
    for(int i = 0; i < shape_dim_orig; ++i) {
        if(input_shape_orig[i] == 1) {
            is_reduce_orig[i] = false;
            continue;    // eliminate dim = 1
        }
        if(is_reduce_orig[i]) {
            axis_list[reduce_pos++] = pos;
            is_reduce[pos] = true;
        }
        input_shape[pos++] = input_shape_orig[i];
    }

    if(pos < 4) {// if dims < 4, fill 1 in the front dim to make dims=4, e.g.(N, c, 1, W)-->(1, N, c, W)
        for(int i = 3; i >= 0; --i) {
            if(i < 4 - pos) {
                input_shape[i] = 1;
            }
            else {
                input_shape[i] = input_shape[i + pos - 4];
            }
        }
        for(int i = 0; i < reduce_pos; ++i) {
            axis_list[i] += (4 - pos);
        }
        pos = 4;
    }
    else if(pos > 4) {// merge dims that are not reduced, e.g.(A, B, c, d, e)-->(AB, c, d, e)
        int shape_dims = pos;
        int minimum_merged_dims = shape_dims;
        for(int i = 1; i < shape_dims; ++i) {
            if(!is_reduce[i - 1] && !is_reduce[i]) { // merge adjacent non-reduced dims
                minimum_merged_dims--;
            }
        }
        // judge whether the remaining dims after merge <= 4
        TPUKERNEL_ASSERT(minimum_merged_dims <= 4);

        pos = reduce_pos = 0;
        int current_dims = 0;
        // e.g. A B c d E F --> AB c d EF E F
        // dim(pos) : 6 --> 4
        for(int i = 1; i < shape_dims; ++i) {
            if(!is_reduce[i - 1] && !is_reduce[i] && (shape_dims - current_dims > 4)) {
                input_shape[pos] *= input_shape[i]; // two adjacent non-reduced dims multiply
                current_dims++; // dim--
            }
            else {
                if(is_reduce[i - 1]) {
                    axis_list[reduce_pos++] = pos;
                }
                input_shape[++pos] = input_shape[i];
            }
        }
        if(is_reduce[shape_dims - 1]) {
            axis_list[reduce_pos++] = pos;
        }
        ++pos;
    }

    int shape_sum = 0;
    for(int i = 0; i < MAX_SHAPE; ++i) {
        shape_sum += input_shape[i];
    }
    if(shape_sum - input_shape[3] == 7 && axis_list[0] == 3) { // one dimension - w
        const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
        const int imm_res1 = (LOCAL_MEM_SIZE - 3 * bank_size) / 2;
        const int imm_res2 = (imm_res1 / bank_size) * bank_size - 64;
        const int reduce_hw_min = (imm_res2 / 64) * 16;

        // if shape_w is too large, then convert to reduce h and w
        if (input_shape[3] > reduce_hw_min) {
            int shape_h = find_smaller_factor(input_shape[3]);
            int shape_w = input_shape[3]/shape_h;
            input_shape[2] = shape_h;
            input_shape[3] = shape_w;
            axis_list[0] = 2;
            axis_list[1] = 3;
            *axis_num = 2;
            TPUKERNEL_ASSERT(pos == 4);
            return pos;
        }
    }

    *axis_num = reduce_pos;
    TPUKERNEL_ASSERT(pos == 4);
    return pos;
}

static void split_cw_align_bank(
const int *input_shape,
int shape_dims,
data_type_t dtype,
secs_info_t *secs_info) {
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    int c_per_npu = DIV_UP(input_shape[1], NPU_NUM);
    int feature_size = tpu_aligned_feature_size(input_shape[2], input_shape[3], dtype);
    int input_tensor_size = input_shape[0] * c_per_npu * feature_size;
    int output_tensor_size = input_shape[0] * c_per_npu * 
                             tpu_aligned_feature_size(1, input_shape[3], dtype);  // reduce h(h=1)
    int total_tensor_size = ALIGN(input_tensor_size, bank_size) * 2
                          + ALIGN(output_tensor_size, bank_size) * 2;
    // if can compute all data at once, return
    if (total_tensor_size <= LOCAL_MEM_SIZE) return;
    
    // else batch, split untill total_tensor_size <= LOCAL_MEM_SIZE
    secs_info->csecs = c_per_npu;
    total_tensor_size = ALIGN(input_tensor_size / c_per_npu, bank_size) * 2
                      + ALIGN(output_tensor_size / c_per_npu, bank_size) * 2;
    int wsecs = DIV_UP(total_tensor_size, LOCAL_MEM_SIZE);
    while (total_tensor_size > LOCAL_MEM_SIZE) {
        if (wsecs++ > input_shape[3]) break;

        int wslice = DIV_UP(input_shape[3], wsecs);
        input_tensor_size = input_shape[0] * tpu_aligned_feature_size(input_shape[2], wslice, dtype);
        output_tensor_size = input_shape[0] * tpu_aligned_feature_size(1, wslice, dtype);
        total_tensor_size = ALIGN(input_tensor_size, bank_size) * 2
                          + ALIGN(output_tensor_size, bank_size) * 2;
    }
    TPUKERNEL_ASSERT(wsecs <= input_shape[3]);

    secs_info->wsecs = wsecs;
}

static void split_ch_align_bank(
const int *input_shape,
int shape_dims,
data_type_t dtype,
secs_info_t *secs_info) {
    // buffer1: store reduce w res with the whole h, only needed when hsecs > 1
    // buffer2: for REDUCE_MIN, store 0-input(here share with input local addr).
    // buffer3: store reduce w res with hslice
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    const int c_per_npu = DIV_UP(input_shape[1], NPU_NUM);

    int input_tensor_size = input_shape[0] * c_per_npu * 
                            tpu_aligned_feature_size(input_shape[2], input_shape[3], dtype);
    int output_tensor_size = input_shape[0] * c_per_npu * 
                             tpu_aligned_feature_size(input_shape[2], 1, dtype);
    int total_tensor_size = ALIGN(input_tensor_size, bank_size) * 2
                          + ALIGN(output_tensor_size, bank_size) * 2;
    if (total_tensor_size <= LOCAL_MEM_SIZE) return;

    // update csecs
    int csecs = 1;
    int input_csize = input_tensor_size;
    int output_csize = output_tensor_size;
    int cslice_per_npu = DIV_UP(c_per_npu, csecs);
    while ((total_tensor_size > LOCAL_MEM_SIZE && csecs < c_per_npu) || (cslice_per_npu * NPU_NUM > CW_LIMIT)) {
        csecs++;
        cslice_per_npu = DIV_UP(c_per_npu, csecs);
        input_csize = input_tensor_size/c_per_npu*cslice_per_npu;
        output_csize = output_tensor_size/c_per_npu*cslice_per_npu;
        total_tensor_size = ALIGN(input_csize, bank_size) * 2
                          + ALIGN(output_csize, bank_size) * 2;
    }
    secs_info->csecs = csecs;
    if (total_tensor_size <= LOCAL_MEM_SIZE) return;

    // update hsecs
    total_tensor_size = ALIGN(input_csize, bank_size) * 2
                      + ALIGN(output_csize, bank_size) * 2;
    int hsecs = DIV_UP(total_tensor_size, LOCAL_MEM_SIZE);
    int hslice = DIV_UP(input_shape[2], hsecs);
    while (total_tensor_size > LOCAL_MEM_SIZE) {
        if (hslice == 1) {
            hslice = -1;
            break;
        }
        hsecs = DIV_UP(input_shape[2], hslice-1);
        hslice = DIV_UP(input_shape[2], hsecs);

        input_csize = input_shape[0] * cslice_per_npu *
            tpu_aligned_feature_size(hslice, input_shape[3], dtype);
        output_csize = input_shape[0] * cslice_per_npu *
                       tpu_aligned_feature_size(hslice, 1, dtype);
        total_tensor_size = ALIGN(input_csize, bank_size) * 2
                          + ALIGN(output_csize, bank_size) * 2;
    }
    TPUKERNEL_ASSERT(hslice > 0);

    secs_info->hsecs = hsecs;
}

static void nodechip_reduce_h(
global_addr_t input_global_addr,
global_addr_t output_global_addr,
const int *shape,
int shape_dims,
int method,
data_type_t dtype) {
    const int dtype_size = tpu_data_type_size(dtype);
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;

    int input_shape[4] = {0};
    memcpy(input_shape, shape, sizeof(int) * 4);
    if(input_shape[0] * input_shape[1] <= CW_LIMIT) {
        input_shape[1] = input_shape[0] * input_shape[1];   // merge n c dims
        input_shape[0] = 1;
    }

    secs_info_t secs_info = {1, 1, 1, 1};
    split_cw_align_bank(input_shape, shape_dims, dtype, &secs_info);
    int csecs = secs_info.csecs;    // secs indicates how many slice
    int wsecs = secs_info.wsecs;

    int c_per_npu = DIV_UP(input_shape[1], NPU_NUM);
    int c_slice_per_npu = DIV_UP(c_per_npu, csecs);
    int w_slice = DIV_UP(input_shape[3], wsecs);    // slice indicates size of per slice
    int isize = c_slice_per_npu * tpu_aligned_feature_size(input_shape[2], w_slice, dtype);
    int osize = c_slice_per_npu * tpu_aligned_feature_size(1, w_slice, dtype);  // reduce h(h=1)

    local_addr_t input_local_addrs[2] = {0, ALIGN(isize, bank_size)};
    local_addr_t output_local_addrs[2] = {0};
    output_local_addrs[0] = input_local_addrs[1] + ALIGN(isize, bank_size);
    output_local_addrs[1] = output_local_addrs[0] + ALIGN(osize, bank_size);

    const int cslice_per_npu = c_per_npu / csecs;       // compute times of per npu
    const int c_residue_per_npu = c_per_npu % csecs;    // 
    const int wslice = input_shape[3] / wsecs;
    const int w_residue = input_shape[3] % wsecs;

    dim4 input_global_stride, output_global_stride;
    dim4 input_global_shape = {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
    dim4 output_global_shape = {input_shape[0], input_shape[1], 1, input_shape[3]};
    tpu_continuous_stride(&input_global_stride, &input_global_shape);
    tpu_continuous_stride(&output_global_stride, &output_global_shape);

    int cidx[3] = {0}, widx[3] = {0};
    int real_cslice[3] = {0}, real_wslice[3] = {0};
    int stage_idx = 0, draining_idx = 0;
    while(cidx[2] < input_shape[1]) {
        tpu_parallel_start();
        if(draining_idx < 1) {
            int csec_idx = stage_idx / wsecs;
            int wsec_idx = stage_idx % wsecs;
            int real_cslice_per_npu = cslice_per_npu + (csec_idx < c_residue_per_npu);
            real_cslice[0] = MIN(NPU_NUM * real_cslice_per_npu, input_shape[1] - cidx[0]);
            real_wslice[0] = wslice + (wsec_idx < w_residue);
        }

        if(stage_idx > 1) {
            long long out_offset = ((long long)cidx[2] * input_shape[3] + widx[2]) * dtype_size;
            dim4 real_local_shape = {.n = input_shape[0],
                                     .c = real_cslice[2],
                                     .h = 1,
                                     .w = real_wslice[2]};
            tpu_gdma_cpy_L2S(output_global_addr + out_offset,
                             output_local_addrs[stage_idx % 2],
                             &real_local_shape,
                             &output_global_stride,
                             NULL,
                             dtype);
        }
        if(stage_idx > 0 && draining_idx < 2) {
            dim4 real_local_shape = {.n = input_shape[0],
                                     .c = real_cslice[1],
                                     .h = input_shape[2],
                                     .w = real_wslice[1]};
            nodechip_reduce_onedim_local(input_local_addrs[(stage_idx-1) % 2],
                                         input_local_addrs[(stage_idx-1) % 2],
                                         output_local_addrs[(stage_idx-1) % 2],
                                         &real_local_shape,
                                         method,
                                         true, // is_reduce_h
                                         dtype);
        }

        if (draining_idx < 1) {
        dim4 real_local_shape = {.n = input_shape[0],
                                 .c = real_cslice[0],
                                 .h = input_shape[2],
                                 .w = real_wslice[0]
        };
        long long in_offset = ((long long)cidx[0] * input_shape[2] * input_shape[3] +
                               (long long)widx[0]) * dtype_size;
        tpu_gdma_cpy_S2L(input_local_addrs[stage_idx % 2],
                         input_global_addr + in_offset,
                         &real_local_shape,
                         NULL,
                         &input_global_stride,
                         dtype);
        }
        tpu_parallel_end();

        pipeline_move(cidx, 3);
        pipeline_move(widx, 3);
        pipeline_move(real_cslice, 3);
        pipeline_move(real_wslice, 3);
        if (draining_idx < 1) {
            widx[0] += real_wslice[0];
            if (widx[0] >= input_shape[3]) {
                widx[0] = 0;
                cidx[0] += real_cslice[0];
                if (cidx[0] >= input_shape[1]) {
                    draining_idx++;
                }
            }
        }
        else {
            draining_idx++;
        }
        stage_idx++;
    }
}

static void nodechip_reduce_w(
global_addr_t input_global_addr,
global_addr_t output_global_addr,
const int *shape,
int shape_dims,
int method,
data_type_t dtype) {
    const int type_size = tpu_data_type_size(dtype);
    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;

    int input_shape[4] = {0};
    memcpy(input_shape, shape, sizeof(int) * 4);
    if(input_shape[0] * input_shape[1] <= CW_LIMIT) {
        input_shape[1] = input_shape[0] * input_shape[1];   // merge n c dims
        input_shape[0] = 1;
    }

    secs_info_t secs_info = {1, 1, 1, 1};
    split_ch_align_bank(input_shape, shape_dims, dtype, &secs_info);
    int csecs = secs_info.csecs;
    int hsecs = secs_info.hsecs;

    int c_per_npu = DIV_UP(input_shape[1], NPU_NUM);
    int c_slice_per_npu = DIV_UP(c_per_npu, csecs);
    int h_slice = DIV_UP(input_shape[2], hsecs);
    int isize = c_slice_per_npu * tpu_aligned_feature_size(h_slice, input_shape[3], dtype);
    int osize = c_slice_per_npu * tpu_aligned_feature_size(h_slice, 1, dtype);

    local_addr_t input_local_addrs[2]  = {0, ALIGN(isize, bank_size)};
    local_addr_t output_local_addrs[2] = {0};
    output_local_addrs[0] = input_local_addrs[1] + ALIGN(isize, bank_size);
    output_local_addrs[1] = output_local_addrs[0] + ALIGN(osize, bank_size);

    const int cslice_per_npu    = c_per_npu / csecs;
    const int c_residue_per_npu = c_per_npu % csecs;
    const int hslice    = input_shape[2] / hsecs;
    const int h_residue = input_shape[2] % hsecs;

    dim4 input_global_stride, output_global_stride;
    dim4 input_global_shape = {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
    dim4 output_global_shape = {input_shape[0], input_shape[1], input_shape[2], 1};
    tpu_continuous_stride(&input_global_stride, &input_global_shape);
    tpu_continuous_stride(&output_global_stride, &output_global_shape);

    int cidx[3] = {0}, hidx[3] = {0};
    int real_cslice[3] = {0}, real_hslice[3] = {0};
    int stage_idx = 0, draining_idx = 0;
    while (cidx[2] < input_shape[1]) {
        tpu_parallel_start();
        if (draining_idx < 1) {
            int hsec_idx = stage_idx % hsecs;
            int csec_idx = stage_idx / hsecs;
            int real_cslice_per_npu = cslice_per_npu + (csec_idx < c_residue_per_npu);
            real_cslice[0] = MIN(NPU_NUM * real_cslice_per_npu, input_shape[1] - cidx[0]);
            real_hslice[0] = hslice + (hsec_idx < h_residue);
        }

        if (stage_idx > 1) {
            long long out_offset = ((long long)cidx[2] * input_shape[2] + hidx[2]) * type_size;
            dim4 real_local_shape = {.n = input_shape[0],
                                     .c = real_cslice[2],
                                     .h = real_hslice[2],
                                     .w = 1};
            tpu_gdma_cpy_L2S(output_global_addr + out_offset,
                             output_local_addrs[stage_idx % 2],
                             &real_local_shape,
                             &output_global_stride,
                             NULL,
                             dtype);
        }

        if (stage_idx > 0 && draining_idx < 2) {
            dim4 real_local_shape = {.n = input_shape[0],
                                     .c = real_cslice[1],
                                     .h = real_hslice[1],
                                     .w = input_shape[3]};
            nodechip_reduce_onedim_local(input_local_addrs[(stage_idx-1) % 2],
                                         input_local_addrs[(stage_idx-1) % 2],
                                         output_local_addrs[(stage_idx-1) % 2],
                                         &real_local_shape,
                                         method,
                                         false, // is_reduce_h
                                         dtype);
        }

        if (draining_idx < 1) {
            dim4 real_local_shape = {.n = input_shape[0],
                                     .c = real_cslice[0],
                                     .h = real_hslice[0],
                                     .w = input_shape[3]
            };
            long long in_offset = ((long long)cidx[0] * input_shape[2] * input_shape[3] +
                                    (long long)hidx[0] * input_shape[3]) * type_size;
            tpu_gdma_cpy_S2L(input_local_addrs[stage_idx % 2],
                             input_global_addr + in_offset,
                             &real_local_shape,
                             NULL,
                             &input_global_stride,
                             dtype);
        }
        tpu_parallel_end();

        pipeline_move(cidx, 3);
        pipeline_move(hidx, 3);
        pipeline_move(real_cslice, 3);
        pipeline_move(real_hslice, 3);
        if (draining_idx < 1) {
            hidx[0] += real_hslice[0];
            if (hidx[0] >= input_shape[2]) {
                hidx[0] = 0;
                cidx[0] += real_cslice[0];
                if (cidx[0] >= input_shape[1]) {
                    draining_idx++;
                }
            }
        }
        else {
            draining_idx++;
        }
        stage_idx++;
    }
}

static void nodechip_reduce_onedim(
global_addr_t input_global_addr,
global_addr_t buffer_global_addr,
global_addr_t output_global_addr,
int *input_shape,
int shape_dims,
int *axis_list,
int method,
unsigned long long *buffer_size, // for bmcompiler alloc global mem buffer
data_type_t dtype) {
    TPUKERNEL_ASSERT(shape_dims == 4);
    const int axis = axis_list[0];
    if (axis == 2) {
        if (buffer_size) {
            return;
        }
        nodechip_reduce_h(input_global_addr,
                          output_global_addr,
                          input_shape,
                          shape_dims,   // 4
                          method,
                          dtype);
        return;
    }

    if (axis != 3) {
        // transpose to w
        unsigned long long ext_buffer_size = 0;
        int tensor_size = update_tensor_size(input_shape);
        int trans_order[4] = {0, 0, 0, 0};
        int trans_shape[4] = {1, 1, 1, 1};
        trans_order[3] = axis;
        for (int i = 0; i < shape_dims - 1; ++i) {
            trans_order[i] = i + (i >= axis);
        }
        for (int i = 0; i < shape_dims; ++i) {
            trans_shape[i] = input_shape[trans_order[i]];
        }
        nodechip_transpose(input_global_addr,
                           buffer_global_addr,
                           input_shape,
                           trans_order,
                           4,
                           buffer_global_addr + tensor_size,
                           buffer_size ? &ext_buffer_size : NULL,
                           dtype);
        int axis_num = 1;
        axis_list[0] = 3;
        process_shape(input_shape, axis_list, &axis_num, trans_shape, 4);
        if (buffer_size) {
            *buffer_size = MAX(*buffer_size, ext_buffer_size + tensor_size);
        }
    }
    if (buffer_size) {
        return;
    }
    nodechip_reduce_w(axis != 3 ? buffer_global_addr : input_global_addr,
                      output_global_addr,
                      input_shape,
                      4,
                      method,
                      dtype);
}

void nodechip_reduce_prod(
global_addr_t input_global_addr,
global_addr_t buffer_global_addr,
global_addr_t output_global_addr,
const int *input_shape_orig,
int shape_dims_orig,
int axis,
data_type_t dtype) {
    // remove 1 dims shape and reshape to 4-D
    int input_shape[MAX_SHAPE] = {0};
    int axis_num = 1;
    int dims = process_shape(input_shape,
                             &axis,
                             &axis_num,
                             input_shape_orig,
                             shape_dims_orig);
    if (dims < 4) {
        // call process_shape again to make dims = 4
        dims = process_shape(input_shape,
                             &axis,
                             &axis_num,
                             input_shape_orig,
                             shape_dims_orig);
    } else if (dims > 4) {
        // merge dims that are reduced and do transpose if needed - todo
        TPUKERNEL_ERR("Not implement this process yet.");
    }
    TPUKERNEL_ASSERT(dims == 4);
    TPUKERNEL_ASSERT(axis_num == 1);

    nodechip_reduce_onedim(input_global_addr,
                           buffer_global_addr,
                           output_global_addr,
                           input_shape,
                           4,
                           &axis,
                           REDUCE_PROD,
                           NULL,
                           dtype);
    
}
void tpu_kernel_api_reduce_prod(const void *args) {
    sg_api_reduce_prod_t *api = (sg_api_reduce_prod_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);

    tpu_initialize();
    nodechip_reduce_prod(api->input_global_addr,
                         api->buffer_global_addr,
                         api->output_global_addr,
                         api->shape,
                         api->dim,
                         api->axis,
                         api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_reduce_prod);

extern void nodechip_reduce_full(
global_addr_t input_global_addr,
global_addr_t buffer_global_addr,
global_addr_t output_global_addr,
const int *input_shape_orig,
const int *axis_list_orig,
int shape_dims_orig,
int axis_num,
int method,
unsigned long long *buffer_size,
data_type_t dtype);

void nodechip_sub_bcast(
global_addr_t input_global_addr,
global_addr_t other_global_addr,
global_addr_t output_global_addr,
int *input_shape_org,
int *output_shape_org,
int dim,
data_type_t dtype) {
    dim4 input_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 other_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 output_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    if(dim >= 1) {
        input_shape.n = input_shape_org[0];
        other_shape.n = output_shape_org[0];
        output_shape.n = input_shape.n > other_shape.n ? input_shape.n : other_shape.n;
    }
    if(dim >= 2) {
        input_shape.c = input_shape_org[1];
        other_shape.c = output_shape_org[1];
        output_shape.c = input_shape.c > other_shape.c ? input_shape.c : other_shape.c;
    }
    if(dim >= 3) {
        input_shape.h = input_shape_org[2];
        other_shape.h = output_shape_org[2];
        output_shape.h = input_shape.h > other_shape.h ? input_shape.h : other_shape.h;
    }
    if(dim >= 4) {
        input_shape.w = input_shape_org[3];
        other_shape.w = output_shape_org[3];
        output_shape.w = input_shape.w > other_shape.w ? input_shape.w : other_shape.w;
    }

    dim4 input_global_stride, other_global_stride, output_global_stride;
    tpu_continuous_stride(&input_global_stride,  &input_shape);
    tpu_continuous_stride(&other_global_stride,  &other_shape);
    tpu_continuous_stride(&output_global_stride, &output_shape);

    const bool input_bcast[4] = {
        input_shape.n != output_shape.n,
        input_shape.c != output_shape.c,
        input_shape.h != output_shape.h,
        input_shape.w != output_shape.w
    };
    const bool other_bcast[4] = {
        other_shape.n != output_shape.n,
        other_shape.c != output_shape.c,
        other_shape.h != output_shape.h,
        other_shape.w != output_shape.w
    };
    bool input_bcast_all = false, other_bcast_all = false;
    for(int i = 0; i < 4; ++i) {
        input_bcast_all = input_bcast_all || input_bcast[i];
        other_bcast_all = other_bcast_all || other_bcast[i];
    }

    const int c_per_npu = DIV_UP ( output_shape.c, NPU_NUM );
    int hmax = output_shape.h, nmax = output_shape.n, cmax = c_per_npu * NPU_NUM;
    local_addr_t output_addr, input_addr, other_addr;
    while(true) {
        output_addr = 0;
        int output_size = tpu_aligned_feature_size(hmax, output_shape.w, dtype) 
                          * DIV_UP(cmax, NPU_NUM) * nmax;
        input_addr = output_addr + output_size;
        int input_size = output_size;
        other_addr = input_addr + input_size;
        int other_size = output_size;
        int total_size = other_addr + other_size;
        if(total_size <= LOCAL_MEM_SIZE) {
            break;
        }
        else {
            if(cmax > NPU_NUM) {
                if(cmax % NPU_NUM == 0) {
                    cmax -= NPU_NUM;
                }
                else {
                    cmax -= (cmax % NPU_NUM);
                }
                continue;
            }
            else if(nmax > 1) {
                nmax /= 2;
                continue;
            }
            else if(hmax > 1) {
                hmax /= 2;
                continue;
            }
            else {
                TPUKERNEL_ASSERT ( false );
            }
        }
    }
    dim4 shape = { .w = output_shape.w };
    dim4 input_local_shape, other_local_shape;
    dim4 input_local_stride, other_local_stride;
    int ctodo = output_shape.c, cdone = 0;
    while(ctodo > 0) {
        shape.c = MIN(ctodo, cmax);
        int ntodo = output_shape.n, ndone = 0;
        while(ntodo > 0) {
            shape.n = MIN(ntodo, nmax);
            int htodo = output_shape.h, hdone = 0;
            while(htodo > 0) {
                shape.h = MIN (htodo, hmax);
                // Move input from global memory to local memory
                tpu_aligned_stride (&input_local_stride, 0, &shape, dtype);
                input_local_shape.n = input_bcast[0] ? 1 : shape.n;
                input_local_shape.c = input_bcast[1] ? 1 : shape.c;
                input_local_shape.h = input_bcast[2] ? 1 : shape.h;
                input_local_shape.w = input_bcast[3] ? 1 : shape.w;
                global_addr_t input_global_addr_gdma = input_global_addr +
                                ((input_bcast[0] ? 0 : ndone) * input_global_stride.n +
                                 (input_bcast[1] ? 0 : cdone) * input_global_stride.c +
                                 (input_bcast[2] ? 0 : hdone) * input_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_S2L(input_addr, input_global_addr_gdma, &input_local_shape,
                                 &input_local_stride, &input_global_stride, dtype);
                
                // Move other from global memory to local memory
                tpu_aligned_stride(&other_local_stride, 0, &shape, dtype);
                other_local_shape.n = other_bcast[0] ? 1 : shape.n;
                other_local_shape.c = other_bcast[1] ? 1 : shape.c;
                other_local_shape.h = other_bcast[2] ? 1 : shape.h;
                other_local_shape.w = other_bcast[3] ? 1 : shape.w;
                global_addr_t other_global_addr_gdma = other_global_addr +
                                ((other_bcast[0] ? 0 : ndone) * other_global_stride.n +
                                 (other_bcast[1] ? 0 : cdone) * other_global_stride.c +
                                 (other_bcast[2] ? 0 : hdone) * other_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_S2L(other_addr, other_global_addr_gdma, &other_local_shape,
                                 &other_local_stride, &other_global_stride, dtype);
                
                // Broadcast input if needed
                if(input_bcast[1]) {
                    input_local_shape.c = NPU_NUM;
                    tpu_bdc_npu_bcast(input_addr, input_addr, &input_local_shape, dtype);
                }
                if(input_bcast[0] || input_bcast[2] || input_bcast[3] || (input_bcast[1] && shape.c > NPU_NUM)) {
                    dim4 input_bcast_stride;
                    input_bcast_stride.n = input_bcast[0] ? 0 : input_local_stride.n;
                    input_bcast_stride.c = input_bcast[1] ? 0 : input_local_stride.c;
                    input_bcast_stride.h = input_bcast[2] ? 0 : input_local_stride.h;
                    input_bcast_stride.w = input_bcast[3] ? 0 : input_local_stride.w;
                    tpu_bdc_cpy(input_addr, input_addr, &shape, NULL, &input_bcast_stride, dtype);
                }
                
                // Broadcast other if needed
                if(other_bcast[1]) {
                    other_local_shape.c = NPU_NUM;
                    tpu_bdc_npu_bcast(other_addr, other_addr, &other_local_shape, dtype);
                }
                if(other_bcast[0] || other_bcast[2] || other_bcast[3] || (other_bcast[1] && shape.c > NPU_NUM)) {
                    dim4 other_bcast_stride;
                    other_bcast_stride.n = other_bcast[0] ? 0 : other_local_stride.n;
                    other_bcast_stride.c = other_bcast[1] ? 0 : other_local_stride.c;
                    other_bcast_stride.h = other_bcast[2] ? 0 : other_local_stride.h;
                    other_bcast_stride.w = other_bcast[3] ? 0 : other_local_stride.w;
                    tpu_bdc_cpy ( other_addr, other_addr, &shape, NULL, &other_bcast_stride, dtype );
                }
                
                // Select
                tpu_bdc_fp_sub(output_addr, input_addr, other_addr, &shape, NULL, NULL, NULL, dtype);
                
                // Move out from local memory to global memory
                global_addr_t output_global_addr_gdma = 
                                            output_global_addr +
                                            (ndone * output_global_stride.n +
                                             cdone * output_global_stride.c +
                                             hdone * output_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_L2S(output_global_addr_gdma, output_addr, &shape,
                                 &output_global_stride, NULL, dtype);
                htodo -= shape.h;
                hdone += shape.h;
            }
            ntodo -= shape.n;
            ndone += shape.n;
        }
        ctodo -= shape.c;
        cdone += shape.c;
    }
}

void nodechip_compute(
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int *input_shape,
int dim,
float scalar,
int mode,
data_type_t dtype) {
    int length = 1;
    for(int i = 0 ; i < dim; ++i) {
        length *= input_shape[i];
    }
    if(0 == length) return;

    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    const int tensor_num = 2 + 2;     // 2 inputs, 2 outputs
    int tensor_size = LOCAL_MEM_BANKS / tensor_num * bank_size;
    TPUKERNEL_ASSERT(tensor_size > 0);

    const int dtype_size = tpu_data_type_size(dtype);
    local_addr_t input_local_addrs[] = {0, tensor_size};
    local_addr_t output_local_addrs[] = {2 * tensor_size, 3 * tensor_size};
    int tensor_w = DIV_UP(MIN(length, tensor_size * NPU_NUM / dtype_size), NPU_NUM);

    scalar_t sub_scalar = {.f32 = scalar};
    int todo = length, done = 0;
    dim4 shape = {.n = 1, .h = 1};
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    while(todo != 0) {
        if(todo > NPU_NUM){
            shape.c = NPU_NUM;
            shape.w = MIN(todo / NPU_NUM, tensor_w);
        }
        else {
            shape.c = todo;
            shape.w = 1;
        }

        tpu_gdma_cpy_S2L(input_local_addrs[index], input_global_addr + done * dtype_size,
                         &shape, NULL, NULL, dtype);

        if(tpu_is_parallel_state()) {
            tpu_parallel_end();
        }
        tpu_parallel_start();

        if(l2s) {
            tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
        }

        if(mode) {
            tpu_bdc_fp_mul(output_local_addrs[index], input_local_addrs[index],
                           input_local_addrs[index], &shape, NULL, NULL, NULL, dtype);
        }
        else {
            tpu_bdc_fp_sub_C(output_local_addrs[index], input_local_addrs[index],
                             sub_scalar, &shape, NULL, NULL, dtype);
        }

        l2s = true;
        l2s_global_addr = output_global_addr + done * dtype_size;
        l2s_local_addr = output_local_addrs[index];
        l2s_shape = shape;
        todo -= shape.c * shape.w;
        done += shape.c * shape.w;
        index = 1- index;
    }
    if(tpu_is_parallel_state()) {
        tpu_parallel_end();
    }
    if(l2s) {
        tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
    }
}

void nodechip_div_n(
global_addr_t input_global_addr,
global_addr_t output_global_addr,
int *input_shape,
int dim,
float scalar,
data_type_t dtype) {
    int length = 1;
    for(int i = 0 ; i < dim; ++i) {
        length *= input_shape[i];
    }
    if(0 == length) return;

    const int bank_size = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS;
    const int tensor_num = 2 + 2;     // 2 inputs, 2 outputs
    int tensor_size = LOCAL_MEM_BANKS / tensor_num * bank_size;
    TPUKERNEL_ASSERT(tensor_size > 0);

    const int dtype_size = tpu_data_type_size(dtype);
    local_addr_t input_local_addrs[] = {0, tensor_size};
    local_addr_t output_local_addrs[] = {2 * tensor_size, 3 * tensor_size};
    int tensor_w = DIV_UP(MIN(length, tensor_size * NPU_NUM / dtype_size), NPU_NUM);

    int todo = length, done = 0;
    dim4 shape = {.n = 1, .h = 1};
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    while(todo != 0) {
        if(todo > NPU_NUM){
            shape.c = NPU_NUM;
            shape.w = MIN(todo / NPU_NUM, tensor_w);
        }
        else {
            shape.c = todo;
            shape.w = 1;
        }

        tpu_gdma_cpy_S2L(input_local_addrs[index], input_global_addr + done * dtype_size,
                         &shape, NULL, NULL, dtype);

        if(tpu_is_parallel_state()) {
            tpu_parallel_end();
        }
        tpu_parallel_start();

        if(l2s) {
            tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
        }

        tpu_bdc_fp32_div_C(output_local_addrs[index], input_local_addrs[index], scalar, &shape, NULL, NULL);

        l2s = true;
        l2s_global_addr = output_global_addr + done * dtype_size;
        l2s_local_addr = output_local_addrs[index];
        l2s_shape = shape;
        todo -= shape.c * shape.w;
        done += shape.c * shape.w;
        index = 1- index;
    }
    if(tpu_is_parallel_state()) {
        tpu_parallel_end();
    }
    if(l2s) {
        tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
    }
}

void nodechip_reduce_var(
global_addr_t input_global_addr,
global_addr_t buffer_global_addr,
global_addr_t mul_global_addr,
global_addr_t sum_global_addr,
global_addr_t output_global_addr,
int *input_shape_org,
int input_dim,
int *output_shape_org,
int output_dim,
int *reduce_list,
int reduce_dim,
int correction,
data_type_t dtype) {
    // reduce mean
    nodechip_reduce_full(input_global_addr, buffer_global_addr, mul_global_addr, input_shape_org,
                         reduce_list, input_dim, reduce_dim, REDUCE_MEAN, NULL, dtype);

    // bcast sub
    nodechip_sub_bcast(input_global_addr, mul_global_addr, buffer_global_addr,
                       input_shape_org, output_shape_org, input_dim, dtype);
    
    // 1 square
    nodechip_compute(buffer_global_addr, mul_global_addr, input_shape_org, input_dim, 0., 1, dtype);

    // reduce sum
    nodechip_reduce_full(mul_global_addr, buffer_global_addr, sum_global_addr, input_shape_org,
                         reduce_list, input_dim, reduce_dim, REDUCE_SUM, NULL, dtype);

    // div (n - correction)
    int n = 1;
    for(int i = 0; i < reduce_dim; ++i) {
        n *= input_shape_org[reduce_list[i]];
    }
    n -= correction;
    int output_shape[output_dim];
    int idx = 0;
    for(int i = 0; i < input_dim; ++i) {
        if(output_shape_org[i] > 1) {
            output_shape[idx++] = output_shape_org[i];
        }
    }
    nodechip_div_n(sum_global_addr, output_global_addr, output_shape, output_dim, (float)n, dtype);
}
void tpu_kernel_api_reduce_var(const void *args) {
    sg_api_reduce_var_t *api = (sg_api_reduce_var_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32);

    tpu_initialize();
    nodechip_reduce_var(api->input_global_addr, api->buffer_global_addr, api->mul_global_addr,
                        api->sum_global_addr, api->output_global_addr, api->input_shape,
                        api->input_dim, api->output_shape, api->output_dim, api->reduce_list,
                        api->reduce_dim, api->correction, api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_reduce_var);

void nodechip_reduce_var_all(
global_addr_t input_global_addr,
global_addr_t buffer_global_addr,
global_addr_t mul_global_addr,
global_addr_t output_global_addr,
int *shape,
int dim,
int correction,
data_type_t dtype) {
    int axis_list[dim];
    for(int i = 0 ; i < dim; ++i) {
        axis_list[i] = i;
    }

    // reduce mean
    nodechip_reduce_full(input_global_addr, buffer_global_addr, mul_global_addr, shape,
                         axis_list, dim, dim, REDUCE_MEAN, NULL, dtype);

    // 0 sub c
    float *scalar = tpu_global_mem_addr(mul_global_addr);
    nodechip_compute(input_global_addr, buffer_global_addr, shape, dim, scalar[0], 0, dtype);

    // 1 square
    nodechip_compute(buffer_global_addr, mul_global_addr, shape, dim, 0., 1, dtype);

    // reduce sum
    nodechip_reduce_full(mul_global_addr, buffer_global_addr, buffer_global_addr, shape,
                         axis_list, dim, dim, REDUCE_SUM, NULL, dtype);

    // div (n - correction)
    int n = 1;
    for(int i = 0; i < dim; ++i) {
        n *= shape[i];
    }
    float *sum = tpu_global_mem_addr(buffer_global_addr);
    float *out = tpu_global_mem_addr(output_global_addr);
    out[0] = sum[0] / (n - correction);
}
void tpu_kernel_api_reduce_var_all(const void *args) {
    sg_api_reduce_var_t *api = (sg_api_reduce_var_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32);

    tpu_initialize();
    nodechip_reduce_var_all(api->input_global_addr, api->buffer_global_addr,api->mul_global_addr,
                            api->output_global_addr, api->input_shape, api->input_dim,
                            api->correction, api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_reduce_var_all);