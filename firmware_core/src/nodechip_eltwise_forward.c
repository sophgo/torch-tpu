#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"

#define ELTWISE_OP_DOT_MUL 0
#define ELTWISE_OP_WEIGHTED_ADD 1
#define ELTWISE_OP_MAX 2

void nodechip_eltwise_twoinput_cal_forward(
    global_addr_t   bottom_A_global_offset,
    global_addr_t   bottom_B_global_offset,
    global_addr_t   top_global_offset,
    global_addr_t   mask_global_offset,
    int             tensor_n,
    int             tensor_c,
    int             tensor_h,
    int             tensor_w,
    int             op_code,
    scalar_t        coeff_A,
    scalar_t        coeff_B,
    int             need_mask,
    scalar_t        mask_index,
    int             is_first,
    int             if_relu,
    data_type_t     dtype
  ) {

    int real_need_mask = (op_code == ELTWISE_OP_MAX) && need_mask;
    int need_num = 2 + real_need_mask * 3;
    if (is_first) need_num = 1 + real_need_mask;
    int total_mem = LOCAL_MEM_BANKS/need_num * (LOCAL_MEM_SIZE/LOCAL_MEM_BANKS);
    local_addr_t bottom_A_local_offset = 0;
    local_addr_t bottom_B_local_offset = 1 * total_mem;
    local_addr_t mask_in_local_offset =  2 * total_mem;
    local_addr_t mask_out_local_offset = 3 * total_mem;
    local_addr_t top_local_offset = bottom_A_local_offset;
    if (real_need_mask) {
        top_local_offset = 4 * total_mem;
    }
    if (is_first) {
        top_local_offset = bottom_A_local_offset;
        mask_in_local_offset = 1 * total_mem;
        mask_out_local_offset = mask_in_local_offset;
    }

    int left_count = tensor_n * tensor_c * tensor_h * tensor_w;
    int row = tensor_n;
    int sec_len = tpu_eu_num(dtype);
    int col = 0;

    int real_n = tensor_n;
    int real_c = tensor_c;
    int real_h = tensor_h;
    int real_w = tensor_w;
    int process_count = 0;

    bool coeff_A_is1 = false;
    if (dtype == DT_FP32) {
        if (coeff_A.f32 == 1.0) coeff_A_is1 = true;
    } else if (dtype == DT_FP16) {
        if (coeff_A.f16.bits == 0x3c00) coeff_A_is1 = true;
    } else if (dtype == DT_BFP16) {
        if (coeff_A.bf16.bits == 0x3f80) coeff_A_is1 = true;
    }

    bool coeff_B_is1 = false;
    if (dtype == DT_FP32) {
        if (coeff_B.f32 == 1.0) coeff_B_is1 = true;
    } else if (dtype == DT_FP16) {
        if (coeff_B.f16.bits == 0x3c00) coeff_B_is1 = true;
    } else if (dtype == DT_BFP16) {
        if (coeff_B.bf16.bits == 0x3f80) coeff_B_is1 = true;
    }

    bool coeff_B_is0 = false;
    if (dtype == DT_FP32) {
        if (coeff_B.f32 == 0.0) coeff_B_is0 = true;
    } else if (dtype == DT_FP16) {
        if (coeff_B.f16.bits == 0x0000) coeff_B_is0 = true;
    } else if (dtype == DT_BFP16) {
        if (coeff_B.bf16.bits == 0x0000) coeff_B_is0 = true;
    }

    unsigned long long global_step = 0;
    dim4 local_stride;
    dim4 tensor_dim;
    const int COL_MAX = 65536;
    while(left_count > 0) {
        if (left_count >= total_mem) {
            col = NPU_NUM * sec_len;
            row = total_mem/(sec_len * (dtype == DT_FP32 ? 4 : 2));
        } else if (left_count > COL_MAX) {
            col = COL_MAX;
            row = 1;
        } else {
            col = left_count;
            row = 1;
        }
        real_n = row;
        real_h = 1;
        real_c = (col + sec_len - 1)/sec_len;
        real_w = sec_len;

        local_stride.w = 1;
        local_stride.h = real_w;
        local_stride.c = real_w * real_h;
        local_stride.n = (real_c + NPU_NUM - 1)/NPU_NUM * local_stride.c;

        tensor_dim.w = real_w;
        tensor_dim.h = real_h;
        tensor_dim.c = real_c;
        tensor_dim.n = real_n;

        // load bottom A , so top = bottomA
        tpu_gdma_cpy_S2L(bottom_A_local_offset,
                         bottom_A_global_offset + global_step,
                         &tensor_dim,
                         &local_stride,
                         NULL,
                         dtype);

        if (is_first) {
            // assign bottom_A to top
            if (op_code == ELTWISE_OP_WEIGHTED_ADD && !coeff_A_is1) {
                // compute A * a_coeff
                tpu_bdc_fp_mul_C(top_local_offset,
                                 bottom_A_local_offset,
                                 coeff_A,
                                 &tensor_dim,
                                 &local_stride,
                                 &local_stride,
                                 dtype);
            }
            if (real_need_mask) {
                tpu_bdc_set_C(mask_out_local_offset,
                              mask_index,
                              &tensor_dim,
                              NULL, //check
                              dtype);
            }
        } else {
            // load bottom B
            // load_bottom A, so top = bottomA
            tpu_gdma_cpy_S2L(bottom_B_local_offset,
                             bottom_B_global_offset + global_step,
                             &tensor_dim,
                             &local_stride,
                             NULL,
                             dtype);
            if (real_need_mask) {
                // load mask data
                tpu_gdma_cpy_S2L(mask_in_local_offset,
                                 mask_global_offset + global_step,
                                 &tensor_dim,
                                 &local_stride,
                                 NULL,
                                 dtype);
            }
            if (op_code == ELTWISE_OP_DOT_MUL) {
                tpu_bdc_fp_mul(top_local_offset,
                               bottom_A_local_offset,
                               bottom_B_local_offset,
                               &tensor_dim,
                               &local_stride,
                               &local_stride,
                               &local_stride,
                               dtype);
            } else if (op_code == ELTWISE_OP_WEIGHTED_ADD) {
                if (!coeff_A_is1)
                    // compute A * a_coeff
                    tpu_bdc_fp_mul_C(bottom_A_local_offset,
                                     bottom_A_local_offset,
                                     coeff_A,
                                     &tensor_dim,
                                     &local_stride,
                                     &local_stride,
                                     dtype);
                if (!coeff_B_is1)
                    // compute B * b_coeff
                    tpu_bdc_fp_mul_C(bottom_B_local_offset,
                                     bottom_B_local_offset,
                                     coeff_B,
                                     &tensor_dim,
                                     &local_stride,
                                     &local_stride,
                                     dtype);
                // top_local_offset is equal to bottom_A_local_offset
                if (!coeff_B_is0)
                    tpu_bdc_fp_add(top_local_offset,
                                   bottom_A_local_offset,
                                   bottom_B_local_offset,
                                   &tensor_dim,
                                   &local_stride,
                                   &local_stride,
                                   &local_stride,
                                   dtype);
            } else if (op_code == ELTWISE_OP_MAX) {
                // compute max(A, B)
                if (!need_mask) {
                    tpu_bdc_max(top_local_offset,
                                bottom_B_local_offset,
                                bottom_A_local_offset,
                                &tensor_dim,
                                &local_stride,
                                &local_stride,
                                &local_stride,
                                dtype);
                } else {
                    // calculate top = bottomA > bottomB ? bottomA : bottomB,
                    //          mask = bottomA > bottomB ? mask_index : mask
                    variable_t src0, src1, src2, src3;
                    src0.type = TENSOR;
                    src0.context.addr = bottom_B_local_offset;
                    src1.type = TENSOR;
                    src1.context.addr = bottom_A_local_offset;
                    src2.type = SCALAR;
                    src2.context.scalar = mask_index;
                    src3.type = TENSOR;
                    src3.context.addr = mask_in_local_offset;
                    tpu_bdc_maximum_greater_select(top_local_offset,
                                                   mask_out_local_offset,
                                                   &src0,
                                                   &src1,
                                                   &src2,
                                                   &src3,
                                                   &tensor_dim,
                                                   dtype,
                                                   dtype);
                }
            } // calculate by op_code
            if (if_relu) {
                tpu_bdc_max_C(top_local_offset,
                              top_local_offset,
                              (scalar_t)0,
                              &tensor_dim,
                              &local_stride,
                              &local_stride,
                              dtype);
            } // if_relu
        } // if !is_first

        // save top_data
        tpu_gdma_cpy_L2S(top_global_offset + global_step,
                         top_local_offset,
                         &tensor_dim,
                         NULL,
                         &local_stride,
                         dtype);

        // save mask data
        if (real_need_mask) {
            tpu_gdma_cpy_L2S(mask_global_offset + global_step,
                             mask_out_local_offset,
                             &tensor_dim,
                             NULL,
                             &local_stride,
                             dtype);

        } // for real_need_mask
        process_count = row * col;
        left_count -= process_count;
        global_step += process_count * (dtype == DT_FP32 ? 4 : 2);
    } // while
}

void nodechip_eltwise_forward(
    global_addr_t*  bottom_global_offset,
    global_addr_t   top_global_offset,
    global_addr_t   mask_global_offset,
    int             input_num,
    int             tensor_n,
    int             tensor_c,
    int             tensor_h,
    int             tensor_w,
    int             op_code, 
    int*            coeff,
    int             need_mask,
    int*            mask_index,
    int             if_relu,
    data_type_t     dtype
  ) {
    if (op_code == ELTWISE_OP_MAX) {
        for (int idx = 0; idx < input_num; ++idx) {
          global_addr_t bottom_A_addr, bottom_B_addr;
          scalar_t      bottom_A_coeff, bottom_B_coeff;
          if (idx == 0) {
              bottom_B_addr = bottom_global_offset[0];
              bottom_B_coeff.s32 = coeff[0];
          } else {
              bottom_B_addr = top_global_offset;
              if (dtype == DT_FP32) bottom_B_coeff.f32 = 1.0;
              else if (dtype == DT_FP16) bottom_B_coeff.f16.bits = 0x3c00;
              else if (dtype == DT_BFP16) bottom_B_coeff.bf16.bits = 0x3f80;
          }
          bottom_A_addr = bottom_global_offset[idx];
          bottom_A_coeff.s32 = coeff[idx];
          scalar_t mask_index_t;
          mask_index_t.s32 = mask_index[idx];
          nodechip_eltwise_twoinput_cal_forward(
              bottom_A_addr,
              bottom_B_addr,
              top_global_offset,
              mask_global_offset,
              tensor_n,
              tensor_c,
              tensor_h,
              tensor_w,
              op_code,
              bottom_A_coeff,
              bottom_B_coeff,
              need_mask,
              mask_index_t,
              idx == 0 ? 1 : 0,
              if_relu,
              dtype);
        }
    } else {
        for (int idx = 0; idx < input_num - 1; ++idx) {
            global_addr_t bottom_A_addr, bottom_B_addr;
            scalar_t      bottom_A_coeff, bottom_B_coeff;
            if (idx == 0) {
                bottom_A_addr = bottom_global_offset[0];
                bottom_A_coeff.s32 = coeff[0];
            } else {
                bottom_A_addr = top_global_offset;
                if (dtype == DT_FP32) bottom_B_coeff.f32 = 1.0;
                else if (dtype == DT_FP16) bottom_B_coeff.f16.bits = 0x3c00;
                else if (dtype == DT_BFP16) bottom_B_coeff.bf16.bits = 0x3f80;
            }

            bottom_B_addr = bottom_global_offset[idx + 1];
            bottom_B_coeff.s32 = coeff[idx + 1];
            scalar_t mask_index_t;
            mask_index_t.s32 = mask_index[idx];
            nodechip_eltwise_twoinput_cal_forward(
                bottom_A_addr,
                bottom_B_addr,
                top_global_offset,
                mask_global_offset,
                tensor_n,
                tensor_c,
                tensor_h,
                tensor_w,
                op_code,
                bottom_A_coeff,
                bottom_B_coeff,
                need_mask,
                mask_index_t,
                0,  // (0 == idx && op_code==ELTWISE_OP_MAX) ? 1 : 0
                if_relu,
                dtype);
        }
    }
}

void tpu_kernel_api_eltwise_forward(const void* args) {
    sg_api_eltwise_forward_t *api = (sg_api_eltwise_forward_t *)args;

    tpu_initialize();
    global_addr_t bottom_global_offset[2] = {api->inputA_global_addr,
                                             api->inputB_global_addr};
    int coeff[2] = {api->coeff_A, api->coeff_B};
    int mask_index[2] = {api->mask_index_A, api->mask_index_B};

    data_type_t idtype = api->idtype;
    data_type_t odtype = api->odtype;
    TPUKERNEL_ASSERT(idtype == odtype);
    data_type_t dtype = idtype;

    nodechip_eltwise_forward(
        bottom_global_offset,
        api->output_global_addr,
        api->mask_global_addr,
        2,
        api->tensor_n,
        api->tensor_c,
        api->tensor_h,
        api->tensor_w,
        api->op_code,
        coeff,
        api->need_mask,
        mask_index,
        api->if_relu,
        dtype);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_eltwise_forward);
