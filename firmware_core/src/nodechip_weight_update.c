#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = input + value * ( other )
 */
static void tpu_bdc_addcmul(local_addr_t dst_addr,
                            local_addr_t src_addr,
                            local_addr_t other_addr,
                            scalar_t     value,
                            const dim4   *shape,
                            const dim4   *dst_stride,
                            const dim4   *src0_stride,
                            const dim4   *src1_stride,
                            data_type_t  dtype)
{
  tpu_bdc_fp_mul_C(other_addr, other_addr, value, shape, src1_stride, src1_stride, dtype);
  tpu_bdc_fp_add(dst_addr, src_addr, other_addr, shape, dst_stride, src0_stride, src1_stride, dtype);
}
typedef void (*binary_fp_func)(local_addr_t, local_addr_t, local_addr_t, scalar_t,
                                     const dim4 *, const dim4 *, const dim4 *, const dim4 *,
                                     data_type_t);
static binary_fp_func get_binary_func(int binary_type)
{
  binary_fp_func func = NULL;
  if (binary_type == BINARY_ADDCMUL) func = tpu_bdc_addcmul;
  else {
    TPUKERNEL_ASSERT(false);
  }
  return func;
}

static inline bool is_local_mem_enough(
    const dim4       *grad_shape,
    data_type_t      dtype) {
  // grad_shape : [ic, kh*kw, DIVUP(oc,32), 32]
  int ic = grad_shape->n;
  int ic_per_npu = DIV_UP(ic, NPU_NUM);
  // [1, ic, kh*kw, DIVUP(oc,32)*32]
  int grad_size = ALIGN(1 * ic_per_npu * tpu_aligned_feature_size(grad_shape->c,
                                    grad_shape->h * grad_shape->w, dtype), BANK_SIZE);
  int _32OC_weight_size = grad_size; 
  return (grad_size + _32OC_weight_size) * 2 <= LOCAL_MEM_SIZE;
}
typedef struct conv_weight_update_secs_info_t
{
  int ic_secs;
  int oc_secs;
} CW_SEC_INFO;

static inline void split_ic_or_oc(
    const dim4            *grad_shape,
    data_type_t           dtype,
    CW_SEC_INFO           *slice_info)
{
  // grad_shape : [ic, kh*kw, DIVUP(oc,32), 32]
  bool valid;
  bool can_split_ic = grad_shape->n > NPU_NUM;
  bool can_split_oc = grad_shape->h > 1;
  dim4 slice_shape = {.n = grad_shape->n, .c = grad_shape->c, .h = grad_shape->h, .w = grad_shape->w};
  do {
    valid = is_local_mem_enough(&slice_shape, dtype);
    if (valid) break;
    if (can_split_ic) {
      slice_shape.n -= NPU_NUM;
      can_split_ic = slice_shape.n > NPU_NUM;
    } else if (can_split_oc) {
      slice_shape.h -= 1;
      can_split_oc = slice_shape.h > 1;
    } else {
      TPUKERNEL_ASSERT(false);
    }
  } while(!valid);
  slice_info->ic_secs = 1 + (grad_shape->n - slice_shape.n) / slice_shape.n;
  slice_info->oc_secs = 1 + (grad_shape->h - slice_shape.h) / slice_shape.h;
}

// weight: 32ic32oc
// [oc, kh*kw, DIVUP(ic, 32), 32]
// [ic, kh*kw, DIVUP(oc, 32), 32]
// grad: 32oc
// [ic, kh*kw, DIVUP(oc,32), 32]
// ========= the main problem: 32OC -> 32IC? HOW TO SLOVE? ======== //
// move in : [1, ic, kh*kw, DIVUP(oc,32)*32]
// cw_trans: [1, DIVUP(oc,32)*32, kh*kw, ic]
// view as : [1, oc, kh*kw, ic]
// stride copy with global stride = [.w = 1,
//                                   .h = divup(ic, 32)*32,
//                                   .c = kh*kw*divup(ic, 32)*32,
//                                   .n = oc* kh*kw*divup(ic, 32)*32]
// ================================================================ //
int nodechip_conv_32ic32oc_weight_32oc_grad_update(
    global_addr_t weight_global_addr,
    global_addr_t grad_global_addr,
    global_addr_t output_global_addr,
    const dim4    *grad_shape,
    const dim4    *weight_32ic_shape,
    scalar_t      value,
    data_type_t   dtype,
    int           binary_type )
{
  binary_fp_func binary_func = get_binary_func(binary_type);
  int ic = grad_shape->n;
  int oc = weight_32ic_shape->n;
  int khxkw = grad_shape->c;
  int aligned_oc = grad_shape->h * grad_shape->w;
  int aligned_ic = weight_32ic_shape->h * weight_32ic_shape->w;

  unsigned long long _32ic_weight_global_offset = oc * khxkw * aligned_ic * tpu_data_type_size(dtype);

  CW_SEC_INFO slice_info;
  split_ic_or_oc(grad_shape, dtype, &slice_info);
  int ic_per_npu = DIV_UP((ic / slice_info.ic_secs), NPU_NUM);
  int grad_size = ic_per_npu * tpu_aligned_feature_size(khxkw, aligned_oc / slice_info.oc_secs, dtype);
  local_addr_t grad_addr_ping         = 0;
  local_addr_t grad_addr_pong         = grad_addr_ping + grad_size;
  local_addr_t _32OC_weight_addr_ping = grad_addr_pong + grad_size;
  local_addr_t _32OC_weight_addr_pong = _32OC_weight_addr_ping + grad_size;
  
  dim4 grad_global_stride = {ic * khxkw * aligned_oc, khxkw * aligned_oc, aligned_oc, 1};
  dim4 weight_32oc_global_stride = grad_global_stride;
  dim4 weight_32ic_global_stride = {oc * khxkw * aligned_ic, khxkw * aligned_ic , aligned_ic, 1};

  bool ping = true;
  bool parallel_branch = false;
  int last_ocstart, last_ocend, last_icstart, last_icend;

  int oc_start = 0; int oc_end = 0;
  for (int idx_oc = 0; idx_oc < slice_info.oc_secs; idx_oc++)
  {
    oc_start = oc_end;
    oc_end   = MIN(oc, oc_start + oc / slice_info.oc_secs);
    int ic_start = 0; int ic_end = 0;
    for (int idx_ic = 0; idx_ic < slice_info.ic_secs; idx_ic++)
    {
      ic_start = ic_end;
      ic_end = MIN(ic, ic_start + ic / slice_info.ic_secs);
      dim4 grad_shape = {1, ic_end - ic_start, khxkw, oc_end - oc_start};
      dim4 weight_32oc_shape = grad_shape;
      // [1, ic, kh*kw, DIVUP(oc,32)*32]
      tpu_gdma_cpy_S2L(
        ping ? grad_addr_ping : grad_addr_pong,
        grad_global_addr + ( ic_start * khxkw * aligned_oc + oc_start) * tpu_data_type_size(dtype),
        &grad_shape,
        NULL,
        &grad_global_stride,
        dtype);
      tpu_gdma_cpy_S2L(
        ping ? _32OC_weight_addr_ping : _32OC_weight_addr_pong,
        weight_global_addr + _32ic_weight_global_offset + 
                          ( ic_start * khxkw * aligned_oc + oc_start) * tpu_data_type_size(dtype),
        &weight_32oc_shape,
        NULL,
        &weight_32oc_global_stride,
        dtype);
      if (parallel_branch) { tpu_parallel_end(); }
      tpu_parallel_start();
      parallel_branch = true;
      // 32oc weight update
      binary_func(
        ping ? _32OC_weight_addr_ping : _32OC_weight_addr_pong,
        ping ? _32OC_weight_addr_ping : _32OC_weight_addr_pong,
        ping ? grad_addr_ping : grad_addr_pong,
        value,
        &grad_shape,
        NULL, NULL, NULL, dtype);
      if (idx_ic > 0 || idx_oc > 0) {
        dim4 last_32OC_weight_shape = {1, last_icend - last_icstart, khxkw, last_ocend - last_ocstart};
        tpu_gdma_cpy_L2S(
          weight_global_addr + _32ic_weight_global_offset + 
                          ( last_icstart * khxkw * aligned_oc + last_ocstart) * tpu_data_type_size(dtype),
          ping ? _32OC_weight_addr_pong : _32OC_weight_addr_ping,
          &last_32OC_weight_shape,
          &weight_32oc_global_stride,
          NULL, dtype);
        dim4 last_32IC_weight_shape = {1, last_ocend - last_ocstart, khxkw, last_icend - last_icstart};
        // [1, oc, kh*kw, DIVUP(ic, 32)*32]
        tpu_gdma_cpy_cw_trans_L2S(
          weight_global_addr + (last_ocstart * khxkw * aligned_ic + last_icstart) * tpu_data_type_size(dtype), // TODO
          ping ? _32OC_weight_addr_pong : _32OC_weight_addr_ping,
          &last_32IC_weight_shape,
          &weight_32ic_global_stride,
          NULL, dtype);
      }
      ping = !ping;
      last_icstart = ic_start;
      last_icend   = ic_end;
      last_ocstart = oc_start;
      last_ocend   = oc_end;
    }
  } // ic_
  tpu_parallel_end();

  dim4 last_32OC_weight_shape = {1, last_icend - last_icstart, khxkw, last_ocend - last_ocstart};
  tpu_gdma_cpy_L2S(
    weight_global_addr + _32ic_weight_global_offset + 
                          ( last_icstart * khxkw * aligned_oc + last_ocstart) * tpu_data_type_size(dtype), //TODO
    ping ? _32OC_weight_addr_pong : _32OC_weight_addr_ping,
    &last_32OC_weight_shape,
    &weight_32oc_global_stride,
    NULL, dtype);
  dim4 last_32IC_weight_shape = {1, last_ocend - last_ocstart, khxkw, last_icend - last_icstart};
  tpu_gdma_cpy_cw_trans_L2S(
    weight_global_addr + (last_ocstart * khxkw * aligned_ic + last_icstart) * tpu_data_type_size(dtype), // TODO
    ping ? _32OC_weight_addr_pong : _32OC_weight_addr_ping,
    &last_32IC_weight_shape,
    &weight_32ic_global_stride,
    NULL, dtype);

  return 0;
}

int tpu_kernel_api_weight_update(const void *args) {
  sg_api_weight_update_t *api = (sg_api_weight_update_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP16 || api->dtype == DT_BFP16);
  dim4 grad_shape = {.n = api->other_shape[0], .c = api->other_shape[1],
                     .h = api->other_shape[2], .w = api->other_shape[3]};
  dim4 weight_32ic_shape = {.n = api->in_shape[0], .c = api->in_shape[1],
                     .h = api->other_shape[2], .w = api->other_shape[3]};
  scalar_t value, value_f32 = {.f32 = api->value};
  value = tpu_fp_cast(value_f32, (data_type_t)api->dtype, DT_FP32, RM_HALF_TO_EVEN);
  tpu_initialize();
  nodechip_conv_32ic32oc_weight_32oc_grad_update(
      api->input_global_addr,
      api->other_global_addr,
      api->output_global_addr,
      &grad_shape,
      &weight_32ic_shape,
      value,
      (data_type_t)api->dtype,
      api->binary_type);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_weight_update);