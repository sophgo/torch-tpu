#include "nodechip_active.h"
#include "nodechip_active_local.h"
#include "sg_api_struct.h"

#define pipeline_move(array, num) do { \
  for (int i = (int)num - 1; i > 0; i--) { \
    array[i] = array[i - 1];\
  }\
} while(0)

#define _MAX(x, y) (((int)x) > ((int)y) ? ((int)x) : ((int)y))

static void gelu_local_perbatch(
    local_addr_t in_addr,
    local_addr_t out_addr,
    local_addr_t buffer_addr,
    const dim4* shape,
    data_type_t dtype,
    sg_active_type_t active_type,
    int if_local_layer,
    float* coef)
{
  dim4 stride = {0};
  tpu_aligned_stride(&stride, 0, shape, DT_FP32);
  unsigned int tensor_size = shape->n * stride.n * sizeof(float);
  // try to avoid bank confilct
  local_addr_t work1_addr = buffer_addr;
  local_addr_t work0_addr = ALIGN(work1_addr + tensor_size, ALIGN_BYTES);
  local_addr_t coeff_addr = ALIGN(work0_addr + tensor_size, ALIGN_BYTES);
  // must be same with global layer,
  // 2 bank for work1, 2 bank for work0, 1 bank for coeff and table,
  if (!if_local_layer) {
    int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
    work0_addr = ALIGN(work1_addr + tensor_size, bank_size);
    coeff_addr = ALIGN(work0_addr + tensor_size, bank_size);
  }

  tpu_bdc_cast(coeff_addr, in_addr, shape, NULL, NULL, DT_FP32, DT_FP16, RM_HALF_TO_EVEN);
  // TPUKERNEL_ASSERT(dtype == DT_FP32 && "active_gelu only support float32");
  int align_size = (tpu_local_mem_size_per_npu() / tpu_bank_num());
  local_addr_t work2_addr = ALIGN(work0_addr + tensor_size, align_size);
  local_addr_t work3_addr = ALIGN(work2_addr + tensor_size, align_size);
  local_addr_t exp_coeff_addr = ALIGN(work3_addr + tensor_size, align_size);
  local_addr_t erf_coeff_addr = ALIGN(exp_coeff_addr + 32 * sizeof(float), ALIGN_BYTES);
  local_addr_t exp_table_addr = ALIGN(erf_coeff_addr + 10 * sizeof(float), ALIGN_BYTES);
  tpu_bdc_load_fp32_exp_coeff(exp_coeff_addr);
  tpu_bdc_load_fp32_exp_table(exp_table_addr);
  tpu_bdc_load_fp32_erf_coeff(erf_coeff_addr);
  tpu_bdc_fp32_gelu(
      out_addr,
      coeff_addr,
      work0_addr,
      work1_addr,
      work2_addr,
      work3_addr,
      exp_coeff_addr,
      erf_coeff_addr,
      exp_table_addr,
      shape);
  tpu_bdc_cast(out_addr, out_addr, shape, NULL, NULL, DT_FP16, DT_FP32, RM_HALF_TO_EVEN);
}

// table, coeff, work0, work1 are all local buffer
static void nodechip_gelu_fp16_local(
    local_addr_t in_addr,
    local_addr_t out_addr,
    local_addr_t buffer_addr,
    const int* shape,
    data_type_t dtype,
    sg_active_type_t active_type,
    int if_local_layer,
    float* coef)
{
  dim4 p_shape = {shape[0], shape[1], shape[2], shape[3]};
  // save local memory buffer size
  p_shape.n = if_local_layer ? 1 : shape[0];
  dim4 stride = {0};
  tpu_aligned_stride(&stride, 0, &p_shape, DT_FP32);
  unsigned int tensor_size = stride.n * sizeof(float);
  for (int i = 0; i < shape[0]; i += p_shape.n) {
    gelu_local_perbatch(
      in_addr + tensor_size * i,
      out_addr + tensor_size * i,
      buffer_addr,
      &p_shape,
      dtype,
      active_type,
      if_local_layer,
      coef);
  }
}

static void nodechip_gelu_forward_fp16(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    int*          shape,
    int           dim,
    data_type_t   dtype)
{
  // 2 bank for work0, 2 bank for work1, 1 bank for coeff and table
  // 2 bank for input, 2 bank for output
  int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num();
  local_addr_t in_local_addr[2] = {0, bank_size};
  local_addr_t out_local_addr[2] = {2 * bank_size, 3 * bank_size};
  local_addr_t buffer_addr = 4 * bank_size;

  unsigned long long length = 1;
  for (int i = 0; i < dim; i++) {
    length *= (unsigned long long)shape[i];
  }

  int npu_num = tpu_npu_num();
  int eu_num = tpu_eu_num(dtype);
  /// continus 128byte can get better peformance
  int tensor_w = _MAX(DIV_UP(MIN(length, 16384), npu_num), DIV_UP(128, eu_num * tpu_data_type_size(dtype)));
  unsigned long long slice = MIN(MIN(length, (unsigned long long)npu_num * tensor_w), 16384);

  int max_rows_per_time = (bank_size) / (tensor_w*tpu_data_type_size(dtype));
  int rows = DIV_UP(length, slice);
  int rows_secs = DIV_UP(rows, max_rows_per_time);
  // at least loop two times to overlap all bdc time
  int rows_slice = DIV_UP(rows, _MAX(rows_secs, 2));

  unsigned long long cur_idx[3] = {0}, cur_rows[3] = {0}, cur_cols[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  while (cur_idx[2] < length) {
    tpu_parallel_start();
    // update load info
    if (draning_idx < 1) {
      unsigned long long cur_len = MIN(length - cur_idx[0], rows_slice * slice);
      cur_cols[0] = MIN(cur_len, slice);
      cur_rows[0] = _MAX(1, cur_len / cur_cols[0]); // don't use DIV_UP
    }

    // store output
    if (stage_idx > 1) {
      tpu_gdma_matrix_L2S(
          out_global_addr + cur_idx[2] * tpu_data_type_size(dtype),
          out_local_addr[stage_idx & 0x1],
          cur_rows[2], cur_cols[2], tensor_w,
          cur_cols[2], dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(
          in_local_addr[stage_idx & 0x1],
          in_global_addr + cur_idx[0] * tpu_data_type_size(dtype),
          cur_rows[0], cur_cols[0], tensor_w,
          cur_cols[0], dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2) {
      int cur_shape[4] = {cur_rows[1], DIV_UP(cur_cols[1], tensor_w), 1, tensor_w};
      nodechip_gelu_fp16_local(
          in_local_addr[(stage_idx - 1) & 0x1],
          out_local_addr[(stage_idx - 1) & 0x1],
          buffer_addr,
          cur_shape,
          DT_FP16,
          ACTIVE_GELU,
          0,
          NULL);
    }

    tpu_parallel_end();
    pipeline_move(cur_idx, 3);
    pipeline_move(cur_cols, 3);
    pipeline_move(cur_rows, 3);
    if (draning_idx < 1) {
      cur_idx[0] += cur_cols[0] * cur_rows[0];
      if (cur_idx[0] >= length) {
        draning_idx++;
      }
    } else {
      draning_idx++;
    }
    stage_idx++;
  }
}


void tpu_kernel_api_active_forward(const void *args)
{
    sg_api_active_forward_t *api = (sg_api_active_forward_t*)args;
    tpu_initialize();
    if(api->active_type == ACTIVE_GELU && api->dtype == 1)
    {
        nodechip_gelu_forward_fp16(
            api->in_global_addr,
            api->out_global_addr,
            api->shape,
            api->shape_dim,
            DT_FP16);
    }
    else
    {
        nodechip_active(
            api->in_global_addr,
            api->out_global_addr,
            api->shape,
            api->shape_dim,
            tpu_type_convert((sg_data_type_t)api->dtype),
            api->active_type,
            NULL);
    }
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_active_forward);
