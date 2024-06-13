#include "sg_api_struct.h"
#include "tpu_kernel.h"


inline static void pipeline_move(unsigned long long *array, int num)
{
  for (int i = num - 1; i > 0; i--)
  {
    array[i] = array[i - 1];
  }
}

void nodechip_shift_left (
global_addr_t output_global_addr,
global_addr_t input_global_addr,
global_addr_t other_global_addr,
unsigned long long length,
data_type_t dst_dtype,
data_type_t src_dtype,
data_type_t shift_dtype)
{
  if(length==0) return;
  unsigned int bank_size = tpu_local_mem_size_per_npu()/tpu_bank_num();
  int npu_num = tpu_npu_num();
  int tensor_num = 6; // 2 outputs, 2 input1, 1 work1, 1 work2, 1 coeff
  int tensor_size = tpu_bank_num()/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t input_local_addr[2] = {0, 1 * bank_size};
  local_addr_t other_local_addr[2] = {2* bank_size, 3 * bank_size};
  local_addr_t output_local_addr[2] = {4 * bank_size, 5 * bank_size};

  int dtype_size = tpu_data_type_size(src_dtype);
  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;

  const unsigned int w_dim = tpu_eu_num(src_dtype);
  const int tensor_size_pnpu = tensor_size / dtype_size;
  const int n_dim = tensor_size_pnpu * npu_num / max_m_dim;

  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  while (cur_idx[2] < length)
  {
    tpu_parallel_start();
    // update load info
    if (draning_idx < 1)
    {
      unsigned long long cur_len = MIN(
          length - cur_idx[0],
          n_dim * max_m_dim
      );
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0];
    }

    // store output
    if (stage_idx > 1)
    {
      tpu_gdma_matrix_L2S(
          output_global_addr + cur_idx[2] * tpu_data_type_size(src_dtype),
          output_local_addr[stage_idx & 0x1],
          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2], src_dtype);
    }

    // load input
    if (draning_idx < 1)
    {
      tpu_gdma_matrix_S2L(
          input_local_addr[stage_idx & 0x1],
          input_global_addr + cur_idx[0] * tpu_data_type_size(src_dtype),
          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0], src_dtype);
      tpu_gdma_matrix_S2L(
          other_local_addr[stage_idx & 0x1],
          other_global_addr + cur_idx[0] * tpu_data_type_size(shift_dtype),
          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0], shift_dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2)
    {
      
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1, w_dim};
      tpu_bdc_logical_shift(output_local_addr[(stage_idx - 1) & 0x1],
                  input_local_addr[(stage_idx - 1) & 0x1],
                  other_local_addr[(stage_idx - 1) & 0x1],
                  &cur_shape, NULL, NULL, NULL, dst_dtype, src_dtype, /*not correct*/DT_INT8, 0);
    }

    tpu_parallel_end();
    pipeline_move(cur_idx, 3);
    pipeline_move(cur_m_dim, 3);
    pipeline_move(cur_n_dim, 3);
    if (draning_idx < 1)
    {
      cur_idx[0] += cur_m_dim[0] * cur_n_dim[0];
      if (cur_idx[0] >= length)
      {
        draning_idx++;
      }
    }
    else
    {
      draning_idx++;
    }
    stage_idx++;
  }
}

int tpu_kernel_api_shift_left ( const void * args )
{
  sg_api_shift_left_t * api = ( sg_api_shift_left_t * ) args;
  //src_dtype unsigned, dst_dtype unsigned, shift_dtype signed(not implemented now)
  //TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_shift_left ( api->output_global_addr, api->input_global_addr,api->other_global_addr, length, 
  ( data_type_t ) api->dst_dtype,( data_type_t ) api->src_dtype,( data_type_t ) api->other_dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_shift_left );

#ifdef BACKEND_SG2260
int tpu_kernel_api_shift_left_multi_core(const void *args)
{
  sg_api_shift_left_t * api = ( sg_api_shift_left_t * ) args;
  //src_dtype unsigned, dst_dtype unsigned, shift_dtype signed(not implemented now)
  //TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();

  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1)
    cur_length_slice = length - length_slice * (length_secs - 1);
  nodechip_shift_left(
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->src_dtype),
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->src_dtype),
      api->other_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->other_dtype),
      cur_length_slice,
      (data_type_t)api->src_dtype,(data_type_t)api->dst_dtype, (data_type_t)api->other_dtype);

  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_shift_left_multi_core);
#endif

void nodechip_shift_left_c (
global_addr_t output_global_addr,
global_addr_t input_global_addr,
unsigned long long length,
char shift,
data_type_t dst_dtype,
data_type_t src_dtype)
{
  if(length==0) return;
  unsigned int bank_size = tpu_local_mem_size_per_npu()/tpu_bank_num();
  int npu_num = tpu_npu_num();
  int tensor_num = 4; // 2 outputs, 2 input1, 1 work1, 1 work2, 1 coeff
  int tensor_size = tpu_bank_num()/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t input_local_addr[2] = {0, 1 * bank_size};
  local_addr_t output_local_addr[2] = {4 * bank_size, 5 * bank_size};

  int dtype_size = tpu_data_type_size(src_dtype);
  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;

  const unsigned int w_dim = tpu_eu_num(src_dtype);
  const int tensor_size_pnpu = tensor_size / dtype_size;
  const int n_dim = tensor_size_pnpu * npu_num / max_m_dim;

  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  while (cur_idx[2] < length)
  {
    tpu_parallel_start();
    // update load info
    if (draning_idx < 1)
    {
      unsigned long long cur_len = MIN(
          length - cur_idx[0],
          n_dim * max_m_dim
      );
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0];
    }

    // store output
    if (stage_idx > 1)
    {
      tpu_gdma_matrix_L2S(
          output_global_addr + cur_idx[2] * tpu_data_type_size(src_dtype),
          output_local_addr[stage_idx & 0x1],
          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2], src_dtype);
    }

    // load input
    if (draning_idx < 1)
    {
      tpu_gdma_matrix_S2L(
          input_local_addr[stage_idx & 0x1],
          input_global_addr + cur_idx[0] * tpu_data_type_size(src_dtype),
          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0], src_dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2)
    {
      
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1, w_dim};
      tpu_bdc_logical_shift_C(output_local_addr[(stage_idx - 1) & 0x1],
                  input_local_addr[(stage_idx - 1) & 0x1],
                  (char)shift,&cur_shape, NULL, NULL, dst_dtype, src_dtype,  0);
                 
    }

    tpu_parallel_end();
    pipeline_move(cur_idx, 3);
    pipeline_move(cur_m_dim, 3);
    pipeline_move(cur_n_dim, 3);
    if (draning_idx < 1)
    {
      cur_idx[0] += cur_m_dim[0] * cur_n_dim[0];
      if (cur_idx[0] >= length)
      {
        draning_idx++;
      }
    }
    else
    {
      draning_idx++;
    }
    stage_idx++;
  }
}

int tpu_kernel_api_shift_left_c ( const void * args )
{
  sg_api_shift_left_c_t * api = ( sg_api_shift_left_c_t * ) args;
  //src_dtype unsigned, dst_dtype unsigned, shift_dtype signed(not implemented now)
  //TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_shift_left_c ( api->output_global_addr, api->input_global_addr, length, api->const_value,
  ( data_type_t ) api->dst_dtype,( data_type_t ) api->src_dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_shift_left_c );

#ifdef BACKEND_SG2260
int tpu_kernel_api_shift_left_c_multi_core(const void *args)
{
  sg_api_shift_left_c_t * api = ( sg_api_shift_left_c_t * ) args;
  //src_dtype unsigned, dst_dtype unsigned, shift_dtype signed(not implemented now)
  //TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);
  int length = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    length *= api->shape[i];
  }
  tpu_initialize();

  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int length_slice = DIV_UP(length, core_num);
  int length_secs = DIV_UP(length, length_slice);
  TPUKERNEL_ASSERT(length_secs <= core_num);
  int cur_length_slice = length_slice;
  if (core_idx == length_secs - 1)
    cur_length_slice = length - length_slice * (length_secs - 1);
  nodechip_shift_left_c(
      api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->src_dtype),
      api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->src_dtype),
      cur_length_slice, api->const_value,
      ( data_type_t ) api->dst_dtype,( data_type_t ) api->src_dtype);

  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_shift_left_c_multi_core);
#endif