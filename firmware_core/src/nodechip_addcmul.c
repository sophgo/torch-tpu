#include "sg_api_struct.h"
#include "tpu_kernel.h"


inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

/*
 * output = input + value * ( tensor1 * tensor2 )
 */
void nodechip_addcmul_bcast (
global_addr_t input_global_addr,
global_addr_t tensor1_global_addr,
global_addr_t tensor2_global_addr,
global_addr_t output_global_addr,
scalar_t value,
const dim4  * input_shape,
const dim4  * tensor1_shape,
const dim4  * tensor2_shape,
const dim4  * output_shape,
data_type_t dtype )
{
  dim4 input_global_stride, tensor1_global_stride, tensor2_global_stride, output_global_stride;
  tpu_continuous_stride ( &input_global_stride, input_shape );
  tpu_continuous_stride ( &tensor1_global_stride, tensor1_shape );
  tpu_continuous_stride ( &tensor2_global_stride, tensor2_shape );
  tpu_continuous_stride ( &output_global_stride, output_shape );
  const bool input_bcast[4] =
  {
    input_shape->n != output_shape->n,
    input_shape->c != output_shape->c,
    input_shape->h != output_shape->h,
    input_shape->w != output_shape->w
  };
  const bool tensor1_bcast[4] =
  {
    tensor1_shape->n != output_shape->n,
    tensor1_shape->c != output_shape->c,
    tensor1_shape->h != output_shape->h,
    tensor1_shape->w != output_shape->w
  };
    const bool tensor2_bcast[4] =
  {
    tensor2_shape->n != output_shape->n,
    tensor2_shape->c != output_shape->c,
    tensor2_shape->h != output_shape->h,
    tensor2_shape->w != output_shape->w
  };
  const int c_per_npu = DIV_UP ( output_shape->c, NPU_NUM );
  int hmax = output_shape->h, nmax = output_shape->n, cmax = c_per_npu * NPU_NUM;
  local_addr_t input_local_addr, tensor1_local_addr, tensor2_local_addr, output_local_addr;
  local_addr_t next = 0;
  while ( true )
  {
    next = 0;
    int size = tpu_aligned_feature_size ( hmax, output_shape->w, dtype ) * DIV_UP ( cmax, NPU_NUM ) * nmax;
    input_local_addr = next; next += size;
    tensor1_local_addr = next; next += size;
    tensor2_local_addr = next; next += size;
    output_local_addr = next; next += size;
    int total_size = next;
    if ( total_size <= LOCAL_MEM_SIZE )
    {
      break;
    }
    else
    {
      if ( cmax > NPU_NUM )
      {
        if ( cmax % NPU_NUM == 0 )
        {
          cmax -= NPU_NUM;
        }
        else
        {
          cmax -= ( cmax % NPU_NUM );
        }
        continue;
      }
      else if ( nmax > 1 )
      {
        nmax /= 2;
        continue;
      }
      else if ( hmax > 1 )
      {
        hmax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT ( false );
      }
    }
  }
  dim4 shape = { .w = output_shape->w };
  dim4 input_local_shape, tensor1_local_shape, tensor2_local_shape;
  dim4 input_local_stride, tensor1_local_stride, tensor2_local_stride;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  int ctodo = output_shape->c, cdone = 0;
  while ( ctodo > 0 )
  {
    shape.c = MIN ( ctodo, cmax );
    int ntodo = output_shape->n, ndone = 0;
    while ( ntodo > 0 )
    {
      shape.n = MIN ( ntodo, nmax );
      int htodo = output_shape->h, hdone = 0;
      while ( htodo > 0 )
      {
        shape.h = MIN ( htodo, hmax );
        // Move input from global memory to local memory
        input_local_shape.n = input_bcast[0] ? 1 : shape.n;
        input_local_shape.c = input_bcast[1] ? 1 : shape.c;
        input_local_shape.h = input_bcast[2] ? 1 : shape.h;
        input_local_shape.w = input_bcast[3] ? 1 : shape.w;
        tpu_aligned_stride ( &input_local_stride, 0, &input_local_shape, dtype );
        global_addr_t input_global_addr_gdma =
        input_global_addr + (
        ( input_bcast[0] ? 0 : ndone ) * input_global_stride.n +
        ( input_bcast[1] ? 0 : cdone ) * input_global_stride.c +
        ( input_bcast[2] ? 0 : hdone ) * input_global_stride.h ) * tpu_data_type_size ( dtype );
        tpu_gdma_cpy_S2L ( input_local_addr, input_global_addr_gdma, &input_local_shape, &input_local_stride, &input_global_stride, dtype );
        // Move tensor1 from global memory to local memory
        tensor1_local_shape.n = tensor1_bcast[0] ? 1 : shape.n;
        tensor1_local_shape.c = tensor1_bcast[1] ? 1 : shape.c;
        tensor1_local_shape.h = tensor1_bcast[2] ? 1 : shape.h;
        tensor1_local_shape.w = tensor1_bcast[3] ? 1 : shape.w;
        tpu_aligned_stride ( &tensor1_local_stride, 0, &tensor1_local_shape, dtype );
        global_addr_t tensor1_global_addr_gdma =
        tensor1_global_addr + (
        ( tensor1_bcast[0] ? 0 : ndone ) * tensor1_global_stride.n +
        ( tensor1_bcast[1] ? 0 : cdone ) * tensor1_global_stride.c +
        ( tensor1_bcast[2] ? 0 : hdone ) * tensor1_global_stride.h ) * tpu_data_type_size ( dtype );
        tpu_gdma_cpy_S2L ( tensor1_local_addr, tensor1_global_addr_gdma, &tensor1_local_shape, &tensor1_local_stride, &tensor1_global_stride, dtype );
        // Move tensor2 from global memory to local memory
        tensor2_local_shape.n = tensor2_bcast[0] ? 1 : shape.n;
        tensor2_local_shape.c = tensor2_bcast[1] ? 1 : shape.c;
        tensor2_local_shape.h = tensor2_bcast[2] ? 1 : shape.h;
        tensor2_local_shape.w = tensor2_bcast[3] ? 1 : shape.w;
        tpu_aligned_stride ( &tensor2_local_stride, 0, &tensor2_local_shape, dtype );
        global_addr_t tensor2_global_addr_gdma =
        tensor2_global_addr + (
        ( tensor2_bcast[0] ? 0 : ndone ) * tensor2_global_stride.n +
        ( tensor2_bcast[1] ? 0 : cdone ) * tensor2_global_stride.c +
        ( tensor2_bcast[2] ? 0 : hdone ) * tensor2_global_stride.h ) * tpu_data_type_size ( dtype );
        tpu_gdma_cpy_S2L ( tensor2_local_addr, tensor2_global_addr_gdma, &tensor2_local_shape, &tensor2_local_stride, &tensor2_global_stride, dtype );
        // Broadcast input if needed
        if ( input_bcast[1] )
        {
          input_local_shape.c = NPU_NUM;
          tpu_bdc_npu_bcast ( input_local_addr, input_local_addr, &input_local_shape, dtype );
        }
        if ( input_bcast[0] || input_bcast[2] || input_bcast[3] || ( input_bcast[1] && shape.c > NPU_NUM ) )
        {
          input_local_stride.n = input_bcast[0] ? 0 : input_local_stride.n;
          input_local_stride.c = input_bcast[1] ? 0 : input_local_stride.c;
          input_local_stride.h = input_bcast[2] ? 0 : input_local_stride.h;
          input_local_stride.w = input_bcast[3] ? 0 : input_local_stride.w;
        }
        // Broadcast tensor1 if needed
        if ( tensor1_bcast[1] )
        {
          tensor1_local_shape.c = NPU_NUM;
          tpu_bdc_npu_bcast ( tensor1_local_addr, tensor1_local_addr, &tensor1_local_shape, dtype );
        }
        if ( tensor1_bcast[0] || tensor1_bcast[2] || tensor1_bcast[3] || ( tensor1_bcast[1] && shape.c > NPU_NUM ) )
        {
          tensor1_local_stride.n = tensor1_bcast[0] ? 0 : tensor1_local_stride.n;
          tensor1_local_stride.c = tensor1_bcast[1] ? 0 : tensor1_local_stride.c;
          tensor1_local_stride.h = tensor1_bcast[2] ? 0 : tensor1_local_stride.h;
          tensor1_local_stride.w = tensor1_bcast[3] ? 0 : tensor1_local_stride.w;
        }
        // Broadcast tensor2 if needed
        if ( tensor2_bcast[1] )
        {
          tensor2_local_shape.c = NPU_NUM;
          tpu_bdc_npu_bcast ( tensor2_local_addr, tensor2_local_addr, &tensor2_local_shape, dtype );
        }
        if ( tensor2_bcast[0] || tensor2_bcast[2] || tensor2_bcast[3] || ( tensor2_bcast[1] && shape.c > NPU_NUM ) )
        {
          tensor2_local_stride.n = tensor2_bcast[0] ? 0 : tensor2_local_stride.n;
          tensor2_local_stride.c = tensor2_bcast[1] ? 0 : tensor2_local_stride.c;
          tensor2_local_stride.h = tensor2_bcast[2] ? 0 : tensor2_local_stride.h;
          tensor2_local_stride.w = tensor2_bcast[3] ? 0 : tensor2_local_stride.w;
        }
        //Compute
        tpu_bdc_fp_mul ( output_local_addr, tensor1_local_addr, tensor2_local_addr, &shape, NULL, &tensor1_local_stride, &tensor2_local_stride, dtype );
        tpu_bdc_fp_mul_C ( output_local_addr, output_local_addr, value, &shape, NULL, NULL, dtype );
        tpu_bdc_fp_add ( output_local_addr, output_local_addr, input_local_addr, &shape, NULL, NULL, &input_local_stride, dtype );
        // Move out from local memory to global memory if needed
        l2s = true;
        l2s_global_addr = output_global_addr + (
        ndone * output_global_stride.n +
        cdone * output_global_stride.c +
        hdone * output_global_stride.h ) * tpu_data_type_size ( dtype );
        l2s_shape = shape;
        l2s_local_addr = output_local_addr;
        if ( l2s )
        {
          tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, &output_global_stride, NULL, dtype );
        }
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

int tpu_kernel_api_addcmul_bcast ( const void * args )
{
  sg_api_bcast_addcmul_t * api = ( sg_api_bcast_addcmul_t * ) args;
  data_type_t dtype = ( data_type_t ) api->dtype;
  TPUKERNEL_ASSERT ( dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16 );
  scalar_t value;
  if ( dtype == DT_FP32 )
  {
  value.f32 = api->value;
  }
  else
  {
    scalar_t value_f32 = { .f32 = api->value };
    value = tpu_fp_cast ( value_f32, dtype, DT_FP32, RM_HALF_TO_EVEN );
  }

  dim4 input_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 tensor1_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 tensor2_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 output_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };

  if(api->input_dim>=1) {input_shape.n = api->input_shape[0];}
  if(api->tensor1_dim>=1) {tensor1_shape.n = api->tensor1_shape[0];}
  if(api->tensor2_dim>=1) {tensor2_shape.n = api->tensor2_shape[0];}

  if(api->input_dim>=2) {input_shape.c = api->input_shape[1];}
  if(api->tensor1_dim>=2) {tensor1_shape.c = api->tensor1_shape[1];}
  if(api->tensor2_dim>=2) {tensor2_shape.c = api->tensor2_shape[1];}

  if(api->input_dim>=3) {input_shape.h = api->input_shape[2];}
  if(api->tensor1_dim>=3) {tensor1_shape.h = api->tensor1_shape[2];}
  if(api->tensor2_dim>=3) {tensor2_shape.h = api->tensor2_shape[2];}

  if(api->input_dim>=4) {input_shape.w = api->input_shape[3];}
  if(api->tensor1_dim>=4) {tensor1_shape.w = api->tensor1_shape[3];}
  if(api->tensor2_dim>=4) {tensor2_shape.w = api->tensor2_shape[3];}

  output_shape.n = input_shape.n > tensor1_shape.n ? input_shape.n > tensor2_shape.n ?
          input_shape.n : tensor2_shape.n : tensor1_shape.n > tensor2_shape.n ? tensor1_shape.n : tensor2_shape.n;
  output_shape.c = input_shape.c > tensor1_shape.c ? input_shape.c > tensor2_shape.c ?
          input_shape.c : tensor2_shape.c : tensor1_shape.c > tensor2_shape.c ? tensor1_shape.c : tensor2_shape.c;
  output_shape.h = input_shape.h > tensor1_shape.h ? input_shape.h > tensor2_shape.h ?
          input_shape.h : tensor2_shape.h : tensor1_shape.h > tensor2_shape.h ? tensor1_shape.h : tensor2_shape.h;
  output_shape.w = input_shape.w > tensor1_shape.w ? input_shape.w > tensor2_shape.w ?
          input_shape.w : tensor2_shape.w : tensor1_shape.w > tensor2_shape.w ? tensor1_shape.w : tensor2_shape.w;

  tpu_initialize();
  nodechip_addcmul_bcast (  api->input_global_addr,
                      api->tensor1_global_addr,
                      api->tensor2_global_addr,
                      api->output_global_addr,
                      value,
                      &input_shape,
                      &tensor1_shape,
                      &tensor2_shape,
                      &output_shape,
                      dtype );
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_addcmul_bcast);

void nodechip_addcmul(global_addr_t input_global_addr,
                      global_addr_t tensor1_global_addr,
                      global_addr_t tensor2_global_addr,
                      global_addr_t output_global_addr, scalar_t value,
                      unsigned long long length, data_type_t dtype) {
  if (length == 0) {
    return;
  }

  const unsigned int bank_bsize = tpu_local_mem_size_per_npu() / tpu_bank_num();
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_num = 2 + 2 + 2 + 2; // 4 inputs, 4 + 4 input tensor, 4 outputs
  int tensor_bsize_pnpu = tpu_bank_num() / tensor_num * bank_bsize;
  TPUKERNEL_ASSERT(tensor_bsize_pnpu > 0);

  // max local memory is tpu_local_mem_size_per_npu() * tpu_npu_num()
  // for bm1684x, is (16384   *   16)    *   64
  //                    ↑          ↑          ↑
  //                bank_size * bank_num * npu_num
  // (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1 = 32768,
  // when `m_dim` is set to this value, it ensures both n and m dim to not
  // exceed **shape limit**
  const unsigned int max_m_dim = (tpu_gdma_shape_limit(TENSOR_C_DIM) + 1) >> 1;

  // w should larger equal than tpu_eu_num(dtype) to make full use of
  // eu(execution unit)
  const unsigned int w_dim = tpu_eu_num(dtype);
  const int n_dim = tensor_bsize_pnpu * tpu_npu_num() / dtype_size / max_m_dim;

  local_addr_t in_local_addr[2] = {0, 1 * tensor_bsize_pnpu};
  local_addr_t tensor1_local_addr[2] = {2 * tensor_bsize_pnpu,
                                        3 * tensor_bsize_pnpu};
  local_addr_t tensor2_local_addr[2] = {4 * tensor_bsize_pnpu,
                                        5 * tensor_bsize_pnpu};
  local_addr_t out_local_addr[2] = {6 * tensor_bsize_pnpu,
                                    7 * tensor_bsize_pnpu};

  unsigned long long cur_idx[3] = {0}, cur_n_dim[3] = {0}, cur_m_dim[3] = {0};
  int stage_idx = 0, draning_idx = 0;
  while (cur_idx[2] < length) {
    tpu_parallel_start();

    // update load info
    if (draning_idx < 1) {
      unsigned long long cur_len =
          MIN(length - cur_idx[0], // remained element size
              n_dim * max_m_dim    // max matrix element size, n_dim * m_dim <
                                   // tensor_size_pnpu * npu_num
          );

      // if cur_len is larger than m_dim (for a big matrix), limit `m` to not
      // exceed max_m_dim, in this case n > 1 else, take cur_len as m, in this
      // case n = 1 NOTE: n_dim * max_m_dim <= tensor_size_pnpu * npu_num, it's
      // always a legal size.
      cur_m_dim[0] = MIN(cur_len, max_m_dim);
      cur_n_dim[0] = cur_len / cur_m_dim[0]; // cur_len / cur_m_dim[0] >= 1
    }
    // store output
    if (stage_idx > 1) {
      tpu_gdma_matrix_L2S(output_global_addr + cur_idx[2] * dtype_size,
                          out_local_addr[stage_idx & 0x1],
                          /**rows, cols, cols_per_channel, row_stride*/
                          cur_n_dim[2], cur_m_dim[2], w_dim, cur_m_dim[2],
                          dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(in_local_addr[stage_idx & 0x1],
                          input_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
      tpu_gdma_matrix_S2L(tensor1_local_addr[stage_idx & 0x1],
                          tensor1_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
      tpu_gdma_matrix_S2L(tensor2_local_addr[stage_idx & 0x1],
                          tensor2_global_addr + cur_idx[0] * dtype_size,
                          cur_n_dim[0], cur_m_dim[0], w_dim, cur_m_dim[0],
                          dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2) {
      dim4 cur_shape = {cur_n_dim[1], DIV_UP(cur_m_dim[1], w_dim), 1,
                        w_dim}; // matrix layout shape (n, c, h, w)
      tpu_bdc_fp_mul(out_local_addr[(stage_idx - 1) & 0x1],
                     tensor1_local_addr[(stage_idx - 1) & 0x1],
                     tensor2_local_addr[(stage_idx - 1) & 0x1], &cur_shape,
                     NULL, NULL, NULL, dtype);
      tpu_bdc_fp_mul_C(out_local_addr[(stage_idx - 1) & 0x1],
                       out_local_addr[(stage_idx - 1) & 0x1], value, &cur_shape,
                       NULL, NULL, dtype);
      tpu_bdc_fp_add(out_local_addr[(stage_idx - 1) & 0x1],
                     out_local_addr[(stage_idx - 1) & 0x1],
                     in_local_addr[(stage_idx - 1) & 0x1], &cur_shape, NULL,
                     NULL, NULL, dtype);
    }

    tpu_parallel_end();
    pipeline_move(cur_idx, 3);
    pipeline_move(cur_m_dim, 3);
    pipeline_move(cur_n_dim, 3);
    if (draning_idx < 1) {
      cur_idx[0] += cur_m_dim[0] * cur_n_dim[0];
      if (cur_idx[0] >= length) {
        draning_idx++;
      }
    } else {
      draning_idx++;
    }
    stage_idx++;
  }
}

int tpu_kernel_api_addcmul(const void *args) {
  sg_api_addcmul_t *api = (sg_api_addcmul_t *)args;
  data_type_t dtype = (data_type_t)api->dtype;
  TPUKERNEL_ASSERT(dtype == DT_FP32 || dtype == DT_FP16 || dtype == DT_BFP16);
  scalar_t value;
  if (dtype == DT_FP32) {
    value.f32 = api->value;
  } else {
    scalar_t value_f32 = {.f32 = api->value};
    value = tpu_fp_cast(value_f32, dtype, DT_FP32, RM_HALF_TO_EVEN);
  }
  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_addcmul(api->input_global_addr, api->tensor1_global_addr,
                   api->tensor2_global_addr, api->output_global_addr, value,
                   length, dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_addcmul);
