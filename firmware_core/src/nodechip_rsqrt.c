#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * output = input + value * ( other )
 */

void nodechip_scale_rsqrt(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    int length,
    data_type_t dtype)
{
  const int dsize = tpu_data_type_size(dtype);
  int wmax = DIV_UP(length, NPU_NUM);
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t next = 0;
  while (true)
  {
    next = 0;
    int size = tpu_aligned_feature_size(1, wmax, dtype);
    input_local_addrs[0] = next;
    next += size;
    input_local_addrs[1] = next;
    next += size;
    output_local_addrs[0] = next;
    next += size;
    output_local_addrs[1] = next;
    next += size;
    if ((int)next <= LOCAL_MEM_SIZE)
    {
      break;
    }
    else
    {
      if (wmax > 1)
      {
        wmax /= 2;
        continue;
      }
      else
      {
        TPUKERNEL_ASSERT(false);
      }
    }
  }
  int todo = length;
  int done = 0;
  dim4 shape = {.n = 1, .h = 1};
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  while (todo != 0)
  {
    if (todo > NPU_NUM)
    {
      shape.c = NPU_NUM;
      shape.w = MIN(todo / NPU_NUM, wmax);
    }
    else
    {
      shape.c = todo;
      shape.w = 1;
    }
    tpu_gdma_cpy_S2L(input_local_addrs[index], input_global_addr + done * dsize, &shape, NULL, NULL, dtype);
    if (tpu_is_parallel_state())
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if (l2s)
    {
      tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
    }
    tpu_bdc_fp32_rsqrt(output_local_addrs[index], input_local_addrs[index], &shape);
    l2s = true;
    l2s_global_addr = output_global_addr + done * dsize;
    l2s_local_addr = output_local_addrs[index];
    l2s_shape = shape;
    todo -= shape.c * shape.w;
    done += shape.c * shape.w;
    index = 1 - index;
  }
  if (tpu_is_parallel_state())
  {
    tpu_parallel_end();
  }
  if (l2s)
  {
    tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
  }
}

void tpu_kernel_api_rsqrt(const void *args)
{
  sg_api_rsqrt_t *api = (sg_api_rsqrt_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);
  
  int length = 1;
  for (int i = 0; i < api->dim; ++i)
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_scale_rsqrt(( global_addr_t)api->input_global_addr, (global_addr_t)api->output_global_addr, length, (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_rsqrt);

void tpu_kernel_api_rsqrt_multi_core(const void *args)
{
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_rsqrt_multi_core);

