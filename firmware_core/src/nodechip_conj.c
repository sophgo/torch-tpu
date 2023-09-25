#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "stdio.h"

/*
 * output = input + value * ( other )
 */

void nodechip_scale_conj(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    int dim,
    int length,
    data_type_t dtype)
{
  const int dsize = tpu_data_type_size(dtype);
  int wmax = DIV_UP(length, NPU_NUM);
  //printf(" length=%d NPU_NUM=%d wmax=%d\n",length, NPU_NUM, wmax);
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t next = 0;
  while (true)
  {
    //printf("next=%d\n",next);
    next = 0;
    int size = tpu_aligned_feature_size(1, wmax, dtype);
    //printf("  input_local_addrs[0]: %u\n",next);
    input_local_addrs[0] = next;
    next += size;
    //printf("  input_local_addrs[1]: %u\n",next);
    input_local_addrs[1] = next;
    next += size;
    //printf("  output_local_addrs[0]: %u\n",next);
    output_local_addrs[0] = next;
    next += size;
    //printf("  output_local_addrs[1]: %u\n",next);
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
  //printf("length: %d\n",todo);
  int done = 0;
  dim4 shape = {.n = 1, .h = 1};
  int index = 0;
  bool l2s = false;
  bool real_or_imag =true;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  dim4 copy_in_stride;
  dim4 copy_in_stride_before;

  while (todo != 0)
  {
    if(real_or_imag){
      if (todo > NPU_NUM)  
      {
        shape.c = NPU_NUM;
        shape.w = MIN(todo / NPU_NUM, wmax);
        copy_in_stride.w = 2;
        copy_in_stride.h = shape.w*copy_in_stride.w;
        copy_in_stride.c = shape.h*copy_in_stride.h;
        copy_in_stride.n = shape.c*copy_in_stride.c;
      }
      else
      {
        shape.c = todo;
        shape.w = 1;
        copy_in_stride.w = 2;
        copy_in_stride.h = shape.w*copy_in_stride.w;
        copy_in_stride.c = shape.h*copy_in_stride.h;
        copy_in_stride.n = shape.c*copy_in_stride.c;
      }
    }

    tpu_gdma_cpy_S2L(input_local_addrs[index], input_global_addr + done * dsize*2, &shape, NULL, &copy_in_stride, dtype);
    if (tpu_is_parallel_state())
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if (l2s)
    {
      tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, &copy_in_stride_before, NULL, dtype);
    }
    l2s = true;
    if(!real_or_imag)
    {
        tpu_bdc_neg(output_local_addrs[index], input_local_addrs[index], &shape, NULL, NULL, dtype);
        l2s_global_addr+=dsize;
        real_or_imag = true;
        todo -= shape.c * shape.w ;
        done += shape.c * shape.w ;
        input_global_addr-=dsize;
    }
    else{
        tpu_gdma_cpy_L2L(output_local_addrs[index], input_local_addrs[index], &shape, NULL, NULL, dtype);
        l2s_global_addr = output_global_addr + done * dsize*2;
        real_or_imag = false;
        input_global_addr+=dsize;
    }

    copy_in_stride_before=copy_in_stride;
    l2s_local_addr = output_local_addrs[index];
    index = 1 - index;
    l2s_shape = shape;
  }
  if (tpu_is_parallel_state())
  {
    tpu_parallel_end();
  }
  if (l2s)
  {
    tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, &copy_in_stride_before, NULL, dtype);
  }
}

void tpu_kernel_api_conj(const void *args)
{
  sg_api_real_t *api = (sg_api_real_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);
  int length = 1;
  for (int i = 0; i < api->dim; ++i)
  {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_scale_conj(api->input_global_addr, 
                      api->output_global_addr, 
                      api->dim, 
                      length, 
                      (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_conj);

void tpu_kernel_api_conj_multi_core(const void *args)
{
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_conj_multi_core);