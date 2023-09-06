#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "stdio.h"

/*
 * output = input + value * ( other )
 */

void nodechip_scale_real(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    int dim,
    int length,
    const int   * in_stride_org,
    const int   * out_stride_org, 
    data_type_t dtype)
{
  // int in_stride[FW_MAX_SHAPE_DIMS];
  // int out_stride[FW_MAX_SHAPE_DIMS];
  // for ( int i = 0; i < dim; ++i )
  // {
  //   //shape[i] = shape_org[i];
  //   in_stride[i] = in_stride_org[i];
  //   printf("  in_stride[%d]: %d\n", i, in_stride[i]);
  //   out_stride[i] = out_stride_org[i];
  //   printf("  out_stride[%d]: %d\n", i, out_stride[i]);
  // }
  const int dsize = tpu_data_type_size(dtype);
  int wmax = DIV_UP(length, NPU_NUM);
  //printf(" length=%d NPU_NUM=%d wmax=%d\n",length, NPU_NUM, wmax);
  local_addr_t input_local_addrs[2], output_local_addrs[2];
  local_addr_t next = 0;
  while (true)
  {
    next = 0;
    int size = tpu_aligned_feature_size(1, wmax, dtype);
    // printf("  input_local_addrs[0]: %u\n",next);
    input_local_addrs[0] = next;
    next += size;
    // printf("  input_local_addrs[1]: %u\n",next);
    input_local_addrs[1] = next;
    next += size;
    // printf("  output_local_addrs[0]: %u\n",next);
    output_local_addrs[0] = next;
    next += size;
    // printf("  output_local_addrs[1]: %u\n",next);
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
  // printf("length: %d\n",todo);
  int done = 0;
  dim4 shape = {.n = 1, .h = 1};
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  //dim4 copy_in_stride = { .n = 24, .c = 2, .h = 2, .w = 2 };
  dim4 copy_in_stride;
  //dim4 copy_out_stride = { .n = 1, .c = NPU_NUM, .h = 1, .w = in_stride[1]/2 };
  //input_global_addr+=dsize;    //invert real to imag
  while (todo != 0)
  {
    // printf("  todo=%d\n",todo);
    if (todo > NPU_NUM)  //....................................
    {
      shape.c = NPU_NUM;
      // printf("  shape.c=%d\n",shape.c);
      shape.w = MIN(todo / NPU_NUM, wmax);
            //shape.w = 1;
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
    // printf("  shape= %d,%d,%d,%d\n",shape.n,shape.c,shape.h,shape.w);
    //tpu_gdma_cpy_S2L(input_local_addrs[index], input_global_addr + done * dsize, &shape, NULL, NULL, dtype);
    tpu_gdma_cpy_S2L(input_local_addrs[index], input_global_addr + done * dsize*2, &shape, NULL, &copy_in_stride, dtype);
    if (tpu_is_parallel_state())
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if (l2s)
    {
      //tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, &copy_out_stride, &copy_in_stride, dtype);
      tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
    }
    // printf("    midst output_local_addrs[1]: %u\n",next);
    // printf("    output_stride[-1]: %d\n",out_stride[8]);
    //tpu_bdc_neg(output_local_addrs[index], input_local_addrs[index], &shape, in_stride, out_stride, dtype);
    
    // printf(" copy_in_stride: .n=%d, .c=%d, .h=%d, .w=%d\n", copy_in_stride.n, copy_in_stride.c, copy_in_stride.h, copy_in_stride.w);
    //printf(" copy_out_stride: .c=%d, .w=%d\n", copy_out_stride.c, copy_out_stride.w);
    
    //tpu_gdma_cpy_L2L(output_local_addrs[index], input_local_addrs[index], &shape, &copy_out_stride, &copy_in_stride, dtype);
    tpu_gdma_cpy_L2L(output_local_addrs[index], input_local_addrs[index], &shape, NULL, NULL, dtype);
    l2s = true;
    l2s_global_addr = output_global_addr + done * dsize;
    l2s_local_addr = output_local_addrs[index];
    l2s_shape = shape;
    todo -= shape.c * shape.w ;
    done += shape.c * shape.w ;
    index = 1 - index;
    // printf("  l2s=%d  todo=%d done=%d l2s_global_addr=%llu\n", l2s, todo, done, l2s_global_addr);
  }
  if (tpu_is_parallel_state())
  {
    tpu_parallel_end();
  }
  if (l2s)
  {
    //tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, &copy_out_stride, &copy_in_stride, dtype);
    tpu_gdma_cpy_L2S(l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype);
  }
}

void tpu_kernel_api_real(const void *args)
{
  sg_api_real_t *api = (sg_api_real_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16);
  // printf("tpu_kernel_api_real:\n  api->dtype: %d  api->dim: %d\n", api->dtype, api->dim);
  int length = 1;
  for (int i = 0; i < api->dim; ++i)
  {
    length *= api->shape[i];
    // printf("  api->shape[%d]: %d  api->input_stride: %d api->output_stride: %d\n", i, api->shape[i], api->input_stride[i], api->output_stride[i]);
  }
  tpu_initialize();
  nodechip_scale_real(api->input_global_addr, 
                      api->output_global_addr, 
                      api->dim, 
                      length, 
                      api->input_stride, 
                      api->output_stride, 
                      (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_real);

void tpu_kernel_api_real_multi_core(const void *args)
{
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_real_multi_core);