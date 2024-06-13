#include "sg_api_struct.h"
#include "tpu_kernel.h"


static
void nodechip_bcbinary_fp(
  global_addr_t input_global_addr,
  global_addr_t other_global_addr,
  global_addr_t output_global_addr,
  const dim4 * input_shape,
  const dim4 * other_shape,
  const dim4 * output_shape,
  int binary_type,
  data_type_t dtype,
  int if_relu,
  float relu_upper_limit)
  {
    dim4 input_global_stride, other_global_stride, output_global_stride;
    tpu_continuous_stride ( &input_global_stride, input_shape );
    tpu_continuous_stride ( &other_global_stride, other_shape );
    tpu_continuous_stride ( &output_global_stride, output_shape );
    const bool input_bcast[4] =
    {
      input_shape->n != output_shape->n,
      input_shape->c != output_shape->c,
      input_shape->h != output_shape->h,
      input_shape->w != output_shape->w
    };
    const bool other_bcast[4] =
    {
      other_shape->n != output_shape->n,
      other_shape->c != output_shape->c,
      other_shape->h != output_shape->h,
      other_shape->w != output_shape->w
    };
    const int c_per_npu = DIV_UP ( output_shape->c, NPU_NUM );
    int hmax = output_shape->h, nmax = output_shape->n, cmax = c_per_npu * NPU_NUM;
    local_addr_t input_local_addr, other_local_addr, output_local_addr;
    local_addr_t next = 0;
    while ( true )
    {
      next = 0;
      int size = tpu_aligned_feature_size ( hmax, output_shape->w, dtype ) * DIV_UP ( cmax, NPU_NUM ) * nmax;
      input_local_addr = next; next += size;
      other_local_addr = next; next += size;
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
    dim4 input_local_shape,other_local_shape;
    dim4 input_local_stride, other_local_stride;
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
          // Move other from global memory to local memory
          other_local_shape.n = other_bcast[0] ? 1 : shape.n;
          other_local_shape.c = other_bcast[1] ? 1 : shape.c;
          other_local_shape.h = other_bcast[2] ? 1 : shape.h;
          other_local_shape.w = other_bcast[3] ? 1 : shape.w;
          tpu_aligned_stride ( &other_local_stride, 0, &other_local_shape, dtype );
          global_addr_t other_global_addr_gdma =
          other_global_addr + (
          ( other_bcast[0] ? 0 : ndone ) * other_global_stride.n +
          ( other_bcast[1] ? 0 : cdone ) * other_global_stride.c +
          ( other_bcast[2] ? 0 : hdone ) * other_global_stride.h ) * tpu_data_type_size ( dtype );
          tpu_gdma_cpy_S2L ( other_local_addr, other_global_addr_gdma, &other_local_shape, &other_local_stride, &other_global_stride, dtype );
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
          // Broadcast other if needed
          if ( other_bcast[1] )
          {
            other_local_shape.c = NPU_NUM;
            tpu_bdc_npu_bcast ( other_local_addr, other_local_addr, &other_local_shape, dtype );
          }
          if ( other_bcast[0] || other_bcast[2] || other_bcast[3] || ( other_bcast[1] && shape.c > NPU_NUM ) )
          {
            other_local_stride.n = other_bcast[0] ? 0 : other_local_stride.n;
            other_local_stride.c = other_bcast[1] ? 0 : other_local_stride.c;
            other_local_stride.h = other_bcast[2] ? 0 : other_local_stride.h;
            other_local_stride.w = other_bcast[3] ? 0 : other_local_stride.w;
          }
          //Compute
          if(binary_type == 0){
            tpu_bdc_fp_add ( output_local_addr, input_local_addr, other_local_addr, &shape, NULL, &input_local_stride, &other_local_stride, dtype );
            //sub can be implemented by add
          }else if(binary_type == 2){
            tpu_bdc_fp_mul ( output_local_addr, input_local_addr, other_local_addr, &shape, NULL, &input_local_stride, &other_local_stride, dtype );
          }else if(binary_type == 3){
            tpu_bdc_fp_div ( output_local_addr, input_local_addr, other_local_addr, &shape, NULL, &input_local_stride, &other_local_stride, dtype );
          }
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

int tpu_kernel_api_arithmetic_eltwise ( const void *args )
{
  sg_api_arithmetic_eltwise_t *api = ( sg_api_arithmetic_eltwise_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );

  dim4 input_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 other_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 output_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };

  if(api->input_dim>=1) {input_shape.n = api->input_shape[0];}
  if(api->other_dim>=1) {other_shape.n = api->other_shape[0];}

  if(api->input_dim>=2) {input_shape.c = api->input_shape[1];}
  if(api->other_dim>=2) {other_shape.c = api->other_shape[1];}

  if(api->input_dim>=3) {input_shape.h = api->input_shape[2];}
  if(api->other_dim>=3) {other_shape.h = api->other_shape[2];}

  if(api->input_dim>=4) {input_shape.w = api->input_shape[3];}
  if(api->other_dim>=4) {other_shape.w = api->other_shape[3];}

  output_shape.n = input_shape.n > other_shape.n ? input_shape.n : other_shape.n ;
  output_shape.c = input_shape.c > other_shape.c ? input_shape.c : other_shape.c ;
  output_shape.h = input_shape.h > other_shape.h ? input_shape.h : other_shape.h ;
  output_shape.w = input_shape.w > other_shape.w ? input_shape.w : other_shape.w ;

  tpu_initialize();
  nodechip_bcbinary_fp(
      api->input_global_addr,
      api->other_global_addr,
      api->output_global_addr,
      &input_shape,
      &other_shape,
      &output_shape,
      api->binary_type, // 0:add, 1: sub, 2: mul, 3: div
      (data_type_t)api->dtype,
      0,
      1);
  tpu_poll();
  return 0;
}

// #define TPUKERNEL_FUNC_REGISTER(func) 
//   __attribute__((constructor)) void func##_eltwise() { tpu_kernel_(#func, func); }


// #define ELTWISE(op_name) 
//   auto tpu_kernel_api_##op_name##_eltwise = tpu_kernel_api_eltwise;
//   TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_##op_name##_eltwise);

// // we need mul and div temporarily here
// // ELTWISE(add);
// // ELTWISE(sub);
// ELTWISE(mul);
// ELTWISE(div);

// TPUKERNEL_FUNC_REGISTER (tpu_kernel_api_div_eltwise);

#ifdef BACKEND_SG2260
int tpu_kernel_api_arithmetic_eltwise_multi_core(const void *args)
{
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_arithmetic_eltwise_multi_core);
#endif