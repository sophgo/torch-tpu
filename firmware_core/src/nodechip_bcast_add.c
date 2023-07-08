#include "sg_api_struct.h"
#include "tpu_kernel.h"

void nodechip_bcast_add (
global_addr_t output_global_addr,
global_addr_t input_global_addr,
global_addr_t other_global_addr,
const dim4  * output_shape,
const dim4  * input_shape,
const dim4  * other_shape,
data_type_t   dtype )
{
  dim4 output_global_stride, input_global_stride, other_global_stride;
  tpu_continuous_stride ( &output_global_stride, output_shape );
  tpu_continuous_stride ( &input_global_stride, input_shape );
  tpu_continuous_stride ( &other_global_stride, other_shape );
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
  bool input_bcast_all = false, other_bcast_all = false;
  for ( int i = 0; i < 4; ++i )
  {
    input_bcast_all = input_bcast_all || input_bcast[i];
    other_bcast_all = other_bcast_all || other_bcast[i];
  }
  const int c_per_npu = DIV_UP ( output_shape->c, NPU_NUM );
  int hmax = output_shape->h, nmax = output_shape->n, cmax = c_per_npu * NPU_NUM;
  local_addr_t output_addr, input_addr, other_addr;
  while ( true )
  {
    output_addr = 0;
    int output_size = tpu_aligned_feature_size ( hmax, output_shape->w, dtype ) * DIV_UP ( cmax, NPU_NUM ) * nmax;
    input_addr = output_addr + output_size;
    int input_size = output_size;
    other_addr = input_addr + input_size;
    int other_size = output_size;
    int total_size = other_addr + other_size;
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
  dim4 input_local_shape, other_local_shape;
  dim4 input_local_stride, other_local_stride;
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
        tpu_aligned_stride ( &input_local_stride, 0, &shape, dtype );
        input_local_shape.n = input_bcast[0] ? 1 : shape.n;
        input_local_shape.c = input_bcast[1] ? 1 : shape.c;
        input_local_shape.h = input_bcast[2] ? 1 : shape.h;
        input_local_shape.w = input_bcast[3] ? 1 : shape.w;
        global_addr_t input_global_addr_gdma =
        input_global_addr + (
        ( input_bcast[0] ? 0 : ndone ) * input_global_stride.n +
        ( input_bcast[1] ? 0 : cdone ) * input_global_stride.c +
        ( input_bcast[2] ? 0 : hdone ) * input_global_stride.h ) * tpu_data_type_size ( dtype );
        tpu_gdma_cpy_S2L ( input_addr, input_global_addr_gdma, &input_local_shape, &input_local_stride, &input_global_stride, dtype );
        // Move other from global memory to local memory
        tpu_aligned_stride ( &other_local_stride, 0, &shape, dtype );
        other_local_shape.n = other_bcast[0] ? 1 : shape.n;
        other_local_shape.c = other_bcast[1] ? 1 : shape.c;
        other_local_shape.h = other_bcast[2] ? 1 : shape.h;
        other_local_shape.w = other_bcast[3] ? 1 : shape.w;
        global_addr_t other_global_addr_gdma =
        other_global_addr + (
        ( other_bcast[0] ? 0 : ndone ) * other_global_stride.n +
        ( other_bcast[1] ? 0 : cdone ) * other_global_stride.c +
        ( other_bcast[2] ? 0 : hdone ) * other_global_stride.h ) * tpu_data_type_size ( dtype );
        tpu_gdma_cpy_S2L ( other_addr, other_global_addr_gdma, &other_local_shape, &other_local_stride, &other_global_stride, dtype );
        // Broadcast input if needed
        if ( input_bcast[1] )
        {
          input_local_shape.c = NPU_NUM;
          tpu_bdc_npu_bcast ( input_addr, input_addr, &input_local_shape, dtype );
        }
        if ( input_bcast[0] || input_bcast[2] || input_bcast[3] || ( input_bcast[1] && shape.c > NPU_NUM ) )
        {
          dim4 input_bcast_stride;
          input_bcast_stride.n = input_bcast[0] ? 0 : input_local_stride.n;
          input_bcast_stride.c = input_bcast[1] ? 0 : input_local_stride.c;
          input_bcast_stride.h = input_bcast[2] ? 0 : input_local_stride.h;
          input_bcast_stride.w = input_bcast[3] ? 0 : input_local_stride.w;
          tpu_bdc_cpy ( input_addr, input_addr, &shape, NULL, &input_bcast_stride, dtype );
        }
        // Broadcast other if needed
        if ( other_bcast[1] )
        {
          other_local_shape.c = NPU_NUM;
          tpu_bdc_npu_bcast ( other_addr, other_addr, &other_local_shape, dtype );
        }
        if ( other_bcast[0] || other_bcast[2] || other_bcast[3] || ( other_bcast[1] && shape.c > NPU_NUM ) )
        {
          dim4 other_bcast_stride;
          other_bcast_stride.n = other_bcast[0] ? 0 : other_local_stride.n;
          other_bcast_stride.c = other_bcast[1] ? 0 : other_local_stride.c;
          other_bcast_stride.h = other_bcast[2] ? 0 : other_local_stride.h;
          other_bcast_stride.w = other_bcast[3] ? 0 : other_local_stride.w;
          tpu_bdc_cpy ( other_addr, other_addr, &shape, NULL, &other_bcast_stride, dtype );
        }
        // Select
        tpu_bdc_fp_add ( output_addr, input_addr, other_addr, &shape, NULL, NULL, NULL, dtype );
        // Move out from local memory to global memory
        global_addr_t output_global_addr_gdma =
        output_global_addr + (
        ndone * output_global_stride.n +
        cdone * output_global_stride.c +
        hdone * output_global_stride.h ) * tpu_data_type_size ( dtype );
        tpu_gdma_cpy_L2S ( output_global_addr_gdma, output_addr, &shape, &output_global_stride, NULL, dtype );
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

void tpu_kernel_api_bcast_add ( const void * args )
{
  sg_api_bcast_add_t * api = ( sg_api_bcast_add_t * ) args;
  TPUKERNEL_ASSERT ( api->dim > 0 && api->dim <= 4 );
  dim4 output_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 input_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 other_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  if ( api->dim >= 1 )
  {
    input_shape.n = api->input_shape[0];
    other_shape.n = api->other_shape[0];
    output_shape.n = input_shape.n > other_shape.n ? input_shape.n : other_shape.n;
  }
  if ( api->dim >= 2 )
  {
    input_shape.c = api->input_shape[1];
    other_shape.c = api->other_shape[1];
    output_shape.c = input_shape.c > other_shape.c ? input_shape.c : other_shape.c;
  }
  if ( api->dim >= 3 )
  {
    input_shape.h = api->input_shape[2];
    other_shape.h = api->other_shape[2];
    output_shape.h = input_shape.h > other_shape.h ? input_shape.h : other_shape.h;
  }
  if ( api->dim >= 4 )
  {
    input_shape.w = api->input_shape[3];
    other_shape.w = api->other_shape[3];
    output_shape.w = input_shape.w > other_shape.w ? input_shape.w : other_shape.w;
  }
  tpu_initialize();
  nodechip_bcast_add (
  api->output_global_addr,
  api->input_global_addr,
  api->other_global_addr,
  &output_shape,
  &input_shape,
  &other_shape,
  ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_bcast_add );

extern void nodechip_binary_multi_core(
    global_addr_t A_global_addr,
    global_addr_t B_global_addr,
    global_addr_t res_global_addr,
    const int* A_shape,
    const int* B_shape,
    int A_dim,
    int B_dim,
    int binary_type,          // 0: add, 1: sub, 2: mul, 3:div
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit);

void tpu_kernel_api_binary_multi_core(const void* api_buf) {
    sg_api_binary_multi_core_t *api = (sg_api_binary_multi_core_t *)api_buf;
    TPUKERNEL_ASSERT(api->in0_scale == 1.f && api->in1_scale == 1.f);
    int ashape[FW_MAX_SHAPE_DIMS], bshape[FW_MAX_SHAPE_DIMS];
    for (int i = 0; i < api->in0_dims; ++i) {
      ashape[i] = api->in0_shape[i];
    }
    for (int i = 0; i < api->in1_dims; ++i) {
      bshape[i] = api->in1_shape[i];
    }
    int r_dim = api->in0_dims > api->in1_dims ? api->in0_dims : api->in1_dims;
    int add_dim = r_dim < 4 ? 4 - r_dim : 0;
    for (int i = 0; i < add_dim; ++i) {
      ashape[api->in0_dims + i] = 1;
      bshape[api->in1_dims + i] = 1;
    }

    tpu_initialize();
    nodechip_binary_multi_core(
        api->input0_addr,
        api->input1_addr,
        api->output_addr,
        ashape,
        bshape,
        api->in0_dims + add_dim,
        api->in1_dims + add_dim,
        api->binary_type,
        (data_type_t)api->dtype,
        0,
        -1.f);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_binary_multi_core);