#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"

void nodechip_where_bcast (
global_addr_t out_global_addr,
global_addr_t cond_global_addr,
global_addr_t self_global_addr,
global_addr_t other_global_addr,
bool          self_is_scalar,
bool          other_is_scalar,
const dim4  * out_shape,
const dim4  * cond_shape,
const dim4  * self_shape,
const dim4  * other_shape,
data_type_t   cond_dtype,
data_type_t   dtype )
{
  dim4 out_global_stride, cond_global_stride, self_global_stride, other_global_stride;
  tpu_continuous_stride ( &out_global_stride, out_shape );
  tpu_continuous_stride ( &cond_global_stride, cond_shape );
  if ( self_is_scalar == false )
  {
    tpu_continuous_stride ( &self_global_stride, self_shape );
  }
  if ( other_is_scalar == false )
  {
    tpu_continuous_stride ( &other_global_stride, other_shape );
  }
  const bool cond_bcast[4] =
  {
    cond_shape->n != out_shape->n,
    cond_shape->c != out_shape->c,
    cond_shape->h != out_shape->h,
    cond_shape->w != out_shape->w
  };
  const bool self_bcast[4] =
  {
    self_is_scalar ? false : self_shape->n != out_shape->n,
    self_is_scalar ? false : self_shape->c != out_shape->c,
    self_is_scalar ? false : self_shape->h != out_shape->h,
    self_is_scalar ? false : self_shape->w != out_shape->w
  };
  const bool other_bcast[4] =
  {
    other_is_scalar ? false : other_shape->n != out_shape->n,
    other_is_scalar ? false : other_shape->c != out_shape->c,
    other_is_scalar ? false : other_shape->h != out_shape->h,
    other_is_scalar ? false : other_shape->w != out_shape->w
  };
  bool cond_bcast_all = false, self_bcast_all = false, other_bcast_all = false;
  for ( int i = 0; i < 4; ++i )
  {
    cond_bcast_all = cond_bcast_all || cond_bcast[i];
    self_bcast_all = self_bcast_all || self_bcast[i];
    other_bcast_all = other_bcast_all || other_bcast[i];
  }
  const int c_per_npu = DIV_UP ( out_shape->c, NPU_NUM );
  int hmax = out_shape->h, nmax = out_shape->n, cmax = c_per_npu * NPU_NUM;
  local_addr_t out_addr, cond_addr, self_addr, other_addr;
  while ( true )
  {
    out_addr = 0;
    int out_size = tpu_aligned_feature_size ( hmax, out_shape->w, dtype ) * DIV_UP ( cmax, NPU_NUM ) * nmax;
    cond_addr = out_addr + out_size;
    int cond_size = tpu_aligned_feature_size ( hmax, out_shape->w, cond_dtype ) * DIV_UP ( cmax, NPU_NUM ) * nmax;
    self_addr = cond_addr + cond_size;
    int self_size = self_is_scalar ? 0 : out_size;
    other_addr = self_addr + self_size;
    int other_size = other_is_scalar ? 0 : out_size;
    int total_size = other_addr + other_size;
    if ( total_size <= LOCAL_MEM_SIZE )
    {
      break;
    }
    else
    {
      if ( cmax > NPU_NUM )
      {
        cmax -= NPU_NUM;
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
  variable_t cond_var = { .type = TENSOR, .context = { .addr = cond_addr } };
  variable_t zero_var = { .type = SCALAR, .context = { .scalar = { .u32 = 0 } } };
  variable_t self_var;
  if ( self_is_scalar )
  {
    self_var.type = SCALAR;
    tpu_invalidate_cache ( self_global_addr, 64 );
    scalar_t val = { .u32 = * ( unsigned int * ) tpu_global_mem_addr ( self_global_addr ) };
    self_var.context.scalar = val;
  }
  else
  {
    self_var.type = TENSOR;
    self_var.context.addr = self_addr;
  }
  variable_t other_var;
  if ( other_is_scalar )
  {
    other_var.type = SCALAR;
    tpu_invalidate_cache ( other_global_addr, 64 );
    scalar_t val = { .u32 = * ( unsigned int * ) tpu_global_mem_addr ( other_global_addr ) };
    other_var.context.scalar = val;
  }
  else
  {
    other_var.type = TENSOR;
    other_var.context.addr = other_addr;
  }
  dim4 shape = { .w = out_shape->w };
  dim4 cond_local_shape, self_local_shape, other_local_shape;
  dim4 cond_local_stride, self_local_stride, other_local_stride;
  int ctodo = out_shape->c, cdone = 0;
  while ( ctodo > 0 )
  {
    shape.c = MIN ( ctodo, cmax );
    int ntodo = out_shape->n, ndone = 0;
    while ( ntodo > 0 )
    {
      shape.n = MIN ( ntodo, nmax );
      int htodo = out_shape->h, hdone = 0;
      while ( htodo > 0 )
      {
        shape.h = MIN ( htodo, hmax );
        // Move cond from global memory to local memory
        tpu_aligned_stride ( &cond_local_stride, 0, &shape, cond_dtype );
        cond_local_shape.n = cond_bcast[0] ? 1 : shape.n;
        cond_local_shape.c = cond_bcast[1] ? 1 : shape.c;
        cond_local_shape.h = cond_bcast[2] ? 1 : shape.h;
        cond_local_shape.w = cond_bcast[3] ? 1 : shape.w;
        global_addr_t cond_global_addr_gdma =
        cond_global_addr + (
        ( cond_bcast[0] ? 0 : ndone ) * cond_global_stride.n +
        ( cond_bcast[1] ? 0 : cdone ) * cond_global_stride.c +
        ( cond_bcast[2] ? 0 : hdone ) * cond_global_stride.h ) * tpu_data_type_size ( cond_dtype );
        tpu_gdma_cpy_S2L ( cond_addr, cond_global_addr_gdma, &cond_local_shape, &cond_local_stride, &cond_global_stride, cond_dtype );
        // Move self from global memory to local memory
        if ( self_is_scalar == false )
        {
          tpu_aligned_stride ( &self_local_stride, 0, &shape, dtype );
          self_local_shape.n = self_bcast[0] ? 1 : shape.n;
          self_local_shape.c = self_bcast[1] ? 1 : shape.c;
          self_local_shape.h = self_bcast[2] ? 1 : shape.h;
          self_local_shape.w = self_bcast[3] ? 1 : shape.w;
          global_addr_t self_global_addr_gdma =
          self_global_addr + (
          ( self_bcast[0] ? 0 : ndone ) * self_global_stride.n +
          ( self_bcast[1] ? 0 : cdone ) * self_global_stride.c +
          ( self_bcast[2] ? 0 : hdone ) * self_global_stride.h ) * tpu_data_type_size ( dtype );
          tpu_gdma_cpy_S2L ( self_addr, self_global_addr_gdma, &self_local_shape, &self_local_stride, &self_global_stride, dtype );
        }
        // Move other from global memory to local memory
        if ( other_is_scalar == false )
        {
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
        }
        // Broadcast cond if needed
        if ( cond_bcast[1] )
        {
          cond_local_shape.c = NPU_NUM;
          tpu_bdc_npu_bcast ( cond_addr, cond_addr, &cond_local_shape, cond_dtype );
        }
        if ( cond_bcast[0] || cond_bcast[2] || cond_bcast[3] || ( cond_bcast[1] && shape.c > NPU_NUM ) )
        {
          dim4 cond_bcast_stride;
          cond_bcast_stride.n = cond_bcast[0] ? 0 : cond_local_stride.n;
          cond_bcast_stride.c = cond_bcast[1] ? 0 : cond_local_stride.c;
          cond_bcast_stride.h = cond_bcast[2] ? 0 : cond_local_stride.h;
          cond_bcast_stride.w = cond_bcast[3] ? 0 : cond_local_stride.w;
          tpu_bdc_cpy ( cond_addr, cond_addr, &shape, NULL, &cond_bcast_stride, cond_dtype );
        }
        // Broadcast self if needed
        if ( self_is_scalar == false )
        {
          if ( self_bcast[1] )
          {
            self_local_shape.c = NPU_NUM;
            tpu_bdc_npu_bcast ( self_addr, self_addr, &self_local_shape, dtype );
          }
          if ( self_bcast[0] || self_bcast[2] || self_bcast[3] || ( self_bcast[1] && shape.c > NPU_NUM ) )
          {
            dim4 self_bcast_stride;
            self_bcast_stride.n = self_bcast[0] ? 0 : self_local_stride.n;
            self_bcast_stride.c = self_bcast[1] ? 0 : self_local_stride.c;
            self_bcast_stride.h = self_bcast[2] ? 0 : self_local_stride.h;
            self_bcast_stride.w = self_bcast[3] ? 0 : self_local_stride.w;
            tpu_bdc_cpy ( self_addr, self_addr, &shape, NULL, &self_bcast_stride, dtype );
          }
        }
        // Broadcast other if needed
        if ( other_is_scalar == false )
        {
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
        }
        // Select
        tpu_bdc_equal_select ( out_addr, &cond_var, &zero_var, &other_var, &self_var, &shape, cond_dtype, dtype );
        // Move out from local memory to global memory
        global_addr_t out_global_addr_gdma =
        out_global_addr + (
        ndone * out_global_stride.n +
        cdone * out_global_stride.c +
        hdone * out_global_stride.h ) * tpu_data_type_size ( dtype );
        tpu_gdma_cpy_L2S ( out_global_addr_gdma, out_addr, &shape, &out_global_stride, NULL, dtype );
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
void tpu_kernel_api_where ( const void * args )
{
  sg_api_where_t * api = ( sg_api_where_t * ) args;
  TPUKERNEL_ASSERT ( api->shape_dim > 0 && api->shape_dim <= 4 );
  dim4 out_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 cond_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 self_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  dim4 other_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
  if ( api->shape_dim >= 1 )
  {
    out_shape.n = api->out_shape[0];
    cond_shape.n = api->cond_shape[0];
    self_shape.n = api->self_is_scalar ? 0 : api->self_shape[0];
    other_shape.n = api->other_is_scalar ? 0 : api->other_shape[0];
  }
  if ( api->shape_dim >= 2 )
  {
    out_shape.c = api->out_shape[1];
    cond_shape.c = api->cond_shape[1];
    self_shape.c = api->self_is_scalar ? 0 : api->self_shape[1];
    other_shape.c = api->other_is_scalar ? 0 : api->other_shape[1];
  }
  if ( api->shape_dim >= 3 )
  {
    out_shape.h = api->out_shape[2];
    cond_shape.h = api->cond_shape[2];
    self_shape.h = api->self_is_scalar ? 0 : api->self_shape[2];
    other_shape.h = api->other_is_scalar ? 0 : api->other_shape[2];
  }
  if ( api->shape_dim >= 4 )
  {
    out_shape.w = api->out_shape[3];
    cond_shape.w = api->cond_shape[3];
    self_shape.w = api->self_is_scalar ? 0 : api->self_shape[3];
    other_shape.w = api->other_is_scalar ? 0 : api->other_shape[3];
  }
  tpu_initialize();
  nodechip_where_bcast (
  api->out_global_addr,
  api->cond_global_addr,
  api->self_global_addr,
  api->other_global_addr,
  api->self_is_scalar,
  api->other_is_scalar,
  &out_shape,
  &cond_shape,
  api->self_is_scalar ? NULL : &self_shape,
  api->other_is_scalar ? NULL : &other_shape,
  tpu_type_convert ( ( sg_data_type_t ) api->cond_dtype ),
  tpu_type_convert ( ( sg_data_type_t ) api->dtype ) );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_where );
