#include "sg_api_struct.h"
#include "tpu_kernel.h"

static inline bool is_contiguous ( const int * shape, const int * stride, int dim )
{
  int s = 1;
  for ( int i = dim - 1; i >= 0; --i )
  {
    if ( stride[i] != s )
    {
      return false;
    }
    s *= shape[i];
  }
  return true;
}

static inline void simplify ( int * shape, int * src_stride, int * dst_stride, int * dim )
{
  for ( int i = 0; i < ( *dim ) - 1; )
  {
    if ( src_stride[i + 1] * shape[i + 1] == src_stride[i] &&
         dst_stride[i + 1] * shape[i + 1] == dst_stride[i] )
    {
      shape[i] *= shape[i + 1];
      for ( int j = i + 1; j < ( *dim ) - 1; ++j )
      {
        shape[j] = shape[j + 1];
      }
      for ( int j = i; j < ( *dim ) - 1; ++j )
      {
        src_stride[j] = src_stride[j + 1];
        dst_stride[j] = dst_stride[j + 1];
      }
      -- ( *dim );
    }
    else
    {
      ++i;
    }
  }
}

void nodechip_strided_copy (
global_addr_t in_global_addr,
global_addr_t out_global_addr,
int           dim,
const int   * shape_org,
const int   * in_stride_org,
const int   * out_stride_org,
data_type_t   dtype )
{
  const int dsize = tpu_data_type_size ( dtype );
  int shape[FW_MAX_SHAPE_DIMS];
  int in_stride[FW_MAX_SHAPE_DIMS];
  int out_stride[FW_MAX_SHAPE_DIMS];
  TPUKERNEL_ASSERT ( dim > 0 && dim < FW_MAX_SHAPE_DIMS );
  for ( int i = 0; i < dim; ++i )
  {
    shape[i] = shape_org[i];
    in_stride[i] = in_stride_org[i];
    out_stride[i] = out_stride_org[i];
  }
  simplify ( shape, in_stride, out_stride, &dim );
  if ( dim == 2 && in_stride[0] == 1 && out_stride[1] == 1 )
  {
    dim4 copy_in_stride = { .n = 1, .c = in_stride[1], .h = 1, .w = in_stride[0] };
    dim4 copy_out_stride = { .n = 1, .c = out_stride[0], .h = 1, .w = out_stride[1] };
    int cmax = 32768;
    int wmax = 32768;
    dim4 copy_shape = { .n = 1, .h = 1 };
    int ctodo = shape[0];
    int cdone = 0;
    while ( ctodo != 0 )
    {
      copy_shape.c = MIN ( ctodo, cmax );
      int wtodo = shape[1];
      int wdone = 0;
      while ( wtodo != 0 )
      {
        copy_shape.w = MIN ( wtodo, wmax );
        tpu_gdma_cpy_cw_trans_S2S ( out_global_addr + ( 1UL * cdone * copy_out_stride.c + 1UL * wdone * copy_out_stride.w ) * dsize, in_global_addr + ( 1UL * cdone * copy_in_stride.w + 1UL * wdone * copy_in_stride.c ) * dsize, &copy_shape, &copy_out_stride, &copy_in_stride, dtype );
        wtodo -= copy_shape.w;
        wdone += copy_shape.w;
      }
      ctodo -= copy_shape.c;
      cdone += copy_shape.c;
    }
  }
  else if ( dim == 3 && in_stride[1] == 1 && out_stride[2] == 1 )
  {
    dim4 copy_in_stride = { .n = in_stride[0], .c = in_stride[2], .h = 1, .w = in_stride[1] };
    dim4 copy_out_stride = { .n = out_stride[0], .c = out_stride[1], .h = 1, .w = out_stride[2] };
    int nmax = 32768;
    int cmax = 32768;
    int wmax = 32768;
    dim4 copy_shape = { .h = 1 };
    int ntodo = shape[0];
    int ndone = 0;
    while ( ntodo != 0 )
    {
      copy_shape.n = MIN ( ntodo, nmax );
      int ctodo = shape[1];
      int cdone = 0;
      while ( ctodo != 0 )
      {
        copy_shape.c = MIN ( ctodo, cmax );
        int wtodo = shape[2];
        int wdone = 0;
        while ( wtodo != 0 )
        {
          copy_shape.w = MIN ( wtodo, wmax );
          tpu_gdma_cpy_cw_trans_S2S ( out_global_addr + ( 1UL * ndone * copy_out_stride.n + 1UL * cdone * copy_out_stride.c + 1UL * wdone * copy_out_stride.w ) * dsize, in_global_addr + ( 1UL * ndone * copy_in_stride.n + 1UL * cdone * copy_in_stride.w + 1UL * wdone * copy_in_stride.c ) * dsize, &copy_shape, &copy_out_stride, &copy_in_stride, dtype );
          wtodo -= copy_shape.w;
          wdone += copy_shape.w;
        }
        ctodo -= copy_shape.c;
        cdone += copy_shape.c;
      }
      ntodo -= copy_shape.n;
      ndone += copy_shape.n;
    }
  }
  else if ( dim == 3 && in_stride[2] == 1 && out_stride[2] == 1 /* && in_stride[0] == shape[2] && out_stride[1] == shape[2] */ )
  {
    dim4 copy_shape = { .n = shape[0], .c = shape[1], .h = 1, .w = shape[2] };
    dim4 copy_in_stride = { .n = in_stride[1], .c = in_stride[0], .h = 1, .w = in_stride[2] };
    dim4 copy_out_stride = { .n = out_stride[0], .c = out_stride[1], .h = 1, .w = out_stride[2] };
    tpu_gdma_cpy_nc_trans_S2S (
    out_global_addr,
    in_global_addr,
    &copy_shape,
    &copy_out_stride,
    &copy_in_stride,
    dtype );
  }
  else if ( dim == 4 && in_stride[3] == 1 && out_stride[3] == 1 /* && in_stride[1] == shape[3] && out_stride[2] == shape[3] */ )
  {
    dim4 copy_shape = { .n = shape[1], .c = shape[2], .h = 1, .w = shape[3] };
    dim4 copy_in_stride = { .n = in_stride[2], .c = in_stride[1], .h = 1, .w = in_stride[3] };
    dim4 copy_out_stride = { .n = out_stride[1], .c = out_stride[2], .h = 1, .w = out_stride[3] };
    for ( int i = 0; i < shape[0]; ++i )
    {
      tpu_gdma_cpy_nc_trans_S2S (
      out_global_addr + i * out_stride[0] * tpu_data_type_size ( dtype ),
      in_global_addr + i * in_stride[0] * tpu_data_type_size ( dtype ),
      &copy_shape,
      &copy_out_stride,
      &copy_in_stride,
      dtype );
    }
  }
  else
  {
#if 0
    printf ( "****************************************************************\n" );
    printf ( "shape = [ " );
    for ( int i = 0; i < dim; ++i )
    {
      printf ( "%d ", shape[i] );
    }
    printf ( "]\n" );
    printf ( "in_stride = [ " );
    for ( int i = 0; i < dim; ++i )
    {
      printf ( "%d ", in_stride[i] );
    }
    printf ( "]\n" );
    printf ( "out_stride = [ " );
    for ( int i = 0; i < dim; ++i )
    {
      printf ( "%d ", out_stride[i] );
    }
    printf ( "]\n" );
    printf ( "****************************************************************\n" );
#endif
    dim4 copy_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 copy_in_stride = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 copy_out_stride = { .n = 1, .c = 1, .h = 1, .w = 1 };
    TPUKERNEL_ASSERT ( dim >= 1 && dim <= 4 );
    if ( dim >= 1 )
    {
      copy_shape.n = shape[0];
      copy_in_stride.n = in_stride[0];
      copy_out_stride.n = out_stride[0];
    }
    if ( dim >= 2 )
    {
      copy_shape.c = shape[1];
      copy_in_stride.c = in_stride[1];
      copy_out_stride.c = out_stride[1];
    }
    if ( dim >= 3 )
    {
      copy_shape.h = shape[2];
      copy_in_stride.h = in_stride[2];
      copy_out_stride.h = out_stride[2];
    }
    if ( dim >= 4 )
    {
      copy_shape.w = shape[3];
      copy_in_stride.w = in_stride[3];
      copy_out_stride.w = out_stride[3];
    }
    if ( copy_in_stride.w <= 128 / tpu_data_type_size ( dtype ) &&
         copy_out_stride.w <= 128 / tpu_data_type_size ( dtype ) )
    {
      tpu_gdma_cpy_S2S (
      out_global_addr,
      in_global_addr,
      &copy_shape,
      &copy_out_stride,
      &copy_in_stride,
      dtype );
    }
    else
    {
      int copy_shape_n = copy_shape.n;
      copy_shape.n = copy_shape.c;
      copy_shape.c = copy_shape.h;
      copy_shape.h = copy_shape.w;
      copy_shape.w = 1;
      int copy_in_stride_n = copy_in_stride.n;
      copy_in_stride.n = copy_in_stride.c;
      copy_in_stride.c = copy_in_stride.h;
      copy_in_stride.h = copy_in_stride.w;
      copy_in_stride.w = 1;
      int copy_out_stride_n = copy_out_stride.n;
      copy_out_stride.n = copy_out_stride.c;
      copy_out_stride.c = copy_out_stride.h;
      copy_out_stride.h = copy_out_stride.w;
      copy_out_stride.w = 1;
      for ( int i = 0; i < copy_shape_n; i++ ) {
        tpu_gdma_cpy_S2S (
        out_global_addr + i * copy_out_stride_n * tpu_data_type_size ( dtype ),
        in_global_addr + i * copy_in_stride_n * tpu_data_type_size ( dtype ),
        &copy_shape,
        &copy_out_stride,
        &copy_in_stride,
        dtype );
      }
    }
  }
}

void tpu_kernel_api_strided_copy ( const void *args )
{
  sg_api_strided_copy_t *api = ( sg_api_strided_copy_t* ) args;
  tpu_initialize();
  nodechip_strided_copy (
  api->input_global_addr,
  api->output_global_addr,
  api->dim,
  api->shape,
  api->input_stride,
  api->output_stride,
  ( data_type_t ) api->dtype );
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_strided_copy );
