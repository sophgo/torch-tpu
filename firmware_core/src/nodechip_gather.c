#include "sg_api_struct.h"
#include "tpu_kernel.h"


static void nodechip_gather(
  global_addr_t input_global_addr,
  global_addr_t index_global_addr,
  global_addr_t output_global_addr,
  const int *input_shape,
  const int *index_shape,
  int shape_dims,
  int axis, // axis to do index_select
  data_type_t dtype){
    int index_num = index_shape[axis];
    // check shape
    for (int i = 0; i < shape_dims; ++i)
    {
        if ((i != axis && input_shape[i] != index_shape[i]))
            TPUKERNEL_ASSERT("gather shape error");
    }
    dim4 ishape = {1, 1, 1, 1};
    dim4 index_ishape = {1, 1, 1, 1};
    // e.g. input_shape = (a, b, c, d, e)
    // aixs = 0 -> ishape = (1, 1, a, b*c*d*e)
    // aixs = 1 -> ishape = (1, a, b, c*d*e)
    // axis = 2 -> ishape = (1, a*b, c, d*e)
    // axis = 3 -> ishape = (1, a*b*c, d, e)
    // axis = 4 -> ishape = (1, a*b*c*d, e, 1)
    // if(axis != shape_dims - 1){
    //   tpu_gdma_cpy_cw_trans_S2S(input_global_addr,input_global_addr,&ishape,NULL,NULL,dtype);
    //   tpu_gdma_cpy_cw_trans_S2S(input_global_addr,input_global_addr,&ishape,NULL,NULL,dtype);
    // }
    for (int i = axis + 1; i < shape_dims; ++i)
    {
        ishape.w *= input_shape[i];
        index_ishape.w *= index_shape[i];
    }
    ishape.h = input_shape[axis];
    index_ishape.h = index_shape[axis];
    for (int i = axis - 1; i >= 0; --i)
    {
        ishape.c *= input_shape[i];
        index_ishape.c *= index_shape[i];
    }
    scalar_t const_filler = {.s32 = 0};
    dim4 oshape = {ishape.n, ishape.c, index_num, ishape.w};
    dim4 input_global_stride, output_global_stride;
    tpu_continuous_stride(&input_global_stride, &ishape);
    tpu_continuous_stride(&output_global_stride, &oshape);
    dim4 index_stride = {
        .n = 0,
        .c = index_ishape.h * index_ishape.w,
        .h = 1,
        .w = 1};

    int cidx = 0, widx = 0;
    int type_size = tpu_data_type_size(dtype);
    while (cidx < ishape.c)
    {
        int cslice = MIN(ishape.c - cidx, 65535);
        int wslice = MIN(ishape.w - widx, 65535);

        dim4 real_oshape = {ishape.n, cslice, index_num, wslice};
        int in_offset = (ishape.n * cidx * ishape.h * ishape.w + widx) * type_size;
        int out_offset = (ishape.n * cidx * index_num * ishape.w + widx) * type_size;
        tpu_gdma_h_gather_S2S(
            output_global_addr + out_offset,
            input_global_addr + in_offset,
            index_global_addr,
            false,
            const_filler,
            &real_oshape,
            ishape.h,
            &output_global_stride,
            &input_global_stride,
            &index_stride,
            dtype);

        widx += wslice;
        if (widx >= ishape.w)
        {
            widx = 0;
            cidx += cslice;
        }
    }
    // if(axis != shape_dims - 1){
    //   tpu_gdma_cpy_cw_trans_S2S(input_global_addr,input_global_addr,&ishape,NULL,NULL,dtype);
    //   tpu_gdma_cpy_cw_trans_S2S(input_global_addr,input_global_addr,&ishape,NULL,NULL,dtype);
    // }
  }

void cast_index_int64_to_int32(
  global_addr_t in_global_addr,
  global_addr_t out_global_addr,
  int dim,
  int *index_shape)
{
  local_addr_t in_local_addr = 0;
  data_type_t src_dtype = DT_INT32;
  data_type_t dst_dtype = DT_INT32;

  dim4 bottom_global_shape ;
  bottom_global_shape.n = dim>3 ? index_shape[dim-3] : 1;
  bottom_global_shape.c = dim>2 ? index_shape[dim-2] : 1;
  bottom_global_shape.h = dim>1 ? index_shape[dim-2] : 1;
  bottom_global_shape.w = dim>0 ? index_shape[dim-1] : 1;
  for (int i = 4; i < dim; ++i)
  {
    bottom_global_shape.n *= index_shape[dim-i];
  }
  
  dim4 src_global_stride;
  src_global_stride.w = 2;//2
  src_global_stride.h = bottom_global_shape.w * src_global_stride.w;//w
  src_global_stride.c = bottom_global_shape.h * src_global_stride.h;//h*w
  src_global_stride.n = bottom_global_shape.c * src_global_stride.c;//c*h*w
  
  tpu_poll();

  // tpu_gdma_general_cpy_S2L(
  //     in_local_addr,
  //     in_global_addr,
  //     &bottom_global_shape,
  //     &bottom_global_shape,
  //     NULL,
  //     &src_global_stride,
  //     src_dtype);
  tpu_gdma_cpy_S2L(
      in_local_addr,
      in_global_addr,
      &bottom_global_shape,
      NULL,
      &src_global_stride,
      src_dtype);

  dim4 top_global_stride;
  // tpu_continuous_stride(&top_global_stride, &bottom_global_shape);
  top_global_stride.w = 1;//1
  top_global_stride.h = bottom_global_shape.w * top_global_stride.w;//w
  top_global_stride.c = bottom_global_shape.h * top_global_stride.h;//h*w
  top_global_stride.n = bottom_global_shape.c * top_global_stride.c;//c*h*w

  tpu_gdma_cpy_L2S(
      out_global_addr,
      in_local_addr,
      &bottom_global_shape,
      &top_global_stride,
      NULL,
      dst_dtype);
  tpu_poll();
}
void tpu_kernel_api_gather ( const void *args )
{
  sg_api_gather_t *api = ( sg_api_gather_t * ) args;
  tpu_initialize();
  //need to make sure the statement below is true
  // TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  int input_shape[api->dim];
  int index_shape[api->dim];
  for (int i = 0; i < api->dim; ++i)
  {
    input_shape[i] = api->input_shape[i];
    index_shape[i] = api->index_shape[i];
  }
  if(api->is_index_int64){
    cast_index_int64_to_int32(
      api->index_global_addr,
      api->index_global_addr,
      api->dim,
      index_shape);
  }
  if(api->axis != api->dim-1){
    TPUKERNEL_ASSERT_INFO(false, "not support axis != dim-1 now");
  }
  nodechip_gather (
  api->input_global_addr,
  api->index_global_addr,
  api->output_global_addr,
  input_shape,
  index_shape,
  api->dim,
  api->axis,
  api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gather );

#ifdef BACKEND_SG2260
void tpu_kernel_api_gather_multi_core ( const void *args )
{
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gather_multi_core );
#endif