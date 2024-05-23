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
        .n = 1,
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
        // tpu_poll();

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

  unsigned long long ele_num = 1;
  for (int i=0; i<dim; ++i) {
    ele_num *= index_shape[i];
  }
  dim4 in_shape = {.n = 1, .c = 1, .h = 1, .w=ele_num};

  if (ele_num > 0) {
    dim4 src_stride;
    tpu_continuous_stride(&src_stride, &in_shape);
    src_stride.n *= 2;
    src_stride.c *= 2;
    src_stride.h *= 2;
    src_stride.w *= 2;
    tpu_gdma_cpy_S2L(
        in_local_addr,
        in_global_addr,
        &in_shape,
        NULL,
        &src_stride,
        src_dtype);
  }
  tpu_sync_all();

  if (ele_num > 0) {
    tpu_gdma_cpy_L2S(
        out_global_addr,
        in_local_addr,
        &in_shape,
        NULL,
        NULL,
        dst_dtype);
  }
  tpu_sync_all();
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
  sg_api_gather_t *api = ( sg_api_gather_t * ) args;
  tpu_initialize();
  long long int inner_num = 1, gather_num = 1, gathered_num = 1, outer_num = 1;
  for (int i = 0; i < api->axis; ++i) {
    inner_num *= api->input_shape[i];
  }
  gather_num = api->input_shape[api->axis];
  gathered_num = api->index_shape[api->axis];
  for (int i=api->axis+1; i<api->dim; ++i) {
    outer_num *= api->input_shape[i];
  }
  int new_input_shape[3] = {inner_num, gather_num, outer_num};
  int new_index_shape[3] = {inner_num, gathered_num, outer_num};


  // only support slice inner_num now
  int core_num = tpu_core_num();
  int core_idx = tpu_core_index();
  int slice = DIV_UP(inner_num, core_num);
  if (slice * (core_idx + 1) > inner_num) {
    slice = MAX(0, inner_num - slice * core_idx);
  }
  new_input_shape[0] = slice;
  new_index_shape[0] = slice;
  if(api->is_index_int64){
    cast_index_int64_to_int32(
      api->index_global_addr + slice * core_idx * gathered_num * outer_num * 8,
      api->index_global_addr + slice * core_idx * gathered_num * outer_num * 4,
      3,
      new_index_shape);
  }
  if (slice > 0) {
    nodechip_gather (
      api->input_global_addr + slice * core_idx * gather_num * outer_num * tpu_data_type_size(api->dtype),
      api->index_global_addr + slice * core_idx * gathered_num * outer_num * 4,
      api->output_global_addr  + slice * core_idx * gathered_num * outer_num * tpu_data_type_size(api->dtype),
      new_input_shape,
      new_index_shape,
      3,
      1,
      api->dtype );
  }

  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_gather_multi_core );
#endif