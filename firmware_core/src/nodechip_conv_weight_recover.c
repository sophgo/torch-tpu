#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "debug.h"
/*
 * [oc, kh * kw, DIV_UP(ic, 32), 32] => [oc, ic, kh, kw]
 *
 * [oc, kh * kw, DIV_UP(ic, 32), 32] regard as [oc, kh * kw, 1, ALIGN(ic,32)]
 * [oc, kh * kw, 1, ALIGN(ic,32)]  => [oc, ALIGN(ic, 32), 1, kh * kw]
 * [oc, ALIGN(ic, 32), 1, kh * kw] => [oc, ic, 1, kh * kw]
 * [oc, ic, 1, kh * kw]            => [oc, ic, kh, kw]
*/
void nodechip_conv_weight_32ic_to_normal_simluate (
global_addr_t    input_global_addr,
global_addr_t    output_global_addr,
const dim4      *shape
) {
  const int oc = shape->n;
  const int ic = shape->c;
  const int kh = shape->h;
  const int kw = shape->w;
  const int dtype = DT_FP16;

  int icslice = ic;
  unsigned int per_oc_isize = DIV_UP ( kh * kw, NPU_NUM ) * tpu_aligned_feature_size ( 1, ALIGN(icslice, 32), dtype );
  unsigned int per_oc_osize = DIV_UP ( ALIGN(icslice, 32), NPU_NUM ) * tpu_aligned_feature_size ( kh, kw, dtype );

  unsigned int per_oc_total_size = ALIGN ( per_oc_isize, BANK_SIZE ) * 2 + ALIGN ( per_oc_osize, BANK_SIZE ) * 2;
  TPUKERNEL_ASSERT ( per_oc_total_size <= ( unsigned int ) LOCAL_MEM_SIZE );
  int ocsecs = 1;
  int ocslice = oc;
  bool first_time = true;
  while ( 1 ) {
    if ( !first_time ) {
      ocsecs++;
      ocslice = DIV_UP ( oc, ocsecs );
    }
    unsigned int some_oc_isize = ocslice * per_oc_isize;
    unsigned int some_oc_osize = ocslice * per_oc_osize;
    unsigned int some_oc_total_size = ALIGN ( some_oc_isize, BANK_SIZE ) * 2 + ALIGN ( some_oc_osize, BANK_SIZE ) * 2;
    if ( some_oc_total_size <= ( unsigned int ) LOCAL_MEM_SIZE ) break;
    first_time = false;
  }
  unsigned int isize = ocslice * per_oc_isize;
  unsigned int osize = ocslice * per_oc_osize;
  local_addr_t iaddr_ping = 0;
  local_addr_t iaddr_pong = ALIGN ( iaddr_ping + isize, BANK_SIZE );
  local_addr_t oaddr_ping = ALIGN ( iaddr_pong + isize, BANK_SIZE );
  local_addr_t oaddr_pong = ALIGN ( oaddr_ping + osize, BANK_SIZE );
  TPUKERNEL_ASSERT ( oaddr_pong + osize <= ( unsigned int ) LOCAL_MEM_SIZE );
  bool ping = true;
  bool parallel_branch = false;
  int last_ocstart = 0;
  int last_ocslice = 0;
  int ocstart = 0;
  int ocend = 0;
  for ( int oc_idx = 0; oc_idx < ocsecs; oc_idx++ ) {
    ocstart = ocend;
    ocslice = MIN ( ocslice, oc - ocend );
    ocend = ocstart + ocslice;
    //[oc, kh * kw, 1, align(ic,32)] S2L
    dim4 islice_shape = {ocslice, kh * kw, 1, ALIGN(icslice, 32)};
    tpu_gdma_cpy_S2L (
    ping ? iaddr_ping : iaddr_pong,
    input_global_addr +
    ocstart * kh * kw * ALIGN(ic, 32) * tpu_data_type_size ( dtype ),
    &islice_shape,
    NULL,
    NULL,
    dtype );
    if ( parallel_branch ) {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    parallel_branch = true;
    //[oc, kh * kw, 1, ic] cw/wc trans [oc, ic, 1, kh * kw]
    dim4 trans_dst_shape = {ocslice, icslice, 1, kh * kw};
    if ( trans_dst_shape.w >= trans_dst_shape.c ) {
      tpu_bdc_cw_trans (
      ping ? oaddr_ping : oaddr_pong,
      ping ? iaddr_ping : iaddr_pong,
      &trans_dst_shape,
      dtype );
    } else {
      tpu_bdc_wc_trans (
      ping ? oaddr_ping : oaddr_pong,
      ping ? iaddr_ping : iaddr_pong,
      &trans_dst_shape,
      dtype );
    }
    //[oc, ic, 1, kh * kw] L2S
    if ( oc_idx > 0 ) {
      dim4 last_reordered_shape = {last_ocslice,
                                   icslice,
                                   kh,
                                   kw
                                  };
      tpu_gdma_cpy_L2S (
      output_global_addr + last_ocstart * ic * kh * kw * tpu_data_type_size ( dtype ),
      ping ? oaddr_pong : oaddr_ping,
      &last_reordered_shape,
      NULL,
      NULL,
      dtype );
    }
    ping = !ping;
    last_ocstart = ocstart;
    last_ocslice = ocslice;
  }
  tpu_parallel_end();
  // move last slice L2S
  dim4 last_reordered_shape = {last_ocslice,
                               icslice,
                               kh,
                               kw
                              };
  tpu_gdma_cpy_L2S (
  output_global_addr +
  last_ocstart * ic * kh * kw * tpu_data_type_size ( dtype ),
  ping ? oaddr_pong : oaddr_ping,
  &last_reordered_shape,
  NULL,
  NULL,
  dtype );
}

int tpu_kernel_api_conv_weight_recover ( const void* args ) {
  sg_api_conv_weight_recover_t* api = ( sg_api_conv_weight_recover_t* ) args;
  dim4 shape = {api->shape[0], api->shape[1], api->shape[2], api->shape[3]};
  tpu_initialize();
  if ( api->mode == 0 ) {
    nodechip_conv_weight_32ic_to_normal_simluate (
    api->input_global_addr,
    api->output_global_addr,
    &shape );
  } else {
    TPUKERNEL_ASSERT ( 0 );
  }
  tpu_poll();
  return 0;
}

static void cpy_32ic_into_normal_in_local(ltensor_t* input, ltensor_t* output, int cur_slice){
  // 单次考虑32个ic的结果 如果不满足32ic再说
  int oc       = cur_slice;
  int ic       = output->shape.h;
  int khkw     = output->shape.w;
  int cur_idx  = 0;
  while(cur_idx * 32 < ic){
    int remain_ic      = MIN(32, ic - cur_idx * 32);
    laddr_t curi_laddr = get_ltensor_idx_addr(input,  (dim4){0, 0, cur_idx * khkw, 0});
    laddr_t curo_laddr = get_ltensor_idx_addr(output, (dim4){0, 0, cur_idx * 32, 0});
    // use bdc_cpy
    dim4 target_shape  = {1, oc, remain_ic, khkw};
    dim4 new_istride   = { input->stride.n, input->stride.c,  1, 32};
    tpu_bdc_cpy(curo_laddr, curi_laddr, &target_shape, &(output->stride), &new_istride, input->dtype);
    cur_idx ++;
  }
}

// oc, ic_32, khkw, 32
/**
[oc, ic_32, khkw, 32] ddr -> [1, oc, ic_32 * kh * kw, 32]
[1, oc, ic_32 * kh * kw, 32] - cpy -> [1, oc, ic_32 * 32, kh * kw] local
*/
void nodechip_conv_weight_32ic_to_normal_2(gaddr_t input, gaddr_t output, const dim4* shape){
  // TODO: impl with different shape with better performance (such as ic is very large)
  int oc    = shape->n;
  int ic    = shape->c;
  int kh    = shape->h;
  int kw    = shape->w;
  int dtype = DT_FP16;
  // do not consider kh * kw == 1
  int ic_32 = DIV_UP(ic, 32);
  gtensor_t input_gtensor   = init_gtensor(input,  (dim4){oc, ic_32, kh * kw, 32}, dtype);
  gtensor_t output_gtensor  = init_gtensor(output, (dim4){oc, ic, kh, kw},         dtype);

  global_reshape(&input_gtensor,  (dim4){1, oc, ic_32 * kh * kw, 32});
  global_reshape(&output_gtensor, (dim4){1, oc, ic, kh * kw });
  show_tensor(input_gtensor);
  show_tensor(output_gtensor);
  int oc_slice    = NPU_NUM;

  ltensor_t input_ltensor_1  = init_ltensor((dim4){1, oc_slice, ic_32 * kh * kw, 32}, dtype);
  ltensor_t output_ltensor_1 = init_ltensor((dim4){1, oc_slice, ic, kh * kw},         dtype);
  ltensor_t input_ltensor_2  = init_ltensor((dim4){1, oc_slice, ic_32 * kh * kw, 32}, dtype);
  ltensor_t output_ltensor_2 = init_ltensor((dim4){1, oc_slice, ic, kh * kw},         dtype);

  malloc_after(output_ltensor_1, input_ltensor_1, BANK_SIZE);
  malloc_after(input_ltensor_2, output_ltensor_1, BANK_SIZE);
  malloc_after(output_ltensor_2, input_ltensor_2, BANK_SIZE);

  ltensor_t* load_tensors[2]  = {&input_ltensor_1, &input_ltensor_2};
  ltensor_t* store_tensors[2] = {&output_ltensor_1, &output_ltensor_2};

  int cur_idx[3]    = {0, 0, 0};
  int cur_slice[3]  = {0, 0, 0};
  int draning_idx   = 0;
  int stage_idx     = 0;

  while(cur_idx[2] < oc){
    tpu_parallel_start();

    if(draning_idx < 1){
      cur_slice[0] = MIN(oc_slice - cur_idx[0], oc_slice);
    }

    if(stage_idx > 1){
      // do store
      ltensor_t* curo_ltensor  = store_tensors[stage_idx & 0x1];
      gaddr_t cur_output_gaddr = get_gtensor_idx_addr(&output_gtensor, (dim4){0, cur_idx[2], 0, 0});
      dim4 cur_store_shape     = {1, cur_slice[2], ic, kh * kw};
      tpu_gdma_cpy_L2S(cur_output_gaddr, curo_ltensor->addr, &cur_store_shape, &(output_gtensor.stride), &(curo_ltensor->stride), dtype);
    }

    if(stage_idx > 0 && draning_idx < 2){
      // do compute
      ltensor_t* curi_ltensor = load_tensors[(stage_idx + 1) & 0x1];
      ltensor_t* curo_ltensor = store_tensors[(stage_idx + 1) & 0x1];
      cpy_32ic_into_normal_in_local(curi_ltensor, curo_ltensor, cur_slice[1]);
    }

    if(draning_idx < 1){
      // do load
      gaddr_t cur_input_gaddr = get_gtensor_idx_addr(&input_gtensor, (dim4){0, cur_idx[0], 0, 0});
      ltensor_t* cur_ltensor  = load_tensors[stage_idx & 0x1];
      dim4 cur_load_shape     = {1, cur_slice[0], ic_32 * kh * kw, 32};
      tpu_gdma_cpy_S2L( cur_ltensor->addr, cur_input_gaddr, &cur_load_shape, &(cur_ltensor->stride), &(input_gtensor.stride), dtype);
    }

    tpu_parallel_end();

    pipeline_move(cur_slice, 3);
    pipeline_move(cur_idx, 3);

    if(draning_idx < 1){
      cur_idx[0] += cur_slice[0];
      if(cur_idx[0] >= oc){
        draning_idx ++;
      }
    }else{
      draning_idx++;
    }
    stage_idx ++;
  }

}

int tpu_kernel_api_conv_weight_recover_multi_core ( const void* args ) {
  sg_api_conv_weight_recover_t* api = ( sg_api_conv_weight_recover_t* ) args;
  tpu_initialize();
  int core_idx = tpu_core_index();
  int core_num = tpu_core_num();
  int oc       = api->shape[0];
  int ic       = api->shape[1];
  int oc_slice = MAX(DIV_UP(oc, core_num), NPU_NUM);
  int ic_32    = DIV_UP(ic, 32);
  if(core_idx * oc_slice < oc) {
    dim4 shape = {MIN(oc_slice, oc - core_idx * oc_slice), ic, api->shape[2], api->shape[3]};
    global_addr_t cur_input_gaddr  = api->input_global_addr  + core_idx * oc_slice * ic_32 * shape.h * shape.w * 32 * tpu_data_type_size(DT_FP16);
    global_addr_t cur_output_gaddr = api->output_global_addr + core_idx * oc_slice * shape.h * shape.w * ic * tpu_data_type_size(DT_FP16);
    nodechip_conv_weight_32ic_to_normal_2(
      cur_input_gaddr,
      cur_output_gaddr,
      &shape
    );
  }
  tpu_poll();
  return 0;
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_conv_weight_recover_multi_core );
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_conv_weight_recover );
