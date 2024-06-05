#include "sg_api_struct.h"
#include "tpu_kernel.h"

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

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_conv_weight_recover );
