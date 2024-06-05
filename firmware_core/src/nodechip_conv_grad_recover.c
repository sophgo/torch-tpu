#include "sg_api_struct.h"
#include "tpu_kernel.h"

/*
 * [ic, kh * kw, DIV_UP(oc, 32), 32] => [oc, ic, kh, kw]
 *
 * [ic, kh * kw, DIV_UP(oc, 32), 32]  => [ic, kh * kw, 1, oc]
 * [ic, kh * kw, 1, oc] => [ic, oc, 1, kh * kw]
 * [ic, oc, 1, kh * kw] => [oc, ic, kh, kw]
*/
void nodechip_conv_grad_32oc_to_normal_simluate (
global_addr_t    input_global_addr,
global_addr_t    output_global_addr,
const dim4      *shape
){  
    const int oc = shape->n;
    const int ic = shape->c;
    const int kh = shape->h;
    const int kw = shape->w;

    const int dtype = DT_FP16;

    int icslice = ic;
    unsigned int per_ic_isize = icslice * DIV_UP(kh * kw, NPU_NUM) * tpu_aligned_feature_size(DIV_UP(oc, 32), 32, dtype);
    unsigned int per_ic_osize = DIV_UP(oc, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);

    unsigned int per_ic_total_size = ALIGN(per_ic_isize, BANK_SIZE) * 2 + ALIGN(per_ic_osize, BANK_SIZE) * 2;
    TPUKERNEL_ASSERT(per_ic_total_size <= (unsigned int)LOCAL_MEM_SIZE);

    int icsecs = 1;
    bool first_time = true;
    while(1) {
        if (!first_time) {
            icsecs++;
            icslice = DIV_UP(ic, icsecs);
        }
        unsigned int some_ic_isize = icslice * per_ic_isize;
        unsigned int some_ic_osize = icslice * per_ic_osize;
        unsigned int some_ic_total_size = ALIGN(some_ic_isize, BANK_SIZE) * 2 + ALIGN(some_ic_osize, BANK_SIZE) * 2;
        if (some_ic_total_size <= (unsigned int)LOCAL_MEM_SIZE) break;
        first_time = false;
    }

    unsigned int isize = icslice * per_ic_isize;
    unsigned int osize = icslice * per_ic_osize;

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
    local_addr_t oaddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);

    dim4 istride = {kh * kw * DIV_UP(oc, 32) * 32, DIV_UP(oc, 32) * 32, DIV_UP(oc, 32) * 32, 1};

    bool ping = true;
    bool parallel_branch = false;

    int last_icstart = 0;
    int last_icslice = 0;

    int icstart = 0;
    int icend = 0;
    for (int ic_idx = 0; ic_idx < icsecs; ic_idx++) {
        icstart = icend;
        icslice = MIN(icslice, ic - icend);
        icend = icstart + icslice;

        dim4 islice_shape = {icslice, kh * kw, 1, DIV_UP(oc, 32) * 32};
        tpu_gdma_cpy_S2L(
            ping ? iaddr_ping : iaddr_pong,
            input_global_addr +
                icstart * kh * kw *  DIV_UP(oc, 32) * 32 * tpu_data_type_size(dtype),
            &islice_shape,
            NULL,
            &istride,
            dtype);

        if (parallel_branch) {
            tpu_parallel_end();
        }
        tpu_parallel_start();
        parallel_branch = true;

        // [icslice, kh * kw, 1, oc] => [icslice, oc, 1, kh * kw]
        dim4 trans_dst_shape = {icslice, oc, 1, kh * kw};
        if (trans_dst_shape.w >= trans_dst_shape.c) {
            tpu_bdc_cw_trans(
                ping ? oaddr_ping : oaddr_pong,
                ping ? iaddr_ping : iaddr_pong,
                &trans_dst_shape,
                dtype);
        } else {
            tpu_bdc_wc_trans(
                ping ? oaddr_ping : oaddr_pong,
                ping ? iaddr_ping : iaddr_pong,
                &trans_dst_shape,
                dtype);
        }

        //[icslice, oc, 1, kh * kw,] L2S_NC_Trans
        if (ic_idx > 0) {
            dim4 last_reordered_shape = { oc, last_icslice, kh, kw };
            dim4 reordered_stride = {ic * kh * kw, kh * kw, kw, 1};
            tpu_gdma_cpy_nc_trans_L2S(
                output_global_addr +
                    last_icstart * kh * kw * tpu_data_type_size(dtype),
                ping ? oaddr_pong : oaddr_ping,
                &last_reordered_shape,
                &reordered_stride,
                NULL,
                dtype);
        }
        ping = !ping;
        last_icstart = icstart;
        last_icslice = icslice;
    }
    tpu_parallel_end();
    dim4 last_reordered_shape = { oc, last_icslice, kh, kw };
    dim4 reordered_stride = {ic * kh * kw, kh * kw, kw, 1};
    tpu_gdma_cpy_nc_trans_L2S(
        output_global_addr +
            last_icstart * kh * kw * tpu_data_type_size(dtype),
        ping ? oaddr_pong : oaddr_ping,
        &last_reordered_shape,
        &reordered_stride,
        NULL,
        dtype);
}
int tpu_kernel_api_conv_grad_recover ( const void* args ) {
  sg_api_conv_grad_recover_t* api = ( sg_api_conv_grad_recover_t* ) args;
  dim4 shape = {api->shape[0], api->shape[1], api->shape[2], api->shape[3]};
  tpu_initialize();
  nodechip_conv_grad_32oc_to_normal_simluate (
    api->input_global_addr,
    api->output_global_addr,
    &shape );
  tpu_poll();
  return 0;
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_conv_grad_recover );