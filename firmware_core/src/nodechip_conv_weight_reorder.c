#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"

/*
 * [oc, ic, kh, kw] => [1, oc, DIV_UP(ic, 32) * kh * kw, 32]
 *
 * [oc, ic, kh, kw] regard as [oc, ic, 1, kh, kw]
 * [oc, ic, 1, kh * kw] => [oc, kh * kw, 1, ic]
 * [oc, kh * kw, 1, ic] => [oc, kh * kw, DIV_UP(ic, 32), 32]
 *
*/
void nodechip_conv_weight_to_32ic(
    global_addr_t    input_global_addr,
    global_addr_t    output_global_addr,
    const dim4      *shape
) {

    const int oc = shape->n;
    const int ic = shape->c;
    const int kh = shape->h;
    const int kw = shape->w;

    const int dtype = DT_FP16;

    //(TODO) support split ic
    int icslice = ic;
    unsigned int per_oc_isize = DIV_UP(icslice, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int per_oc_osize = DIV_UP(kh * kw, NPU_NUM) * tpu_aligned_feature_size(1, icslice, dtype);

    unsigned int per_oc_total_size = ALIGN(per_oc_isize, BANK_SIZE) * 2 + ALIGN(per_oc_osize, BANK_SIZE) * 2;
    TPUKERNEL_ASSERT(per_oc_total_size <= (unsigned int)LOCAL_MEM_SIZE);

    int ocsecs = 1;
    int ocslice = oc;
    bool first_time = true;
    while(1) {
        if (!first_time) {
            ocsecs++;
            ocslice = DIV_UP(oc, ocsecs);
        }
        unsigned int some_oc_isize = ocslice * per_oc_isize;
        unsigned int some_oc_osize = ocslice * per_oc_osize;
        unsigned int some_oc_total_size = ALIGN(some_oc_isize, BANK_SIZE) * 2 + ALIGN(some_oc_osize, BANK_SIZE) * 2;
        if (some_oc_total_size <= (unsigned int)LOCAL_MEM_SIZE) break;
        first_time = false;
    }

    unsigned int isize = ocslice * per_oc_isize;
    unsigned int osize = ocslice * per_oc_osize;

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
    local_addr_t oaddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);

    bool ping = true;
    bool parallel_branch = false;

    int last_ocstart = 0;
    int last_ocslice = 0;

    int ocstart = 0;
    int ocend = 0;
    for (int oc_idx = 0; oc_idx < ocsecs; oc_idx++) {
        ocstart = ocend;
        ocslice = MIN(ocslice, oc - ocend);
        ocend = ocstart + ocslice;

        //[oc, ic, 1, kh * kw] S2L
        dim4 islice_shape = {ocslice, icslice, 1, kh * kw};
        tpu_gdma_cpy_S2L(
            ping ? iaddr_ping : iaddr_pong,
            input_global_addr +
                ocstart * ic * kh * kw * tpu_data_type_size(dtype),
            &islice_shape,
            NULL,
            NULL,
            dtype);

        if (parallel_branch) {
            tpu_parallel_end();
        }
        tpu_parallel_start();
        parallel_branch = true;

        scalar_t zero = {.f32 = 0.0f};
        dim4 zero_shape = {ocslice, kh * kw, 1, ALIGN(icslice, 32)};
        tpu_bdc_set_C(
            ping ? oaddr_ping : oaddr_pong,
            tpu_cast(zero, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
            &zero_shape,
            NULL,
            dtype);

        //[oc, ic, 1, kh * kw] cw/wc trans [oc, kh * kw, 1, ic]
        dim4 trans_dst_shape = {ocslice, kh * kw, 1, icslice};
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

        //[oc, kh * kw, 1, ic] L2S
        if (oc_idx > 0) {
            dim4 last_reordered_shape = {last_ocslice,
                                         kh * kw,
                                         DIV_UP(icslice, 32),
                                         32};
            dim4 reordered_stride = {DIV_UP(ic, 32) * kh * kw * 32, 32, kh * kw * 32, 1};
            tpu_gdma_cpy_L2S(
                output_global_addr +
                    last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 * tpu_data_type_size(dtype),
                ping ? oaddr_pong : oaddr_ping,
                &last_reordered_shape,
                &reordered_stride,
                NULL,
                dtype);
        }
        ping = !ping;
        last_ocstart = ocstart;
        last_ocslice = ocslice;
    }
    tpu_parallel_end();
    // move last slice L2S
    dim4 last_reordered_shape = {last_ocslice,
                                 kh * kw,
                                 DIV_UP(icslice, 32),
                                 32};

    dim4 reordered_stride = {DIV_UP(ic, 32) * kh * kw * 32, 32, kh * kw * 32, 1};
    tpu_gdma_cpy_L2S(
        output_global_addr +
            last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 * tpu_data_type_size(dtype),
        ping ? oaddr_pong : oaddr_ping,
        &last_reordered_shape,
        &reordered_stride,
        NULL,
        dtype);
}

/*
 * [oc, ic, kh, kw] => [1, ic, DIV_UP(oc, 32) * kh * kw, 32]
 *
 * [oc, ic, kh, kw] => [ic, oc, 1, kh * kw]
 * [ic, oc, 1, kh * kw] => [ic, kh * kw, 1, oc]
 * [ic, kh * kw, 1, oc] => [ic, kh * kw, DIV_UP(oc, 32), 32]
 *
*/
void nodechip_conv_weight_to_32oc(
    global_addr_t    input_global_addr,
    global_addr_t    output_global_addr,
    const dim4      *shape
 ) {

    const int oc = shape->n;
    const int ic = shape->c;
    const int kh = shape->h;
    const int kw = shape->w;

    const int dtype = DT_FP16;

    //(TODO) support split oc
    int ocslice = oc;
    unsigned int per_ic_isize = DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int per_ic_osize = DIV_UP(kh * kw, NPU_NUM) * tpu_aligned_feature_size(1, ocslice, dtype);

    unsigned int per_ic_total_size = ALIGN(per_ic_isize, BANK_SIZE) * 2 + ALIGN(per_ic_osize, BANK_SIZE) * 2;
    TPUKERNEL_ASSERT(per_ic_total_size <= (unsigned int)LOCAL_MEM_SIZE);

    int icsecs = 1;
    int icslice = ic;
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

    dim4 istride = {ic * kh * kw, kh * kw, kw, 1};

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

        //[oc, icslice, kh, kw] => [icslice, oc, kh, kw]
        dim4 islice_shape = {icslice, ocslice, kh, kw};
        tpu_gdma_cpy_nc_trans_S2L(
            ping ? iaddr_ping : iaddr_pong,
            input_global_addr +
                icstart * kh * kw * tpu_data_type_size(dtype),
            &islice_shape,
            NULL,
            &istride,
            dtype);

        if (parallel_branch) {
            tpu_parallel_end();
        }
        tpu_parallel_start();
        parallel_branch = true;

        scalar_t zero = {.f32 = 0.0f};
        dim4 zero_shape = {icslice, kh * kw, 1, ALIGN(ocslice, 32)};
        tpu_bdc_set_C(
            ping ? oaddr_ping : oaddr_pong,
            tpu_cast(zero, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
            &zero_shape,
            NULL,
            dtype);

        //[ic, oc, 1, kh * kw] => [ic, kh * kw, 1, oc]
        dim4 trans_dst_shape = {icslice, kh * kw, 1, ocslice};
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

        //[ic, kh * kw, 1, oc] L2S
        if (ic_idx > 0) {
            dim4 last_reordered_shape = {last_icslice,
                                         kh * kw,
                                         DIV_UP(ocslice, 32),
                                         32};
            dim4 reordered_stride = {DIV_UP(oc, 32) * kh * kw, 32, kh * kw * 32, 1};
            tpu_gdma_cpy_L2S(
                output_global_addr +
                    last_icstart * DIV_UP(oc, 32) * kh * kw * 32 * tpu_data_type_size(dtype),
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
    dim4 last_reordered_shape = {last_icslice,
                                 kh * kw,
                                 DIV_UP(ocslice, 32),
                                 32};
    dim4 reordered_stride = {DIV_UP(oc, 32) * kh * kw, 32, kh * kw * 32, 1};
    tpu_gdma_cpy_L2S(
        output_global_addr +
            last_icstart * DIV_UP(oc, 32) * kh * kw * 32 * tpu_data_type_size(dtype),
        ping ? oaddr_pong : oaddr_ping,
        &last_reordered_shape,
        &reordered_stride,
        NULL,
        dtype);
}

void tpu_kernel_api_conv_weight_reorder(const void* args) {
    sg_api_conv_weight_reorder_t* api = (sg_api_conv_weight_reorder_t*)args;

    dim4 shape = {api->shape[0], api->shape[1], api->shape[2], api->shape[3]};

    tpu_initialize();
    if (api->reorder_mode == 0) {
        nodechip_conv_weight_to_32ic(
            api->input_global_addr,
            api->output_global_addr,
            &shape);
    } else if (api->reorder_mode == 1) {
        nodechip_conv_weight_to_32oc(
            api->input_global_addr,
            api->output_global_addr,
            &shape);
    } else {
        TPUKERNEL_ASSERT(0);
    }
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_conv_weight_reorder);
