#include "config.h"
#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"
#ifdef USING_CMODEL
#include "cmodel_memory.h"
#endif

static inline bool is_local_mem_enough(
        int *size,
        int ic,
        int oc,
        int ihw,
        int ohw,
        int weight_bias_size,
        data_type_t idtype,
        data_type_t odtype) {
    int ic_per_npu = DIV_UP(ic, NPU_NUM);
    int oc_per_npu = DIV_UP(oc, NPU_NUM);
    size[0] = ic_per_npu * tpu_aligned_feature_size(1, ihw, idtype);
    size[1] = oc_per_npu * tpu_aligned_feature_size(1, ohw, odtype);
    int total_size = ALIGN(weight_bias_size, BANK_SIZE)  +
                     ALIGN(size[0], BANK_SIZE) * 2 +
                     ALIGN(size[1], BANK_SIZE) * 2;
    return total_size <= LOCAL_MEM_SIZE;
}

static inline bool split_oh_or_ow(
        int *slice,
        int *secs,
        int stride,
        int ic,
        int oc,
        int ih_or_iw,
        int oh_or_ow,
        int kh_or_kw_ext,
        int weight_bias_size,
        data_type_t idtype,
        data_type_t odtype,
        bool transposed) {
    bool valid;
    int size[2];
    int slice_new = *slice;
    do {
        if (slice_new == 1) {
            *slice = slice_new;
            --(*secs);
            return false;
        }
        slice_new = DIV_UP(*slice, *secs);
        // for conv:
        //      o = (i - k_ext) / stride + 1
        // for transposed conv:
        //      i_ext = (i - 1) * stride + 1
        //      o = i_ext - k_ext + 1
        int islice = transposed ?
            DIV_UP(slice_new - 1 + kh_or_kw_ext - 1, stride) + 1 :
            (slice_new - 1) * stride + kh_or_kw_ext;
        valid = is_local_mem_enough(
                size,
                ic,
                oc,
                islice * ih_or_iw,
                slice_new * oh_or_ow,
                weight_bias_size,
                idtype,
                odtype);
        ++(*secs);
    } while(!valid);
    *slice = slice_new;
    --(*secs);
    return true;
}

typedef struct {
    int nsecs;
    int ohsecs;
    int owsecs;
    int ocsecs;
} conv_secs_info_t;

static void split_conv_float(
        const dim4    *ishape,
        const dim4    *oshape,
        int           groups,
        const dim2    *kernel,
        const dim2    *dilation,
        const dim2    *stride,
        data_type_t   idtype,
        data_type_t   odtype,
        bool          grad_bias_enable,
        conv_secs_info_t  *conv_secs
  ) {
    int kh_ext = dilation->h * (kernel->h - 1) + 1;
    int kw_ext = dilation->w * (kernel->w - 1) + 1;
    int ic = ishape->c / groups;
    int oc = oshape->c / groups;

    conv_secs->nsecs = ishape->n;
    conv_secs->ocsecs = 1;
    conv_secs->ohsecs = 1;
    conv_secs->owsecs = 1;
    int ohslice = oshape->h;
    int owslice = oshape->w;

    // split oc
    int w_coeff = (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) *
        kernel->h * kernel->w * tpu_data_type_size(idtype);
    int okslice = BANK_SIZE * (LOCAL_MEM_BANKS - 4) / w_coeff;

    TPUKERNEL_ASSERT(okslice > 0);
    int ocslice = okslice * NPU_NUM;
    conv_secs->ocsecs = MIN(DIV_UP(oc, ocslice), __UINT16_MAX__);
    ocslice = DIV_UP(oc, conv_secs->ocsecs);
    int wsize = DIV_UP(ocslice, NPU_NUM) * w_coeff;

    int size[2];   // {input, output}
    bool valid = is_local_mem_enough(
            size,
            ic,
            ocslice,
            ishape->h * ishape->w,
            ohslice * owslice,
            wsize,
            idtype,
            odtype);
    if (valid) return;
    // split oh and ow
    // split_w_first may lead to small wslice and low efficiency of GDMA
    bool split_h_first = true;
    int hwsecs = 1 + (size[0] + size[1]) / (LOCAL_MEM_SIZE - ALIGN(wsize, BANK_SIZE));
    if (split_h_first) {
        conv_secs->ohsecs = hwsecs;
        valid = split_oh_or_ow(
                &ohslice,
                &(conv_secs->ohsecs),
                stride->h,
                ic,
                ocslice,
                ishape->w,
                owslice,
                kh_ext,
                wsize,
                idtype,
                odtype,
                false);
        if (!valid) {
            int ihslice = (ohslice - 1) * stride->h + kh_ext;
            valid = split_oh_or_ow(
                    &owslice,
                    &(conv_secs->owsecs),
                    stride->w,
                    ic,
                    ocslice,
                    ihslice,
                    ohslice,
                    kw_ext,
                    wsize,
                    idtype,
                    odtype,
                    false);
        }
    } else {
        conv_secs->owsecs = hwsecs;
        valid = split_oh_or_ow(
                &owslice,
                &(conv_secs->owsecs),
                stride->w,
                ic,
                ocslice,
                ishape->h,
                ohslice,
                kw_ext,
                wsize,
                idtype,
                odtype,
                false);
        if (!valid) {
            int iwslice = (owslice - 1) * stride->w + kw_ext;
            valid = split_oh_or_ow(
                    &ohslice,
                    &(conv_secs->ohsecs),
                    stride->h,
                    ic,
                    ocslice,
                    iwslice,
                    owslice,
                    kh_ext,
                    wsize,
                    idtype,
                    odtype,
                    false);
        }
    }
    TPUKERNEL_ASSERT(valid);
}

typedef struct {
    int nsecs;
    int ocsecs;
    int khsecs;
    int kwsecs;
} grad_weight_secs_info_t;

static void grad_weight_split(
        const dim4    *ishape,
        const dim4    *oshape,
        int           groups,
        const dim2    *kernel,
        const dim2    *dilation,
        const dim2    *stride,
        data_type_t   idtype,
        data_type_t   odtype,
        bool          grad_bias_enable,
        grad_weight_secs_info_t  *secs_info
  ) {

    int n = ishape->n;
    int ic = ishape->c / groups;
    int oc = oshape->c / groups;
    int ih = ishape->h;
    int iw = ishape->w;
    int oh = oshape->h;
    int ow = oshape->w;
    int kh = kernel->h;
    int kw = kernel->w;
    //int kh_ext = dilation->h * (kernel->h - 1) + 1;
    //int kw_ext = dilation->w * (kernel->w - 1) + 1;

    secs_info->nsecs = 1;
    secs_info->ocsecs = 1;
    secs_info->khsecs = 1;
    secs_info->kwsecs = 1;

    int nslice = n;
    int ocslice = oc;
    int khslice = kh;
    int kwslice = kw;

    int isize_pern = DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(ih, iw, idtype);
    // split n simply
    // isize can use 4 banks(if not grad_bias_enable, will be 5)
    nslice = (grad_bias_enable ? 4 : 5) * LOCAL_BANK_SIZE / isize_pern;
    TPUKERNEL_ASSERT(nslice >= 1);
    secs_info->nsecs = DIV_UP(n, nslice);
    unsigned int isize = nslice * DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(ih, iw, idtype);

    // split kh & kw
    unsigned int i_bank_size = ALIGN(isize, BANK_SIZE) / LOCAL_BANK_SIZE * 2;
    unsigned int w_bank_size = LOCAL_MEM_BANKS - i_bank_size - (2 + (grad_bias_enable ? 2 : 0) + 1);
    unsigned int w_coeff = (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) *
                     kernel->h * kernel->w * tpu_data_type_size(idtype);
    int khwslice = w_bank_size / 2 * LOCAL_BANK_SIZE / w_coeff;
    if (khwslice == 0) {
        int w_coeff_perw = (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) * 1 * kernel->w * tpu_data_type_size(idtype);
        khwslice = w_bank_size / 2 * LOCAL_BANK_SIZE / w_coeff_perw;
        if (khwslice >= 1) {
            khslice = khwslice;
            secs_info->khsecs = DIV_UP(kh, khslice);
        } else {
            int w_coeff_min = (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) * 1 * 1 * tpu_data_type_size(idtype);
            khwslice = w_bank_size / 2 * LOCAL_BANK_SIZE / w_coeff_min;
            if (khwslice == 0) TPUKERNEL_ASSERT(0);
            secs_info->kwsecs = DIV_UP(kw, khwslice);
        }
    }

    //split oc
    int max_size_per_64oc = MAX(nslice * tpu_aligned_feature_size(oh, ow, odtype), tpu_aligned_feature_size(1, 32, idtype));
    ocslice = LOCAL_BANK_SIZE / max_size_per_64oc;
    TPUKERNEL_ASSERT(ocslice > 0);
    ocslice = MIN(oc, ocslice * NPU_NUM);
    secs_info->ocsecs = DIV_UP(oc, ocslice);

    unsigned int osize = nslice * DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(oh, ow, odtype);
    unsigned int grad_bias_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 1, idtype): 0;
    unsigned int buffer_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 32, idtype) : 0;
    buffer_size = MAX(buffer_size, osize);

    int wsize = DIV_UP(ocslice, NPU_NUM) * (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) * khslice * kwslice * tpu_data_type_size(idtype);

    unsigned int total_size = ALIGN(isize, BANK_SIZE) * 2 +
                              ALIGN(osize, BANK_SIZE) * 2 +
                              ALIGN(wsize, BANK_SIZE) * 2 +
                              ALIGN(grad_bias_size, BANK_SIZE) * 2 +
                              ALIGN(buffer_size, BANK_SIZE);

    TPUKERNEL_ASSERT(total_size <= (unsigned int)LOCAL_MEM_SIZE);
}

static void split_deconv_float(
        const dim4        *ishape,
        const dim4        *oshape,
        int                groups,
        const dim2        *kernel,
        const dim2        *dilation,
        const dim2        *stride,
        data_type_t        idtype,
        data_type_t        odtype,
        conv_secs_info_t  *p_secs
) {
    int kh_ext = dilation->h * (kernel->h - 1) + 1;
    int kw_ext = dilation->w * (kernel->w - 1) + 1;
    int ic = ishape->c / groups;
    int oc = oshape->c / groups;

    p_secs->nsecs = ishape->n;
    p_secs->ocsecs = 1;
    p_secs->ohsecs = 1;
    p_secs->owsecs = 1;
    int ohslice = oshape->h;
    int owslice = oshape->w;

    // split oc
    const int w_coeff = (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) *
         kernel->h * kernel->w * tpu_data_type_size(idtype);
    int okslice = BANK_SIZE * (LOCAL_MEM_BANKS - 4) / w_coeff;
    TPUKERNEL_ASSERT(okslice > 0);
    int ocslice = okslice * NPU_NUM;
    p_secs->ocsecs = MIN(DIV_UP(oc, ocslice), __UINT16_MAX__);
    ocslice = DIV_UP(oc, p_secs->ocsecs);
    int wsize = DIV_UP(ocslice, NPU_NUM) * w_coeff;

    int size[2];   // {input, output}
    bool valid = is_local_mem_enough(
            size,
            ic,
            ocslice,
            ishape->h * ishape->w,
            ohslice * owslice,
            wsize,
            idtype,
            odtype);
    if (valid) return;
    // split oh and ow
    // split_w_first may lead to small wslice and low efficiency of GDMA
    bool split_h_first = true;
    int hwsecs = 1 + (size[0] + size[1]) / (LOCAL_MEM_SIZE - ALIGN(wsize, BANK_SIZE));
    if (split_h_first) {
        p_secs->ohsecs = hwsecs;
        valid = split_oh_or_ow(
                &ohslice,
                &(p_secs->ohsecs),
                stride->h,
                ic,
                ocslice,
                ishape->w,
                owslice,
                kh_ext,
                wsize,
                idtype,
                odtype,
                true);
        if (!valid) {
            // i_ext = (i - 1) * stride + 1
            // o = i_ext - k_ext + 1
            int ihslice = DIV_UP(ohslice - 1 + kh_ext - 1, stride->h) + 1;
            valid = split_oh_or_ow(
                    &owslice,
                    &(p_secs->owsecs),
                    stride->w,
                    ic,
                    ocslice,
                    ihslice,
                    ohslice,
                    kw_ext,
                    wsize,
                    idtype,
                    odtype,
                    true);
        }
    } else {
        p_secs->owsecs = hwsecs;
        valid = split_oh_or_ow(
                &owslice,
                &(p_secs->owsecs),
                stride->w,
                ic,
                ocslice,
                ishape->h,
                ohslice,
                kw_ext,
                wsize,
                idtype,
                odtype,
                true);
        if (!valid) {
            int iwslice = DIV_UP(owslice - 1 + kw_ext - 1, stride->w) + 1;
            valid = split_oh_or_ow(
                    &ohslice,
                    &(p_secs->ohsecs),
                    stride->h,
                    ic,
                    ocslice,
                    iwslice,
                    owslice,
                    kh_ext,
                    wsize,
                    idtype,
                    odtype,
                    true);
        }
    }
    TPUKERNEL_ASSERT(valid);
}

// TODO (zhaoyang.wang) consider if kernel is arranged as 3IC
// [1, oc, DIV_UP(ic, 32) * kh * kw, 32] => [1, ic, DIV_UP(oc, 32) * kh * kw, 32]
/*
  [1, oc, DIV_UP(ic, 32) * kh * kw, 32]
  [kh * kw, oc, DIV_UP(ic, 32), 32] => [kh * kw, oc, 1, ic]
  [kh * kw, ic, 1, oc] => [kh * kw, ic, DIV_UP(oc, 32), 32]
  [1, ic, DIV_UP(ic, 32) * kh * kw, 32]
*/
void nodechip_weight_reorder(
    global_addr_t   weight_global_addr,
    global_addr_t   weight_reordered_global_addr,
    int             oc,
    int             ic,
    int             kh,
    int             kw,
    data_type_t     dtype
 ) {

    int ocsecs = 1;
    int icsecs = 1;
    int ocslice = oc;
    int icslice = ic;
    // ocslice icslice min 64
    // unsigned int kernel_hw_size = kh * kw;
    unsigned int kernel_min_size = MAX(DIV_UP(64, 32) * kh * kw * 32,
                                       DIV_UP(64, 32) * kh * kw * 32);//32 * kh * kw;
    TPUKERNEL_ASSERT(kernel_min_size <= (unsigned int)(LOCAL_MEM_SIZE / 4));
    icslice = DIV_UP((unsigned int)(LOCAL_MEM_SIZE / 4), kernel_min_size);
    icsecs = DIV_UP(ic, icslice);
    ocslice = DIV_UP((unsigned int)(LOCAL_MEM_SIZE / 4),
                        DIV_UP(oc, 64) * ALIGN(DIV_UP(ic, 32) * kh * kw * 32, tpu_eu_num(dtype)));
    ocsecs = DIV_UP(oc, ocslice);

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = 4 * LOCAL_BANK_SIZE;
    local_addr_t oaddr_ping = 8 * LOCAL_BANK_SIZE;
    local_addr_t oaddr_pong = 12 * LOCAL_BANK_SIZE;

    bool ping = true;
    bool parallel_branch = false;
    // [ic, oc, kh, kw] => [oc, ic, 1, kh * kw] => [oc, 1, kh * kw, ic] => [1, oc, kh * kw, ic]

    int last_ocstart = 0;
    int last_ocslice = 0;
    int last_icstart = 0;
    int last_icslice = 0;

    int ocstart = 0;
    int ocend = 0;
    for (int ocidx = 0; ocidx < ocsecs; ocidx++) {
        ocstart = ocend;
        ocslice = MIN(ocslice, oc - ocend);
        ocend = ocstart + ocslice;

        int icstart = 0;
        int icend = 0;
        for (int icidx = 0; icidx < icsecs; icidx++) {
            icstart = icend;
            icslice = MIN(icslice, ic - icend);
            icend = icstart + icslice;

            // [1, oc, DIV_UP(ic, 32) * kh * kw, 32] => [kh * kw, oc, DIV_UP(ic, 32), 32]
            //dim4 src_shape = {1, ocslice, DIV_UP(icslice, 32) * kh * kw, 32};
            dim4 dst_shape = {kh * kw,
                              ocslice,
                              DIV_UP(icslice, 32),
                              icslice >= 32 ? 32 : last_icslice};
            dim4 src_stride = {32,
                               DIV_UP(ic, 32) * kh * kw * 32,
                               32 * kh * kw,
                               1};
            tpu_gdma_cpy_S2L(
                ping ? iaddr_ping : iaddr_pong,
                weight_global_addr +
                    ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                    last_icstart / 32 * kh * kw * 32 +
                    last_icstart - last_icstart / 32 * 32,
                &dst_shape,
                NULL,
                &src_stride,
                dtype);

            if (parallel_branch) {
                tpu_parallel_end();
            }
            tpu_parallel_start();
            parallel_branch = true;

            // [kh * kw, oc, DIV_UP(ic, 32), 32] can consider as [kh * kw, oc, 1, ic]
            dim4 cw_dst_shape = {kh * kw, ic, 1, oc};
            // [kh * kw, oc, 1, ic] => [kh * kw, ic, 1, oc]
            if (cw_dst_shape.w <= cw_dst_shape.c) {
                tpu_bdc_cw_trans(
                    ping ? oaddr_ping : oaddr_pong,
                    ping ? iaddr_ping : iaddr_pong,
                    &cw_dst_shape,
                    dtype);
            } else {
                tpu_bdc_wc_trans(
                    ping ? oaddr_ping : oaddr_pong,
                    ping ? iaddr_ping : iaddr_pong,
                    &cw_dst_shape,
                    dtype);
            }

            // [kh * kw, ic, 1, oc] can consider as [kh * kw, ic, DIV_UP(oc, 32), 32]
            // [kh * kw, ic, DIV_UP(oc, 32), 32] => [1, ic, DIV_UP(oc, 32) * kh * kw, 32]
            if (icidx > 0 || ocidx > 0) {
                dim4 reordered_shape = {kh * kw,
                                        last_icslice,
                                        DIV_UP(last_ocslice, 32),
                                        last_icslice >= 32 ? 32 : last_icslice};
                dim4 reordered_stride = {32,
                                         DIV_UP(oc, 32) * kh * kw * 32,
                                         32 * kh * kw,
                                         1};
                tpu_gdma_cpy_L2S(
                    weight_reordered_global_addr +
                        last_icstart * DIV_UP(oc, 32) * kh * kw * 32 +
                        last_ocstart / 32 * kh * kw * 32 +
                        last_ocstart - last_ocstart / 32 * 32,
                    ping ? oaddr_pong : oaddr_ping,
                    &reordered_shape,
                    &reordered_stride,
                    NULL,
                    dtype);
            }
            ping = !ping;
            last_ocstart = ocstart;
            last_ocslice = ocslice;
            last_icstart = icstart;
            last_icslice = icslice;
        }
    }
    tpu_parallel_end();
    dim4 last_reordered_shape = {kh * kw,
                                 last_icslice,
                                 DIV_UP(last_ocslice, 32),
                                 last_icslice >= 32 ? 32 : last_icslice};
    dim4 last_reordered_stride = {32,
                                  DIV_UP(oc, 32) * kh * kw * 32,
                                  32 * kh * kw,
                                  1};
    tpu_gdma_cpy_L2S(
        weight_reordered_global_addr +
            last_icstart * DIV_UP(oc, 32) * kh * kw * 32 +
            last_ocstart / 32 * kh * kw * 32 +
            last_ocstart - last_ocstart / 32 * 32,
        ping ? oaddr_pong : oaddr_ping,
        &last_reordered_shape,
        &last_reordered_stride,
        NULL,
        dtype);
}

// grad_out deconv (trans)weight => grad_input
void nodechip_conv_backward_input(
    global_addr_t    grad_out_global_addr,
    global_addr_t    grad_input_global_addr,
    global_addr_t    forward_weight_global_addr,
    // need weight_reordered_addr
    global_addr_t    weight_reordered_global_addr,
    const int        groups,
    const dim4       *grad_input_shape,
    const dim4       *grad_out_shape,
    const dim2       *kernel,
    const dim2       *stride,
    const dim2       *dilation,
    const padding_t  *pad
 ) {

    const data_type_t idtype = DT_FP16, odtype = DT_FP16;

    dim4 ishape = {.n = grad_out_shape->n,
                   .c = grad_out_shape->c,
                   .h = grad_out_shape->h,
                   .w = grad_out_shape->w};
    dim4 oshape = {.n = grad_input_shape->n,
                   .c = grad_input_shape->c,
                   .h = grad_input_shape->h,
                   .w = grad_input_shape->w};
    dim2 insert = {stride->h - 1, stride->w - 1};

    int ic = ishape.c / groups;
    int oc = oshape.c / groups;

    int kh = kernel->h;
    int kw = kernel->w;
    int kh_ext = dilation->h * (kh - 1) + 1;
    int kw_ext = dilation->w * (kw - 1) + 1;
    int ih_ext = (ishape.h - 1) * stride->h + 1;
    int iw_ext = (ishape.w - 1) * stride->w + 1;
    int pad_h0 = kh_ext - pad->top - 1;
    int pad_w0 = kw_ext - pad->left - 1;
    int output_h = grad_input_shape->h;
    int output_w = grad_input_shape->w;

    conv_secs_info_t secs_info;
    split_deconv_float(
            &ishape,
            &oshape,
            groups,
            kernel,
            dilation,
            stride,
            idtype,
            odtype,
            &secs_info);

    int nslice = DIV_UP(ishape.n, secs_info.nsecs);
    int ocslice = DIV_UP(oc, secs_info.ocsecs);
    ocslice = secs_info.ocsecs > 1 ? ALIGN(ocslice, NPU_NUM) : ocslice;
    int ohslice = DIV_UP(output_h, secs_info.ohsecs);
    int owslice = DIV_UP(output_w, secs_info.owsecs);
    int ihslice = MIN(DIV_UP(ohslice - 1 + kh_ext - 1,  stride->h) + 1, ishape.h);
    int iwslice = MIN(DIV_UP(owslice - 1 + kw_ext - 1,  stride->w) + 1, ishape.w);
    TPUKERNEL_DBG("nsecs=%d  nslice=%d\n", secs_info.nsecs, nslice);
    TPUKERNEL_DBG("ocsecs=%d  ocslice=%d\n", secs_info.ocsecs, ocslice);
    TPUKERNEL_DBG("ohsecs=%d  ohslice=%d\n", secs_info.ohsecs, ohslice);
    TPUKERNEL_DBG("owsecs=%d  owslice=%d\n", secs_info.owsecs, owslice);
    // get weight/bias/input/output size of per NPU
    unsigned int wsize = DIV_UP(ocslice, NPU_NUM) *
        (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) *
        kernel->h * kernel->w * tpu_data_type_size(idtype);
    unsigned int isize = nslice * DIV_UP(ic, NPU_NUM) *
        tpu_aligned_feature_size(ihslice, iwslice, idtype);
    unsigned int osize = nslice * DIV_UP(ocslice, NPU_NUM) *
        tpu_aligned_feature_size(ohslice, owslice, odtype);
    // assign weight/bias/input/output local memory address
    local_addr_t waddr = 0;
    local_addr_t iaddr_ping = ALIGN(waddr + wsize, BANK_SIZE);
    local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
    local_addr_t oaddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);
    TPUKERNEL_DBG("weight   local addr = 0x%5x, bank id = %d\n", waddr, waddr / BANK_SIZE);
    TPUKERNEL_DBG("in  ping local addr = 0x%5x, bank id = %d\n", iaddr_ping, iaddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("in  pong local addr = 0x%5x, bank id = %d\n", iaddr_pong, iaddr_pong / BANK_SIZE);
    TPUKERNEL_DBG("out ping local addr = 0x%5x, bank id = %d\n", oaddr_ping, oaddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("out pong local addr = 0x%5x, bank id = %d\n", oaddr_pong, oaddr_pong / BANK_SIZE);

    nodechip_weight_reorder(forward_weight_global_addr,
                            weight_reordered_global_addr,
                            oc,
                            ic,
                            kh,
                            kw,
                            idtype);
/*
    // weight reorder
    u32 weight_size = oc * ic * kh * kw * sizeof(short);
    u32 weight_reordered_size = ALIGN(ic, 32) * kh * kw * oc * sizeof(short);
    u8 weight_data[weight_size], weight_reordered_data[weight_reordered_size];
    memset((void*)weight_reordered_data, 0, weight_reordered_size);
    tpu_invalidate_cache(forward_weight_global_addr, ALIGN(weight_size, 64));
    memcpy(weight_data, GET_GLOBAL_ADDR(forward_weight_global_addr), weight_size);
    for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
      for (int ic_idx = 0; ic_idx < ceiling_func(ic, 32); ic_idx++) {
        for (int k_idx = 0; k_idx < kh * kw;  k_idx++) {
          for (int inner = 0; inner < 32; inner++) {
            if (ic_idx * 32 + inner >= ic) break;
//            int src_idx = oc_idx * ic * kh * kw +
//                          (ic_idx * 32 + inner) * kh * kw +
//                          k_idx;
            int src_idx = (ic_idx * 32 + inner) * oc * kh * kw +
                          oc_idx * kh * kw +
                          k_idx;
            int dst_idx = oc_idx * ceiling_func(ic, 32) * kh * kw * 32 +
                          ic_idx * kh * kw * 32 +
                          k_idx * 32 +
                          inner;
            ((u16*)weight_reordered_data)[dst_idx] = ((u16*)weight_data)[src_idx];
          }
        }
      }
    }
    memcpy(GET_GLOBAL_ADDR(weight_reordered_global_addr), weight_reordered_data, weight_reordered_size);
    tpu_flush_cache(weight_reordered_global_addr, ALIGN(weight_reordered_size, 64));
*/
    global_addr_t input_global_addr = grad_out_global_addr;
    global_addr_t weight_global_addr = weight_reordered_global_addr;
    global_addr_t output_global_addr = grad_input_global_addr;

    bool ping = true;
    int last_nstart = 0;
    int last_nslice = 0;
    int last_ohstart = 0;
    int last_ohslice = 0;
    int last_owstart = 0;
    int last_owslice = 0;

    dim4 istride = {
        ishape.c * ishape.h * ishape.w,
        ishape.h * ishape.w,
        ishape.w,
        1};

    dim4 ostride = {
        oshape.c * oshape.h * oshape.w,
        oshape.h * oshape.w,
        oshape.w,
        1};

    // 1st loop: groups
    for (int gidx = 0; gidx < groups; gidx++) {
        int ocstart = 0;
        int ocend = 0;
        // 2nd loop: split OC dim
        for (int ocidx = 0; ocidx < secs_info.ocsecs; ocidx++) {
            // oc boundary: [ocstart, ocend)
            ocstart = ocend;
            ocslice = MIN(ocslice, oc - ocend);
            ocend = ocstart + ocslice;

            // move weight to local memory
            dim4 wshape = {
                1,
                ocslice,
                (idtype == DT_FP32 ? ic : DIV_UP(ic, 32)) * kernel->h * kernel->w,
                idtype == DT_FP32 ? 1 : 32};
            dim4 wstride;
            tpu_compact_stride(&wstride, 0, &wshape);
            tpu_gdma_cpy_S2L(
                    waddr,
                    weight_global_addr + (gidx * oc + ocstart) * wstride.c * tpu_data_type_size(idtype),
                    &wshape,
                    &wstride,
                    NULL,
                    idtype);

            bool parallel_branch = false;
            int nstart = 0;
            int nend = 0;
            // 3th loop: split N dim
            for (int nidx = 0; nidx < secs_info.nsecs; nidx++) {
                nstart = nend;
                nslice = ishape.n / secs_info.nsecs + ((ishape.n % secs_info.nsecs) > nidx);
                nend = nstart + nslice;

                int ohstart = 0;
                int ohend = 0;
                padding_t slice_pad;
                // 4th loop: split OH dim
                for (int ohidx = 0; ohidx < secs_info.ohsecs; ohidx++) {
                    // oh boundary: [ohstart, ohend)
                    // ih boundary: [ihstart, ihend)
                    ohstart = ohend;
                    ohslice = output_h / secs_info.ohsecs + ((output_h % secs_info.ohsecs) > ohidx);
                    ohend = ohstart + ohslice;
                    int ihstart = ohstart - pad_h0;
                    int ihend = ohend - 1 - pad_h0 + kh_ext;
                    slice_pad.top = ihstart < 0 ? -ihstart : ALIGN(ihstart, stride->h) - ihstart;
                    slice_pad.bottom = ihend > ih_ext ? ihend - ih_ext : (ihend - 1) % stride->h;
                    ihstart = ihstart < 0 ? 0 : DIV_UP(ihstart, stride->h);
                    ihend = ihend > ih_ext ? ishape.h : DIV_UP(ihend, stride->h);
                    ihslice = ihend - ihstart;

                    int owstart = 0;
                    int owend = 0;
                    // 5th loop: split OW dim
                    for (int owidx = 0; owidx < secs_info.owsecs; owidx++) {
                        // ow boundary: [owstart, owend)
                        // iw boundary: [iwstart, iwend)
                        owstart = owend;
                        owslice = output_w / secs_info.owsecs + ((output_w % secs_info.owsecs) > owidx);
                        owend = owstart + owslice;
                        int iwstart = owstart - pad_w0;
                        int iwend = owend - 1 - pad_w0 + kw_ext;
                        slice_pad.left = iwstart < 0 ? -iwstart : ALIGN(iwstart, stride->w) - iwstart;
                        slice_pad.right = iwend > iw_ext ? iwend - iw_ext : (iwend - 1) % stride->w;
                        iwstart = iwstart < 0 ? 0 : DIV_UP(iwstart, stride->w);
                        iwend = iwend > iw_ext ? ishape.w : DIV_UP(iwend, stride->w);
                        iwslice = iwend - iwstart;

                        dim4 islice_shape = {nslice, ic, ihslice, iwslice};
                        dim4 oslice_shape = {nslice, ocslice, ohslice, owslice};
                        // copy input from global
                        tpu_gdma_cpy_S2L(
                            (ping ? iaddr_ping : iaddr_pong),
                            input_global_addr + (nstart * istride.n + gidx * ic * istride.c +
                                ihstart * istride.h + iwstart) * tpu_data_type_size(idtype),
                               &islice_shape,
                               NULL,
                               &istride,
                               idtype);

                        if (parallel_branch) {
                            tpu_parallel_end();
                        }
                        tpu_parallel_start();
                        parallel_branch = true;

                        // deconv caculation
                        tpu_bdc_fp_conv2d_for_deconv2d(
                             ping ? oaddr_ping : oaddr_pong,
                             ping ? iaddr_ping : iaddr_pong,
                             waddr,
                             0,
                             &islice_shape,
                             NULL,
                             oslice_shape.c,
                             kernel,
                             &insert,
                             &slice_pad,
                             dilation,
                             odtype,
                             idtype,
                             false,
                             false);

                        // move output to global memory
                        if (nidx > 0 || ohidx > 0 || owidx > 0) {
                            dim4 last_oslice_shape = {last_nslice, ocslice, last_ohslice, last_owslice};
                            tpu_gdma_cpy_L2S(
                                    output_global_addr + (last_nstart * ostride.n +
                                        (gidx * oc + ocstart) * ostride.c +
                                        last_ohstart * ostride.h + last_owstart) *
                                        tpu_data_type_size(odtype),
                                    ping ? oaddr_pong : oaddr_ping,
                                    &last_oslice_shape,
                                    &ostride,
                                    NULL,
                                    odtype);
                        }
                        ping = !ping;
                        // save current info used for moving output to global memory next loop
                        last_nstart = nstart;
                        last_nslice = nslice;
                        last_ohstart = ohstart;
                        last_ohslice = ohslice;
                        last_owstart = owstart;
                        last_owslice = owslice;
                    }
                }
            }
            tpu_parallel_end();
            // move the last output to global memory
            dim4 last_oslice_shape = {last_nslice, ocslice, last_ohslice, last_owslice};
            tpu_gdma_cpy_L2S(
                    output_global_addr + (last_nstart * ostride.n +
                        (gidx * oc + ocstart) * ostride.c +
                        last_ohstart * ostride.h + last_owstart) *
                        tpu_data_type_size(odtype),
                    ping ? oaddr_pong : oaddr_ping,
                    &last_oslice_shape,
                    &ostride,
                    NULL,
                    odtype);
        }
    }
}

//grad_output reorder
// [ic, oc, kh, kw] => [1, oc, DIV_UP(ic, 32) * kh * kw, 32]
/*
  [ic, oc, 1, kh * kw] => [oc, ic, 1, kh * kw]
  [oc, ic, 1, kh * kw] => [oc, kh * kw, 1, ic]
  [oc, kh * kw, 1, ic] => [oc, kh * kw, DIV_UP(ic, 32), 32]
*/
void nodechip_grad_output_reorder(
    global_addr_t   grad_out_global_addr,
    global_addr_t   grad_out_reordered_global_addr,
    int             ic,
    int             oc,
    int             kh,
    int             kw,
    data_type_t     dtype
 ) {

    int ocsecs = 1;
    int icsecs = 1;
    int ocslice = oc;
    int icslice = ic;

    TPUKERNEL_ASSERT(ALIGN(kh * kw, tpu_eu_num(dtype)) <= 4 * LOCAL_BANK_SIZE);
    TPUKERNEL_ASSERT(DIV_UP(kh * kw, NPU_NUM) * ALIGN(1 * 1, tpu_eu_num(dtype)) <= 4 * LOCAL_BANK_SIZE);

    unsigned int ic_size = DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int ic_size_cwtrans = DIV_UP(kh * kw, NPU_NUM) * tpu_aligned_feature_size(1, ic, dtype);
    unsigned int max_ic_size = MAX(ic_size, ic_size_cwtrans);
    int slice = (unsigned int)(4 * LOCAL_BANK_SIZE) / max_ic_size;
    if (slice == 0) {
        bool valid = false;
        while (!valid) {
            icsecs++;
            icslice = DIV_UP(ic, icsecs);
            ic_size = DIV_UP(icslice, NPU_NUM) * tpu_aligned_feature_size(kh , kw, dtype);
            ic_size_cwtrans = DIV_UP(kh * kw, NPU_NUM) * tpu_aligned_feature_size(1, icslice, dtype);
            max_ic_size = MAX(ic_size, ic_size_cwtrans);
            valid = (unsigned int)(4 * LOCAL_BANK_SIZE) / max_ic_size > 0;
        }
        ocslice = 1;
        ocsecs = oc;
    } else if (slice == 1) {
        icslice = ic;
        icsecs = 1;
        ocslice = 1;
        ocsecs = oc;
    } else if (slice > 1) {
        icslice = ic;
        icsecs = 1;
        ocslice = slice;
        ocsecs = DIV_UP(oc, ocslice);
    }

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = 4 * LOCAL_BANK_SIZE;
    local_addr_t oaddr_ping = 8 * LOCAL_BANK_SIZE;
    local_addr_t oaddr_pong = 12 * LOCAL_BANK_SIZE;

    bool ping = true;
    bool parallel_branch = false;

    // [ic, oc, kh, kw] => [oc, ic, 1, kh * kw] => [oc, 1, kh * kw, ic] => [1, oc, kh * kw, ic]
    int last_ocstart = 0;
    int last_ocslice = 0;
    int last_icstart = 0;
    int last_icslice = 0;

    int ocstart = 0;
    int ocend = 0;
    for (int ocidx = 0; ocidx < ocsecs; ocidx++) {
        ocstart = ocend;
        ocslice = MIN(ocslice, oc - ocend);
        ocend = ocstart + ocslice;

        int icstart = 0;
        int icend = 0;
        for (int icidx = 0; icidx < icsecs; icidx++) {
            icstart = icend;
            icslice = MIN(icslice, ic - icend);
            icend = icstart + icslice;

            // [ic, oc, 1, kh * kw] => [oc, ic, 1, kh * kw]
            dim4 nc_dst_shape = {ocslice, icslice, 1, kh * kw};
            dim4 src_stride = {kh * kw, oc * kh * kw, kh * kw, 1};
            tpu_gdma_cpy_nc_trans_S2L(
                ping ? iaddr_ping : iaddr_pong,
                grad_out_global_addr +
                    (icstart * oc  + ocstart) * kh * kw * tpu_data_type_size(dtype),
                &nc_dst_shape,
                NULL,
                &src_stride,
                dtype);

            if (parallel_branch) {
                tpu_parallel_end();
            }
            tpu_parallel_start();
            parallel_branch = true;

            dim4 cw_dst_shape = {ocslice, kh * kw, 1, icslice};
            // [oc, ic, 1, kh * kw] => [oc, kh * kw, 1, ic]
            if (cw_dst_shape.w <= cw_dst_shape.c) {
                tpu_bdc_cw_trans(
                    ping ? oaddr_ping : oaddr_pong,
                    ping ? iaddr_ping : iaddr_pong,
                    &cw_dst_shape,
                    dtype);
            } else {
                tpu_bdc_wc_trans(
                    ping ? oaddr_ping : oaddr_pong,
                    ping ? iaddr_ping : iaddr_pong,
                    &cw_dst_shape,
                    dtype);
            }
            if (icidx > 0 || ocidx > 0) {
                dim4 reordered_shape = {last_ocslice,
                                        kh * kw,
                                        DIV_UP(last_icslice, 32),
                                        last_icslice >= 32 ? 32 : last_icslice};
                dim4 reordered_stride = {DIV_UP(ic, 32) * kh * kw * 32, 32, kh * kw * 32, 1};
                tpu_gdma_cpy_L2S(
                    grad_out_reordered_global_addr +
                        last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                        last_icstart / 32 * kh * kw * 32 +
                        last_icstart % 32,
                    ping ? oaddr_pong : oaddr_ping,
                    &reordered_shape,
                    &reordered_stride,
                    NULL,
                    dtype);
            }
            ping = !ping;
            last_ocstart = ocstart;
            last_ocslice = ocslice;
            last_icstart = icstart;
            last_icslice = icslice;
        }
    }
    tpu_parallel_end();
    dim4 last_reordered_shape = {last_ocslice,
                                 kh * kw,
                                 DIV_UP(last_icslice, 32),
                                 last_icslice >= 32 ? 32 : last_icslice};
    dim4 last_reordered_stride = {DIV_UP(ic, 32) * kh * kw * 32, 32, kh * kw * 32, 1};
    tpu_gdma_cpy_L2S(
        grad_out_reordered_global_addr +
            last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
            last_icstart / 32 * kh * kw * 32 +
            last_icstart % 32,
        ping ? oaddr_pong : oaddr_ping,
        &last_reordered_shape,
        &last_reordered_stride,
        NULL,
        dtype);
}

// forward_input op grad_out => grad_weight
// [ic, n, ih, iw] conv [oc, n, oh, ow] => [ic, oc, kh, kw]
void nodechip_conv_backward_weight(
    global_addr_t    forward_input_global_addr,
    global_addr_t    grad_out_global_addr,
    global_addr_t    grad_out_reordered_global_addr,
    global_addr_t    grad_weight_global_addr,
    global_addr_t    grad_bias_global_addr,
    const int        groups,
    const dim4       *forward_input_shape,
    const dim4       *grad_out_shape,
    const dim2       *forward_kernel,
    const dim2       *stride,
    const dim2       *dilation,
    const padding_t  *pad,
    bool             grad_bias_enable
 ) {
    const data_type_t idtype = DT_FP16, odtype = DT_FP16;

    int n = forward_input_shape->c;
    int ic = forward_input_shape->n;
    int ih = forward_input_shape->h;
    int iw = forward_input_shape->w;
    int oc = grad_out_shape->c;
    int oh = forward_kernel->h;
    int ow = forward_kernel->w;
    int kh = grad_out_shape->h;
    int kw = grad_out_shape->w;
    int kh_ext = dilation->h * (kh - 1) + 1;
    int kw_ext = dilation->w * (kw - 1) + 1;
    int ih_ext = (ih - 1) + pad->top + pad->bottom + 1;
    int iw_ext = (iw - 1) + pad->left + pad->right + 1;
    int output_h = (ih_ext - kh_ext) / stride->h + 1;
    int output_w = (iw_ext - kw_ext) / stride->w + 1;
    TPUKERNEL_ASSERT(output_h == oh);
    TPUKERNEL_ASSERT(output_w == ow);

    dim4 ishape = {n, ic, ih, iw};
    dim4 oshape = {n, oc, oh, ow};
    dim2 kernel = {kh, kw};

    grad_weight_secs_info_t secs_info;
    // ocsecs can more
    // nsecs need less
    grad_weight_split(
            &ishape,
            &oshape,
            groups,
            &kernel,
            dilation,
            stride,
            idtype,
            odtype,
            grad_bias_enable,
            &secs_info);

    bool split_kernel = secs_info.khsecs != 1 || secs_info.kwsecs != 1;

    int nslice = DIV_UP(ishape.n, secs_info.nsecs);
    int ocslice = DIV_UP(oc, secs_info.ocsecs);
    ocslice = secs_info.ocsecs > 1 ? ALIGN(ocslice, NPU_NUM) : ocslice;
    int khslice = DIV_UP(kh, secs_info.khsecs);
    int kwslice = DIV_UP(kw, secs_info.kwsecs);
    kh_ext = dilation->h * (kh - 1) + 1;
    kw_ext = dilation->w * (kw - 1) + 1;
    int ihslice = MIN((oh - 1) * stride->h + kh_ext, ishape.h);
    int iwslice = MIN((ow - 1) * stride->w + kw_ext, ishape.w);
    TPUKERNEL_DBG("nsecs=%d  nslice=%d\n", secs_info.nsecs, nslice);
    TPUKERNEL_DBG("ocsecs=%d  ocslice=%d\n", secs_info.ocsecs, ocslice);
    TPUKERNEL_DBG("khsecs=%d  khslice=%d\n", secs_info.khsecs, khslice);
    TPUKERNEL_DBG("kwsecs=%d  kwslice=%d\n", secs_info.kwsecs, kwslice);
    unsigned int wsize = DIV_UP(ocslice, NPU_NUM) *
        (idtype == DT_FP32 ? ic: ALIGN(ic, 32)) *
        khslice * kwslice * tpu_data_type_size(idtype);
    unsigned int isize = nslice * DIV_UP(ic, NPU_NUM) *
        tpu_aligned_feature_size(ihslice, iwslice, idtype);
    unsigned int osize = nslice * DIV_UP(ocslice, NPU_NUM) *
        tpu_aligned_feature_size(oh, ow, odtype);
    unsigned int buffer_size = 0;
    if (grad_bias_enable) buffer_size = DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 32, idtype);
    if (secs_info.khsecs != 1 || secs_info.kwsecs != 1) buffer_size = MAX(buffer_size, osize);
    unsigned int grad_bias_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * ALIGN(1 * 1, tpu_eu_num(idtype)) : 0;
    local_addr_t iaddr = 0;
    local_addr_t waddr_ping = ALIGN(iaddr + isize, BANK_SIZE);
    local_addr_t waddr_pong = ALIGN(waddr_ping + wsize, BANK_SIZE);
    if (!split_kernel) waddr_pong = waddr_ping;
    local_addr_t oaddr_ping = ALIGN(waddr_pong + isize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    local_addr_t buffer_addr = ALIGN(oaddr_pong + buffer_size, BANK_SIZE);
    TPUKERNEL_ASSERT(buffer_addr + buffer_size <= (unsigned int)LOCAL_MEM_SIZE);
    local_addr_t grad_bias_addr_ping = grad_bias_enable ? ALIGN(buffer_addr + buffer_size, BANK_SIZE) : 0;
    local_addr_t grad_bias_addr_pong = grad_bias_enable ? ALIGN(grad_bias_addr_ping + grad_bias_size, BANK_SIZE) : 0;
    TPUKERNEL_ASSERT(buffer_addr + buffer_size <= (unsigned int)LOCAL_MEM_SIZE);
    if (grad_bias_enable)
        TPUKERNEL_ASSERT(grad_bias_addr_pong + grad_bias_size <= (unsigned int)LOCAL_MEM_SIZE);

    TPUKERNEL_DBG("in local addr = 0x%5x, bank id = %d\n", iaddr, iaddr / BANK_SIZE);
    TPUKERNEL_DBG("weight ping local addr = 0x%5x, bank id = %d\n", waddr_ping, waddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("weight pong local addr = 0x%5x, bank id = %d\n", waddr_pong, waddr_pong / BANK_SIZE);
    TPUKERNEL_DBG("out ping local addr = 0x%5x, bank id = %d\n", oaddr_ping, oaddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("out pong local addr = 0x%5x, bank id = %d\n", oaddr_pong, oaddr_pong / BANK_SIZE);
    TPUKERNEL_DBG("buffer local addr = 0x%5x, bank id = %d\n", buffer_addr, buffer_addr / BANK_SIZE);
    if (grad_bias_enable) {
        TPUKERNEL_DBG("grad_bias ping local addr = 0x%5x, bank id = %d\n", grad_bias_addr_ping, grad_bias_addr_ping / BANK_SIZE);
        TPUKERNEL_DBG("grad_bias ping local addr = 0x%5x, bank id = %d\n", grad_bias_addr_pong, grad_bias_addr_pong / BANK_SIZE);
    }

    // input nc_trans cpy
    dim4 istride = {
        ishape.n * ishape.h * ishape.w,
        ishape.h * ishape.w,
        ishape.w,
        1};
    dim4 ostride = {
        oshape.c * oshape.h * oshape.w,
        oshape.c * oshape.w,
        oshape.w,
        1};
    dim4 wshape = {
        1,
        oc,
        (idtype == DT_FP32 ? ic : DIV_UP(ic, 32)) * kh * kw,
        idtype == DT_FP32 ? 1 : 32};
    dim4 wstride;
    tpu_compact_stride(&wstride, 0, &wshape);

    nodechip_grad_output_reorder(grad_out_global_addr,
                                 grad_out_reordered_global_addr,
                                 ic,
                                 oc,
                                 kh,
                                 kw,
                                 idtype);
/*
    // grad_out 32n reorder, noneed nc_trans
    u32 grad_out_size = grad_out_shape->n * grad_out_shape->c * grad_out_shape->h * grad_out_shape->w * sizeof(short);
    u32 grad_out_reordered_size = ALIGN(grad_out_shape->n, 32) * grad_out_shape->c * grad_out_shape->h * grad_out_shape->w * sizeof(short); 
    u8 grad_out_data[grad_out_size];
    u8 grad_out_reordered_data[grad_out_reordered_size];
    memset((void*)grad_out_reordered_data, 0, grad_out_reordered_size);
    tpu_invalidate_cache(grad_out_global_addr, ALIGN(grad_out_size, 64));
    memcpy(grad_out_data, GET_GLOBAL_ADDR(grad_out_global_addr), grad_out_size);

    for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
      for (int ic_idx = 0; ic_idx < ceiling_func(ic, 32); ic_idx++) {
        for (int k_idx = 0; k_idx < kh * kw;  k_idx++) {
          for (int inner = 0; inner < 32; inner++) {
            if (ic_idx * 32 + inner >= ic) break;
//            int src_idx = oc_idx * ic * kh * kw +
//                          (ic_idx * 32 + inner) * kh * kw +
//                          k_idx;
            int src_idx = (ic_idx * 32 + inner) * oc * kh * kw +
                          oc_idx * kh * kw +
                          k_idx;
            int dst_idx = oc_idx * ceiling_func(ic, 32) * kh * kw * 32 +
                          ic_idx * kh * kw * 32 +
                          k_idx * 32 +
                          inner;
            ((u16*)grad_out_reordered_data)[dst_idx] = ((u16*)grad_out_data)[src_idx];
          }
        }
      }
    }
    memcpy(GET_GLOBAL_ADDR(grad_out_reordered_global_addr), grad_out_reordered_data, grad_out_reordered_size);
    tpu_flush_cache(grad_out_reordered_global_addr, ALIGN(grad_out_reordered_size, 64));
*/

    global_addr_t input_global_addr = forward_input_global_addr;
    global_addr_t weight_global_addr = grad_out_reordered_global_addr;
    global_addr_t output_global_addr = grad_weight_global_addr;

    int last_ocslice = 0;
    int last_ocstart = 0;

    bool ping = true;
    // 1st loop: groups
    for (int gidx = 0; gidx < groups; gidx++) {

        int nstart = 0;
        int nend = 0;
        // 2nd loop: split N dim
        for (int nidx = 0; nidx < secs_info.nsecs; nidx++) {
            nstart = nend;
            nslice = ishape.n / secs_info.nsecs + ((ishape.n % secs_info.nsecs) > nidx);
            nend = nstart + nslice;

            dim4 input_nc_shape = {nslice, ic, ih, iw};
            tpu_gdma_cpy_nc_trans_S2L(
                    iaddr,
                    input_global_addr +
                        (gidx * ic * istride.n +
                         nstart * istride.c +
                         ih * istride.h +
                         iw) * tpu_data_type_size(idtype),
                    &input_nc_shape,
                    NULL,
                    &istride,
                    idtype);

            bool parallel_branch = false;
            int ocstart = 0;
            int ocend = 0;
            // 2nd loop: split OC dim
            for (int ocidx = 0; ocidx < secs_info.ocsecs; ocidx++) {
                // oc boundary: [ocstart, ocend)
                ocstart = ocend;
                ocslice = MIN(ocslice, oc - ocend);
                ocend = ocstart + ocslice;

                int khstart = 0;
                int khend = 0;
                // 2nd loop: split KH dim
                for (int khidx = 0; khidx < secs_info.khsecs; khidx++) {
                    khstart = khend;
                    khslice = MIN(khslice, kh - khend);
                    khend = khstart + khslice;

                    int kwstart = 0;
                    int kwend = 0;
                    // 2nd loop: split KW dim
                    for (int kwidx = 0; kwidx < secs_info.kwsecs; kwidx++) {
                        kwstart = kwend;
                        kwslice = MIN(kwslice, kw - kwend);
                        kwend = kwstart + kwslice;

                        dim4 wslice_shape = {
                            1,
                            ocslice,
                            (idtype == DT_FP32 ? ic : DIV_UP(ic, 32)) * khslice * kwslice,
                            idtype == DT_FP32 ? 1 : 32};

                        dim4 wslice_stride;
                        tpu_compact_stride(&wslice_stride, 0, &wslice_shape);
                        tpu_gdma_cpy_S2L(
                            ping ? waddr_ping : waddr_pong,
                            weight_global_addr +
                                ((gidx * oc + ocstart) * wstride.c +
                                 (ic * kh * kw + khstart * kw + kwstart) * (idtype == DT_FP32 ? 1 : 32)) * tpu_data_type_size(idtype),
                            &wslice_shape,
                            &wslice_stride,
                            &wstride,
                            idtype);

                        kh_ext = dilation->h * (kh - 1) + 1;
                        kw_ext = dilation->w * (kw - 1) + 1;
                        padding_t pad_slice;
                        int ihstart = khstart;
                        int ihend = (oh - 1) * stride->h + kh_ext - pad->top;
                        pad_slice.top = ihstart < 0 ? -ihstart : 0;
                        pad_slice.bottom = ihend > ishape.h ? ihend - ishape.h : 0;
                        ihstart = ihstart < 0 ? 0 : ihstart;
                        ihend = ihend > ishape.h ? ishape.h : ihend;
                        ihslice = ihend - ihstart;

                        int iwstart = kwstart;
                        int iwend = (ow - 1) * stride->w + kw_ext - pad->left;
                        pad_slice.left = iwstart < 0 ? -iwstart : 0;
                        pad_slice.right = iwend > ishape.w ? iwend - ishape.w : 0;
                        iwstart = iwstart < 0 ? 0 : iwstart;
                        iwend = iwend > ishape.w ? ishape.w : iwend;
                        iwslice = iwend - iwstart;

                        dim4 islice_shape = {nslice, ic, ihslice, iwslice};
                        dim4 oslice_shape = {nslice, ocslice, oh, ow};

                        if (parallel_branch) {
                            tpu_parallel_end();
                        }
                        tpu_parallel_start();
                        parallel_branch = true;

                        dim2 kernel_slice;
                        kernel_slice.h = khslice;
                        kernel_slice.w = kwslice;
                        tpu_bdc_fp_conv2d(
                                khidx == 0 && kwidx == 0 ? ping ? oaddr_ping : oaddr_pong : buffer_addr,
                                iaddr,
                                ping ? waddr_ping : waddr_pong,
                                0,
                                &islice_shape,
                                NULL,
                                oslice_shape.c,
                                &kernel_slice,
                                &pad_slice,
                                stride,
                                dilation,
                                odtype,
                                idtype,
                                false,
                                false);
                        if (khidx > 0 || kwidx > 0)
                            tpu_bdc_fp_add(
                                ping ? oaddr_ping : oaddr_pong,
                                ping ? oaddr_ping : oaddr_pong,
                                buffer_addr,
                                &oslice_shape,
                                NULL,
                                NULL,
                                NULL,
                                odtype);
                        if (grad_bias_enable && (nidx == 0)) {
                            dim4 grad_bias_shape = {wslice_shape.n,
                                                    wslice_shape.c,
                                                    wslice_shape.h,
                                                    wslice_shape.w};
                            scalar_t scale = {.f32 = 1.f};
                            dim2 pooling_kernel = {grad_bias_shape.h, 1};
                            padding_t pooling_padding = {0, 0, 0, 0};
                            dim2 pooling_stride = {1, 1};
                            dim2 pooling_dilation = {1, 1};
                            tpu_bdc_fp_avg_pool2d(
                                buffer_addr,
                                ping ? waddr_ping : waddr_pong,
                                &grad_bias_shape,
                                &pooling_kernel,
                                &pooling_padding,
                                &pooling_stride,
                                &pooling_dilation,
                                idtype,
                                tpu_cast(scale, idtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
                            grad_bias_shape.h = 1;
                            pooling_kernel.h = 1;
                            pooling_kernel.w = grad_bias_shape.w;
                            tpu_bdc_fp_avg_pool2d(
                                khidx == 0 && kwidx == 0 ?
                                    ping ? grad_bias_addr_ping : grad_bias_addr_pong : buffer_addr,
                                buffer_addr,
                                &grad_bias_shape,
                                &pooling_kernel,
                                &pooling_padding,
                                &pooling_stride,
                                &pooling_dilation,
                                idtype,
                                tpu_cast(scale, idtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
                            if (khidx != 0 || kwidx != 0)
                                tpu_bdc_fp_add(
                                    ping ? grad_bias_addr_ping : grad_bias_addr_pong,
                                    ping ? grad_bias_addr_ping : grad_bias_addr_pong,
                                    buffer_addr,
                                    &grad_bias_shape,
                                    NULL,
                                    NULL,
                                    NULL,
                                    idtype);
                            if (ocidx > 0) {
                                bool do_cw_trans = last_ocslice > 1;
                                dim4 move_bias_shape = {1,
                                                        do_cw_trans ? 1 : last_ocslice,
                                                        1,
                                                        do_cw_trans ? last_ocslice : 1};
                                if (do_cw_trans) {
                                    tpu_bdc_cw_trans(
                                        buffer_addr,
                                        ping ? grad_bias_addr_pong : grad_bias_addr_ping,
                                        &move_bias_shape,
                                        idtype);
                                }
                                tpu_gdma_cpy_L2S(
                                    grad_bias_global_addr +
                                        last_ocstart * tpu_data_type_size(idtype),
                                    do_cw_trans ? buffer_addr :
                                                  ping ? grad_bias_addr_pong :
                                                         grad_bias_addr_ping,
                                    &move_bias_shape,
                                    NULL,
                                    NULL,
                                    idtype);
                            }
                        }

                        if (ocidx > 0) {
                            dim4 last_oslice_shape = {nslice, last_ocslice, oh, ow};
                            tpu_gdma_cpy_L2S(
                                output_global_addr +
                                    (nstart * ostride.n +
                                     (gidx * oc + last_ocstart) * ostride.c +
                                     oh * ostride.h + ow) * tpu_data_type_size(odtype),
                                ping ? oaddr_pong : oaddr_ping,
                                &last_oslice_shape,
                                &ostride,
                                NULL,
                                odtype);
                        }
                        ping = !ping;
                        // save current info used for moving output to global memory next loop
                        last_ocstart = ocstart;
                        last_ocslice = ocslice;
                    }
                }
            }
            tpu_parallel_end();
            // move last output to global memory
            dim4 last_oslice_shape = {nslice, last_ocslice, oh, ow};
            tpu_gdma_cpy_L2S(
                output_global_addr +
                    (nstart * ostride.n +
                     ((groups - 1) * oc + last_ocstart) * ostride.c +
                     oh * ostride.h + ow) * tpu_data_type_size(odtype),
                ping ? oaddr_pong : oaddr_ping,
                &last_oslice_shape,
                &ostride,
                NULL,
                odtype);
            if (grad_bias_enable && nidx == 0) {
                bool do_cw_trans = last_ocslice > 1;
                dim4 move_bias_shape = {1,
                                        do_cw_trans ? 1 : last_ocslice,
                                        1,
                                        do_cw_trans ? last_ocslice : 1};
                if (do_cw_trans) {
                    tpu_bdc_cw_trans(
                        buffer_addr,
                        ping ? grad_bias_addr_pong : grad_bias_addr_ping,
                        &move_bias_shape,
                        idtype);
                }
                tpu_gdma_cpy_L2S(
                    grad_bias_global_addr +
                        last_ocstart * tpu_data_type_size(idtype),
                    do_cw_trans ? buffer_addr :
                                  ping ? grad_bias_addr_pong :
                                         grad_bias_addr_ping,
                    &move_bias_shape,
                    NULL,
                    NULL,
                    idtype);
            }
        }
    }
}

void nodechip_conv_backward(
    global_addr_t    grad_out_global_addr,
    global_addr_t    input_global_addr,
    global_addr_t    weight_global_addr,
    global_addr_t    grad_input_global_addr,
    global_addr_t    grad_weight_global_addr,
    global_addr_t    grad_bias_global_addr,
    global_addr_t    buffer_global_addr,
    const dim4       *input_shape,
    const dim4       *grad_out_shape,
    const dim2       *kernel,
    const dim2       *stride,
    const dim2       *dilation,
    const padding_t  *pad,
    bool             grad_input_enable,
    bool             grad_weight_enable,
    bool             grad_bias_enable
 ) {
    if (grad_input_enable) {
        nodechip_conv_backward_input(
            grad_out_global_addr,
            grad_input_global_addr,
            weight_global_addr,
            buffer_global_addr,
            1,//groups TODO
            input_shape,
            grad_out_shape,
            kernel,
            stride,
            dilation,
            pad);
    }
    if (grad_weight_enable) {
        nodechip_conv_backward_weight(
            input_global_addr,
            grad_out_global_addr,
            buffer_global_addr,
            grad_weight_global_addr,
            grad_bias_global_addr,
            1,//groups TODO
            input_shape,
            grad_out_shape,
            kernel,
            stride,
            dilation,
            pad,
            grad_bias_enable);
    }
}

void tpu_kernel_api_conv_backward(const void* args) {
    sg_api_conv_backward_t* api = (sg_api_conv_backward_t*)args;

    dim4 input_shape = {api->input_shape[0], api->input_shape[1],
                        api->input_shape[2], api->input_shape[3]};
    dim4 output_shape = {api->output_shape[0], api->output_shape[1],
                         api->output_shape[2], api->output_shape[3]};
    dim2 kernel = {api->kernel[0], api->kernel[1]};
    dim2 stride = {api->stride[0], api->stride[1]};
    dim2 dilation = {api->dilation[0], api->dilation[1]};
    padding_t pad = {api->pad[0], api->pad[1], api->pad[2], api->pad[3]};

    tpu_initialize();
    nodechip_conv_backward(
        api->grad_output_global_addr,
        api->input_global_addr,
        api->weight_global_addr,
        api->grad_input_global_addr,
        api->grad_weight_global_addr,
        api->grad_bias_global_addr,
        api->buffer_global_addr,
        &input_shape,
        &output_shape,
        &kernel,
        &stride,
        &dilation,
        &pad,
        api->grad_input_enable == 1 ? true : false,
        api->grad_weight_enable == 1 ? true : false,
        api->grad_bias_enable == 1 ? true : false);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_conv_backward);
