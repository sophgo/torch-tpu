#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"

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

static inline bool split_kh_or_kw(
        int *slice,
        int *secs,
        int stride,
        int ic,
        int oc,
        int ih_or_iw,
        int oh_or_ow,
        int kh_or_kw,
        int dh_or_dw,
        int other_size,
        data_type_t idtype,
        data_type_t odtype) {
    bool valid = false;
    int slice_new = *slice;
    do {
        if (slice_new == 1) {
            *slice = slice_new;
            --(*secs);
            return false;
        }
        slice_new = DIV_UP(*slice, *secs);

        int kslice_ext = dh_or_dw * (slice_new - 1) + 1;
        int islice = (oh_or_ow - 1) * stride + kslice_ext;

        unsigned int wsize = (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) * slice_new * kh_or_kw * tpu_data_type_size(idtype);
        unsigned int isize = DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(islice, ih_or_iw, idtype);
        unsigned int total_size = ALIGN(isize, BANK_SIZE) * 2 +
                                  ALIGN(wsize, BANK_SIZE) * 2 +
                                  other_size;
        valid = total_size <= (unsigned int)LOCAL_MEM_SIZE;

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

    unsigned int isize = nslice * DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(ih, iw, idtype);
    unsigned int wsize = DIV_UP(ocslice, NPU_NUM) * (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) * khslice * kwslice * tpu_data_type_size(idtype);
    unsigned int osize = nslice * DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(oh, ow, odtype);
    unsigned int grad_bias_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 1, idtype): 0;
    unsigned int buffer_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 32, idtype) : 0;

    unsigned int total_size = ALIGN(isize, BANK_SIZE) +
                              ALIGN(wsize, BANK_SIZE) +
                              ALIGN(osize, BANK_SIZE) +
                              ALIGN(grad_bias_size, BANK_SIZE) +
                              (buffer_size > 0 ? ALIGN(buffer_size, BANK_SIZE) : 0);
    if (total_size <= (unsigned int)LOCAL_MEM_SIZE) return;

    unsigned int pern_isize = DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(ih, iw, idtype);
    unsigned int pern_osize = DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(oh, ow, odtype);
    unsigned int pern_buffer_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 32, idtype) : 0;
    unsigned int pern_total_size = ALIGN(pern_isize, BANK_SIZE) * 2 +
                                   ALIGN(pern_osize, BANK_SIZE) * 2 +
                                   ALIGN(wsize, BANK_SIZE) +
                                   ALIGN(grad_bias_size, BANK_SIZE) +
                                   (pern_buffer_size > 0 ? ALIGN(pern_buffer_size, BANK_SIZE) : 0);
    if (pern_total_size <= (unsigned int)LOCAL_MEM_SIZE) {
        unsigned int nslice_iosize = (unsigned int)LOCAL_MEM_SIZE
                                   - ALIGN(wsize, BANK_SIZE)
                                   - ALIGN(grad_bias_size, BANK_SIZE);
        while(1) {
            if (secs_info->nsecs == n) break;
            unsigned int nslice_isize = nslice * pern_isize;
            unsigned int nslice_osize = nslice * pern_osize;
            pern_buffer_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 32, idtype) : 0;
            if (ALIGN(nslice_isize, BANK_SIZE) * 2 +
                ALIGN(nslice_osize, BANK_SIZE) * 2 +
                (pern_buffer_size > 0 ? ALIGN(pern_buffer_size, BANK_SIZE) : 0) <= nslice_iosize) {
                break;
            } else {
                secs_info->nsecs++;
                nslice = DIV_UP(n, secs_info->nsecs);
            }
        }
        return;
    } else {
        nslice = 1;
        secs_info->nsecs = n;
    }

    unsigned int pern_64oc_isize = DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(ih, iw, idtype);
    unsigned int pern_64oc_osize = tpu_aligned_feature_size(oh, ow, odtype);
    unsigned int pern_64oc_wsize = (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) * khslice * kwslice * tpu_data_type_size(idtype);
    unsigned int pern_64oc_grad_bias_size = grad_bias_enable ? tpu_aligned_feature_size(1, 1, idtype): 0;
    unsigned int pern_64oc_buffer_size = grad_bias_enable ? tpu_aligned_feature_size(1, 32, idtype) : 0;
    unsigned int pern_64oc_total_size = ALIGN(pern_64oc_isize, BANK_SIZE) * 2 +
                                        ALIGN(pern_64oc_osize, BANK_SIZE) * 2 +
                                        ALIGN(pern_64oc_wsize, BANK_SIZE) * (oc > NPU_NUM ? 2 : 1) +
                                        ALIGN(pern_64oc_grad_bias_size, BANK_SIZE) * (oc > NPU_NUM ? 2 : 1) +
                                        (pern_64oc_buffer_size > 0 ? ALIGN(pern_64oc_buffer_size, BANK_SIZE) : 0);
    if (pern_64oc_total_size <= (unsigned int)LOCAL_MEM_SIZE) {
        ocslice = oc;
        secs_info->ocsecs = 1;
        bool first_time = true;
        while(1) {
            if (!first_time) {
                secs_info->ocsecs++;
                ocslice = DIV_UP(oc, secs_info->ocsecs);
            }
            unsigned int pern_some_64oc_wsize = DIV_UP(ocslice, NPU_NUM) * pern_64oc_wsize;
            unsigned int pern_some_64oc_osize = DIV_UP(ocslice, NPU_NUM) * pern_64oc_osize;
            unsigned int pern_some_64oc_grad_bias_size = grad_bias_enable ? pern_64oc_grad_bias_size : 0;
            unsigned int pern_some_64oc_buffer_size = grad_bias_enable ? pern_64oc_buffer_size : 0;
            unsigned int pern_some_64oc_total_size = ALIGN(pern_64oc_isize, BANK_SIZE) * 2 +
                                                     ALIGN(pern_some_64oc_osize, BANK_SIZE) * 2 +
                                                     ALIGN(pern_some_64oc_wsize, BANK_SIZE) * (secs_info->ocsecs > 1 ? 2 : 1) +
                                                     ALIGN(pern_some_64oc_grad_bias_size, BANK_SIZE) * (secs_info->ocsecs > 1 ? 2 : 1) +
                                                     (pern_some_64oc_buffer_size > 0 ? ALIGN(pern_some_64oc_buffer_size, BANK_SIZE) : 0);
            if (pern_some_64oc_total_size <= (unsigned int)LOCAL_MEM_SIZE) break;
            first_time = false;
        }
        return;
    } else {
        ocslice = MIN(NPU_NUM, oc);
        secs_info->ocsecs = DIV_UP(oc, ocslice);
    }

    //split kh&kw
    bool valid = false;
    pern_64oc_buffer_size = MAX(pern_64oc_buffer_size, pern_64oc_osize);
    unsigned int other_size = ALIGN(pern_64oc_osize, BANK_SIZE) * 2 +
                              ALIGN(pern_64oc_grad_bias_size, BANK_SIZE) * (secs_info->ocsecs > 1 ? 2 : 1) +
                              ALIGN(pern_64oc_buffer_size, BANK_SIZE);
    valid = split_kh_or_kw(
            &khslice,
            &(secs_info->khsecs),
            stride->h,
            ic,
            oc,
            iw,
            oh,
            kwslice,
            dilation->h,
            other_size,
            idtype,
            odtype);
    if (!valid) {
        int khslice_ext_ = dilation->h * (khslice - 1) + 1;
        int ihslice_ = MIN((oh - 1) * stride->h + khslice_ext_, ih);
        valid = split_kh_or_kw(
                &kwslice,
                &(secs_info->kwsecs),
                stride->w,
                ic,
                oc,
                ihslice_,
                ow,
                khslice,
                dilation->w,
                other_size,
                idtype,
                odtype);
    }
    TPUKERNEL_ASSERT(valid);

    //can process more n one time
    TPUKERNEL_ASSERT(secs_info->nsecs == n);
    TPUKERNEL_ASSERT(nslice == 1);
    TPUKERNEL_ASSERT(secs_info->ocsecs == DIV_UP(oc, NPU_NUM));
    TPUKERNEL_ASSERT(ocslice == MIN(oc, NPU_NUM));
    secs_info->nsecs = 1;
    nslice = n;
    bool first_time = true;
    int khslice_ext = dilation->h * (khslice - 1) + 1;
    int kwslice_ext = dilation->w * (kwslice - 1) + 1;
    int ihslice = MIN((oh - 1) * stride->h + khslice_ext, ih);
    int iwslice = MIN((ow - 1) * stride->w + kwslice_ext, iw);
    unsigned int some_n_wsize = DIV_UP(ocslice, NPU_NUM) * (idtype == DT_FP32 ? ic : ALIGN(ic, 32)) * khslice * kwslice * tpu_data_type_size(idtype);
    unsigned int some_n_grad_bias_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 1, idtype): 0;
    unsigned int some_n_buffer_size = grad_bias_enable ? DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(1, 32, idtype) : 0;
    while(1) {
        if (!first_time) {
            secs_info->nsecs++;
            nslice = DIV_UP(n, secs_info->nsecs);
        }
        unsigned int some_n_isize = nslice * DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(ihslice, iwslice, idtype);
        unsigned int some_n_osize = nslice * DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(oh, ow, odtype);
        some_n_buffer_size = MAX(some_n_buffer_size, some_n_osize);
        unsigned int some_n_total_size = ALIGN(some_n_isize, BANK_SIZE) * 2 +
                                         ALIGN(some_n_osize, BANK_SIZE) * 2 +
                                         ALIGN(some_n_wsize, BANK_SIZE) * 2 +
                                         ALIGN(some_n_grad_bias_size, BANK_SIZE) * (secs_info->ocsecs > 1 ? 2 : 1) +
                                         ALIGN(some_n_buffer_size, BANK_SIZE);
        if (some_n_total_size <= (unsigned int)LOCAL_MEM_SIZE) break;
        first_time = false;
    }
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
  [1, ic, DIV_UP(oc, 32) * kh * kw, 32]
*/

/*
 * 32oc -> 32ic
 * [1, ic, DIV_UP(oc, 32) * kh * kw, 32] => [1, oc, DIV_UP(ic, 32) * kh * kw, 32]
 *
 * [1, ic, DIV_UP(oc, 32) * kh * kw, 32]
   [kh * kw, ic, DIV_UP(oc, 32), 32] => [kh * kw, ic, 1, oc]
   [kh * kw, oc, 1, ic] => [kh * kw, oc, DIV_UP(ic, 32), 32]
   [1, oc, DIV_UP(ic, 32) * kh * kw, 32]
 *
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
    unsigned int kernel_min_size = MAX(DIV_UP(64, 32) * kh * kw * 32,
                                       DIV_UP(64, 32) * kh * kw * 32);//32 * kh * kw;
    TPUKERNEL_ASSERT(kernel_min_size <= (unsigned int)(LOCAL_MEM_SIZE / 4));
    ocslice = DIV_UP((unsigned int)(LOCAL_MEM_SIZE / 4), kernel_min_size);
    icslice = DIV_UP((unsigned int)(LOCAL_MEM_SIZE / 4),
                        DIV_UP(ic, 64) * ALIGN(DIV_UP(oc, 32) * kh * kw * 32, tpu_eu_num(dtype)));

    icsecs = DIV_UP(ic, icslice);
    ocsecs = DIV_UP(oc, ocslice);

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = 4 * BANK_SIZE;
    local_addr_t oaddr_ping = 8 * BANK_SIZE;
    local_addr_t oaddr_pong = 12 * BANK_SIZE;

    bool ping = true;
    bool parallel_branch = false;

    int last_ocstart = 0;
    int last_ocslice = 0;
    int last_icstart = 0;
    int last_icslice = 0;

    int icstart = 0;
    int icend = 0;
    for (int icidx = 0; icidx < icsecs; icidx++) {
        icstart = icend;
        //icslice = MIN(icslice, ic - icend);
        icslice = ic / icsecs + ((ic % icsecs) > icidx);
        icend = icstart + icslice;

        int ocstart = 0;
        int ocend = 0;
        for (int ocidx = 0; ocidx < ocsecs; ocidx++) {
            ocstart = ocend;
            //ocslice = MIN(ocslice, oc - ocend);
            ocslice = oc / ocsecs + ((oc % ocsecs) > ocidx);
            ocend = ocstart + ocslice;

            // [1, ic, DIV_UP(oc, 32) * kh * kw, 32] => [kh * kw, ic, DIV_UP(oc, 32), 32]
            //dim4 src_shape = {1, icslice, DIV_UP(ocslice, 32) * kh * kw, 32};
            if (ocsecs > 1 && ocstart % 32 > 0 && (ocstart % 32 + ocslice > 32)) {
                int ocslice_part_one = 32 - ocstart % 32;
                int ocslice_part_two = ocslice - ocslice_part_one;
                dim4 src_stride = {32, DIV_UP(oc, 32) * kh * kw * 32, 32 * kh * kw, 1};
                dim4 dst_shape = {kh * kw, icslice, 1, ocslice_part_one};
                tpu_gdma_cpy_S2L(
                    ping ? iaddr_ping : iaddr_pong,
                    weight_global_addr +
                        (icstart * DIV_UP(oc, 32) * kh * kw * 32 +
                         ocstart / 32 * kh * kw * 32 +
                         ocstart % 32) * tpu_data_type_size(dtype),
                    &dst_shape,
                    NULL,
                    &src_stride,
                    dtype);
                dst_shape.h = DIV_UP(ocslice_part_two, 32);
                dst_shape.w = 32;
                tpu_gdma_cpy_S2L(
                    ping ? iaddr_ping + ocslice_part_one * tpu_data_type_size(dtype)
                         : iaddr_pong + ocslice_part_one * tpu_data_type_size(dtype),
                    weight_global_addr +
                        (icstart * DIV_UP(oc, 32) * kh * kw * 32 +
                         (ocstart + ocslice_part_one) / 32 * kh * kw * 32 +
                         (ocstart + ocslice_part_one) % 32) * tpu_data_type_size(dtype),
                    &dst_shape,
                    NULL,
                    &src_stride,
                    dtype);
            } else {
                dim4 dst_shape = {kh * kw, icslice, DIV_UP(ocslice, 32), 32};
                dim4 src_stride = {32,
                                   DIV_UP(oc, 32) * kh * kw * 32,
                                   32 * kh * kw,
                                   1};
                tpu_gdma_cpy_S2L(
                    ping ? iaddr_ping : iaddr_pong,
                    weight_global_addr +
                        (icstart * DIV_UP(oc, 32) * kh * kw * 32 +
                         ocstart / 32 * kh * kw * 32 +
                         ocstart % 32) * tpu_data_type_size(dtype),
                    &dst_shape,
                    NULL,
                    &src_stride,
                    dtype);
            }

            if (parallel_branch) {
                tpu_parallel_end();
            }
            tpu_parallel_start();
            parallel_branch = true;

            scalar_t zero = {.f32 = 0.0f};
            dim4 zero_shape = {kh * kw, ocslice, 1, ALIGN(icslice, 32)};
            tpu_bdc_set_C(
                ping ? oaddr_ping : oaddr_pong,
                tpu_cast(zero, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                &zero_shape,
                NULL,
                dtype);

            // [kh * kw, ic, DIV_UP(oc, 32), 32] can consider as [kh * kw, ic, 1, oc]
            dim4 cw_dst_shape = {kh * kw, ocslice, 1, icslice};
            // [kh * kw, ic, 1, oc] => [kh * kw, oc, 1, ic]
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

            // [kh * kw, oc, 1, ic] can consider as [kh * kw, oc, DIV_UP(ic, 32), 32]
            // [kh * kw, oc, DIV_UP(ic, 32), 32] => [1, oc, DIV_UP(ic, 32) * kh * kw, 32]
            if (icidx > 0 || ocidx > 0) {
                if (icsecs > 1 && last_icstart % 32 > 0 && ((last_icstart % 32 + last_icslice) > 32)) {
                    int last_icslice_part_one = 32 - last_icstart % 32;
                    int last_icslice_part_two = last_icslice - last_icslice_part_one;
                    dim4 last_reordered_shape;
                    last_reordered_shape.n = kh * kw;
                    last_reordered_shape.c = last_ocslice;
                    dim4 last_reordered_stride = {32, DIV_UP(ic, 32) * kh * kw * 32, 32 * kh * kw, 1};
                    for (int idx = 0; idx < 2; idx++) {
                        if (idx == 0) {
                            last_reordered_shape.h = 1;
                            last_reordered_shape.w = last_icslice_part_one;
                            tpu_gdma_cpy_L2S(
                                weight_reordered_global_addr +
                                    (last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                                     last_icstart / 32 * kh * kw * 32 +
                                     last_icstart % 32) * tpu_data_type_size(dtype),
                                ping ? oaddr_pong : oaddr_ping,
                                &last_reordered_shape,
                                &last_reordered_stride,
                                NULL,
                                dtype);
                        } else if (idx == 1) {
                            last_reordered_shape.h = DIV_UP(last_icslice_part_two, 32);
                            last_reordered_shape.w = 32;
                            tpu_gdma_cpy_L2S(
                                weight_reordered_global_addr +
                                    (last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                                     (last_icstart + last_icslice_part_one) / 32 * kh * kw * 32 +
                                     (last_icstart + last_icslice_part_one) % 32) * tpu_data_type_size(dtype),
                                ping ? oaddr_pong + last_icslice_part_one * tpu_data_type_size(dtype)
                                     : oaddr_ping + last_icslice_part_one * tpu_data_type_size(dtype),
                                &last_reordered_shape,
                                &last_reordered_stride,
                                NULL,
                                dtype);
                        }
                    }
                } else {
                    dim4 last_reordered_shape = {kh * kw,
                                                 last_ocslice,
                                                 DIV_UP(last_icslice, 32),
                                                 //32};
                                                 last_icstart % 32 != 0 ? ALIGN(last_icstart, 32) - last_icstart : 32};
                    dim4 last_reordered_stride = {32,
                                                  DIV_UP(ic, 32) * kh * kw * 32,
                                                  32 * kh * kw,
                                                  1};
                    tpu_gdma_cpy_L2S(
                        weight_reordered_global_addr +
                            (last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                             last_icstart / 32 * kh * kw * 32 +
                             last_icstart % 32) * tpu_data_type_size(dtype),
                        ping ? oaddr_pong : oaddr_ping,
                        &last_reordered_shape,
                        &last_reordered_stride,
                        NULL,
                        dtype);
                }
            }
            ping = !ping;
            last_ocstart = ocstart;
            last_ocslice = ocslice;
            last_icstart = icstart;
            last_icslice = icslice;
        }
    }
    tpu_parallel_end();
    if (icsecs > 1 && last_icstart % 32 > 0 && ((last_icstart % 32 + last_icslice) > 32)) {
        int last_icslice_part_one = 32 - last_icstart % 32;
        int last_icslice_part_two = last_icslice - last_icslice_part_one;
        dim4 last_reordered_shape;
        last_reordered_shape.n = kh * kw;
        last_reordered_shape.c = last_ocslice;
        dim4 last_reordered_stride = {32, DIV_UP(ic, 32) * kh * kw * 32, 32 * kh * kw, 1};
        for (int idx = 0; idx < 2; idx++) {
            if (idx == 0) {
                last_reordered_shape.h = 1;
                last_reordered_shape.w = last_icslice_part_one;
                tpu_gdma_cpy_L2S(
                    weight_reordered_global_addr +
                        (last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                         last_icstart / 32 * kh * kw * 32 +
                         last_icstart % 32) * tpu_data_type_size(dtype),
                    ping ? oaddr_pong : oaddr_ping,
                    &last_reordered_shape,
                    &last_reordered_stride,
                    NULL,
                    dtype);
                //last_icstart += last_reordered_shape.w;
            } else if (idx == 1) {
                last_reordered_shape.h = DIV_UP(last_icslice_part_two, 32);
                last_reordered_shape.w = 32;
                tpu_gdma_cpy_L2S(
                    weight_reordered_global_addr +
                        (last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                         (last_icstart + last_icslice_part_one) / 32 * kh * kw * 32 +
                         (last_icstart + last_icslice_part_one) % 32) * tpu_data_type_size(dtype),
                    ping ? oaddr_pong + last_icslice_part_one * tpu_data_type_size(dtype)
                         : oaddr_ping + last_icslice_part_one * tpu_data_type_size(dtype),
                    &last_reordered_shape,
                    &last_reordered_stride,
                    NULL,
                    dtype);
            }
        }
    } else {
        dim4 last_reordered_shape = {kh * kw,
                                     last_ocslice,
                                     DIV_UP(last_icslice, 32),
                                     //31};
                                     last_icstart % 32 != 0 ? ALIGN(last_icstart, 32) - last_icstart : 32};
        dim4 last_reordered_stride = {32,
                                      DIV_UP(ic, 32) * kh * kw * 32,
                                      32 * kh * kw,
                                      1};
        tpu_gdma_cpy_L2S(
            weight_reordered_global_addr +
                (last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                 last_icstart / 32 * kh * kw * 32 +
                 last_icstart % 32) * tpu_data_type_size(dtype),
            ping ? oaddr_pong : oaddr_ping,
            &last_reordered_shape,
            &last_reordered_stride,
            NULL,
            dtype);
    }
}

// grad_out deconv (trans)weight => grad_input
void nodechip_conv_backward_input(
    global_addr_t       grad_out_global_addr,
    global_addr_t       grad_input_global_addr,
    global_addr_t       forward_weight_global_addr,
    // need weight_reordered_addr
    global_addr_t       weight_reordered_global_addr,
    const int           groups,
    const dim4          *grad_input_shape,
    const dim4          *grad_out_shape,
    const dim2          *kernel,
    const dim2          *stride,
    const dim2          *dilation,
    const padding_t     *pad,
    const data_type_t   dtype
 ) {

    const data_type_t idtype = dtype, odtype = dtype;

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

    if (dtype == DT_FP16) {
        nodechip_weight_reorder(forward_weight_global_addr,
                                weight_reordered_global_addr,
                                oc,
                                ic,
                                kh,
                                kw,
                                idtype);
    }

    global_addr_t input_global_addr = grad_out_global_addr;
    global_addr_t weight_global_addr = dtype == DT_FP32 ? forward_weight_global_addr
                                                        : weight_reordered_global_addr;
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
            if (tpu_data_type_size(idtype) == 2) {
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
            } else if (tpu_data_type_size(idtype) == 4) {
                dim4 wshape = {1, ocslice, ic, kh * kw};
                dim4 wshape_local;
                tpu_compact_stride(&wshape_local, 0, &wshape);
                dim4 wshape_global = {.n = oc * ic * kh * kw, .c = kh * kw, .h = oc * kh * kw, .w = 1};
                tpu_gdma_cpy_S2L(
                        waddr,
                        weight_global_addr + (gidx * oc + ocstart) * kh * kw * tpu_data_type_size(idtype),
                        &wshape,
                        &wshape_local,
                        &wshape_global,
                        idtype);
            }

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
                            ping ? iaddr_ping : iaddr_pong,
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

    TPUKERNEL_ASSERT(ALIGN(kh * kw, tpu_eu_num(dtype)) <= 4 * BANK_SIZE);
    TPUKERNEL_ASSERT(DIV_UP(kh * kw, NPU_NUM) * ALIGN(1 * 1, tpu_eu_num(dtype)) <= 4 * BANK_SIZE);

    //(TODO) optimize icslice = 32?
    unsigned int ic_size = DIV_UP(ic, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int ic_size_cwtrans = DIV_UP(kh * kw, NPU_NUM) * tpu_aligned_feature_size(1, ic, dtype);
    unsigned int max_ic_size = MAX(ic_size, ic_size_cwtrans);
    int slice = (unsigned int)(4 * BANK_SIZE) / max_ic_size;
    if (slice == 0) {
        bool valid = false;
        while (!valid) {
            if (icsecs == ic) {
                TPUKERNEL_LOG("grad output transposed need to split h&w!");
                TPUKERNEL_ASSERT(false);
                }
            icsecs++;
            icslice = DIV_UP(ic, icsecs);
            ic_size = DIV_UP(icslice, NPU_NUM) * tpu_aligned_feature_size(kh , kw, dtype);
            ic_size_cwtrans = DIV_UP(kh * kw, NPU_NUM) * tpu_aligned_feature_size(1, icslice, dtype);
            max_ic_size = MAX(ic_size, ic_size_cwtrans);
            valid = (unsigned int)(4 * BANK_SIZE) / max_ic_size > 0;
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
    local_addr_t iaddr_pong = 4 * BANK_SIZE;
    local_addr_t oaddr_ping = 8 * BANK_SIZE;
    local_addr_t oaddr_pong = 12 * BANK_SIZE;

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
            dim4 src_stride = {oc * kh * kw, kh * kw, kh * kw, 1};
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

            scalar_t zero = {.f32 = 0.0f};
            dim4 zero_shape = {ocslice, kh * kw, 1, ALIGN(icslice, 32)};
            tpu_bdc_set_C(
                ping ? oaddr_ping : oaddr_pong,
                tpu_cast(zero, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                &zero_shape,
                NULL,
                dtype);

            dim4 cw_dst_shape = {ocslice, kh * kw, 1, icslice};
            // [oc, ic, 1, kh * kw] => [oc, kh * kw, 1, ic]
            if (cw_dst_shape.w >= cw_dst_shape.c) {
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
                                        32};//last_icslice >= 32 ? 32 : last_icslice};
                dim4 reordered_stride = {DIV_UP(ic, 32) * kh * kw * 32, 32, kh * kw * 32, 1};
                tpu_gdma_cpy_L2S(
                    grad_out_reordered_global_addr +
                        (last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
                         last_icstart / 32 * kh * kw * 32 +
                         last_icstart % 32) * tpu_data_type_size(dtype),
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
                                 32};//last_icslice >= 32 ? 32 : last_icslice};
    dim4 last_reordered_stride = {DIV_UP(ic, 32) * kh * kw * 32, 32, kh * kw * 32, 1};
    tpu_gdma_cpy_L2S(
        grad_out_reordered_global_addr +
            (last_ocstart * DIV_UP(ic, 32) * kh * kw * 32 +
             last_icstart / 32 * kh * kw * 32 +
             last_icstart % 32) * tpu_data_type_size(dtype),
        ping ? oaddr_pong : oaddr_ping,
        &last_reordered_shape,
        &last_reordered_stride,
        NULL,
        dtype);
}

// forward_input op grad_out => grad_weight
// [ic, n, ih, iw] conv [oc, n, oh, ow] => [ic, oc, kh, kw]
void nodechip_conv_backward_weight(
    global_addr_t       forward_input_global_addr,
    global_addr_t       grad_out_global_addr,
    global_addr_t       grad_out_reordered_global_addr,
    global_addr_t       grad_weight_global_addr,
    global_addr_t       grad_bias_global_addr,
    const int           groups,
    const dim4          *forward_input_shape,
    const dim4          *grad_out_shape,
    const dim2          *forward_kernel,
    const dim2          *forward_stride,
    const dim2          *forward_dilation,
    const padding_t     *pad,
    bool                grad_bias_enable,
    data_type_t         dtype
 ) {
    const data_type_t idtype = dtype, odtype = dtype;

    dim2 stride, dilation;
    stride.h = forward_dilation->h;
    stride.w = forward_dilation->w;
    dilation.h = forward_stride->h;
    dilation.w = forward_stride->w;
    //stride.h = forward_stride->h;
    //stride.w = forward_stride->w;
    //dilation.h = forward_dilation->h;
    //dilation.w = forward_dilation->w;

    int n = forward_input_shape->c;
    int ic = forward_input_shape->n;
    int ih = forward_input_shape->h;
    int iw = forward_input_shape->w;
    int oc = grad_out_shape->c;
    int oh = forward_kernel->h;
    int ow = forward_kernel->w;
    int kh = grad_out_shape->h;
    int kw = grad_out_shape->w;
    //int ih_ext = (ih - 1) + pad->top + pad->bottom + 1;
    //int iw_ext = (iw - 1) + pad->left + pad->right + 1;
    //int output_h = (ih_ext - kh_ext) / stride->h + 1;
    //int output_w = (iw_ext - kw_ext) / stride->w + 1;
    //TPUKERNEL_ASSERT(output_h == oh);
    //TPUKERNEL_ASSERT(output_w == ow);

    dim4 ishape = {n, ic, ih, iw};
    dim4 oshape = {n, oc, oh, ow};
    dim2 kernel = {kh, kw};

    grad_weight_secs_info_t secs_info;
    grad_weight_split(
            &ishape,
            &oshape,
            groups,
            &kernel,
            &dilation,
            &stride,
            idtype,
            odtype,
            grad_bias_enable,
            &secs_info);

    bool split_n = secs_info.nsecs != 1;
    bool split_oc = secs_info.ocsecs != 1;
    bool split_kernel = secs_info.khsecs != 1 || secs_info.kwsecs != 1;
    bool split_input = split_n || split_kernel;
    bool split_weight = split_oc || split_kernel;
    bool split_output = split_n || split_oc;

    int nslice = DIV_UP(ishape.n, secs_info.nsecs);
    int ocslice = DIV_UP(oc, secs_info.ocsecs);
    ocslice = secs_info.ocsecs > 1 ? ALIGN(ocslice, NPU_NUM) : ocslice;
    int khslice = DIV_UP(kh, secs_info.khsecs);
    int kwslice = DIV_UP(kw, secs_info.kwsecs);
    int khslice_ext = dilation.h * (khslice - 1) + 1;
    int kwslice_ext = dilation.w * (kwslice - 1) + 1;
    int ihslice = MIN((oh - 1) * stride.h + khslice_ext, ishape.h);
    int iwslice = MIN((ow - 1) * stride.w + kwslice_ext, ishape.w);
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

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
    if (!split_input) iaddr_pong = iaddr_ping;
    local_addr_t waddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
    local_addr_t waddr_pong = ALIGN(waddr_ping + wsize, BANK_SIZE);
    if (!split_weight) waddr_pong = waddr_ping;
    local_addr_t oaddr_ping = ALIGN(waddr_pong + wsize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    if (!split_output) oaddr_pong = oaddr_ping;
    local_addr_t buffer_addr = ALIGN(oaddr_pong + osize, BANK_SIZE);
    if (!split_kernel) {
        buffer_addr = 0;
        TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);
    } else {
        TPUKERNEL_ASSERT(buffer_addr + buffer_size <= (unsigned int)LOCAL_MEM_SIZE);
    }

    local_addr_t grad_bias_addr_ping = grad_bias_enable ?
                                           split_kernel ? ALIGN(buffer_addr + buffer_size, BANK_SIZE) :
                                                          ALIGN(oaddr_pong + osize, BANK_SIZE) : 0;
    local_addr_t grad_bias_addr_pong = grad_bias_enable ? ALIGN(grad_bias_addr_ping + grad_bias_size, BANK_SIZE) : 0;
    if (grad_bias_enable && !split_oc)
        grad_bias_addr_pong = grad_bias_addr_ping;
    if (grad_bias_enable)
        TPUKERNEL_ASSERT(grad_bias_addr_pong + grad_bias_size <= (unsigned int)LOCAL_MEM_SIZE);
    TPUKERNEL_DBG("in ping local addr = 0x%5x, bank id = %d\n", iaddr_ping, iaddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("in pong local addr = 0x%5x, bank id = %d\n", iaddr_pong, iaddr_pong / BANK_SIZE);
    TPUKERNEL_DBG("weight ping local addr = 0x%5x, bank id = %d\n", waddr_ping, waddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("weight pong local addr = 0x%5x, bank id = %d\n", waddr_pong, waddr_pong / BANK_SIZE);
    TPUKERNEL_DBG("out ping local addr = 0x%5x, bank id = %d\n", oaddr_ping, oaddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("out pong local addr = 0x%5x, bank id = %d\n", oaddr_pong, oaddr_pong / BANK_SIZE);
    TPUKERNEL_DBG("buffer local addr = 0x%5x, bank id = %d\n", buffer_addr, buffer_addr / BANK_SIZE);
    if (grad_bias_enable) {
        TPUKERNEL_DBG("grad_bias ping local addr = 0x%5x, bank id = %d\n", grad_bias_addr_ping, grad_bias_addr_ping / BANK_SIZE);
        TPUKERNEL_DBG("grad_bias ping local addr = 0x%5x, bank id = %d\n", grad_bias_addr_pong, grad_bias_addr_pong / BANK_SIZE);
    }

    dim4 istride = {
        n * ih * iw,
        ih * iw,
        iw,
        1};

    //dim4 ostride = {
    //    oshape.c * oshape.h * oshape.w,
    //    oshape.h * oshape.w,
    //    oshape.w,
    //    1};

    dim4 ostride_nc_trans = {
        oshape.n * oshape.h * oshape.w,
        oshape.h * oshape.w,
        oshape.w,
        1};

    dim4 wshape = {
        1,
        oc,
        (idtype == DT_FP32 ? ic : DIV_UP(ic, 32)) * kh * kw,
        idtype == DT_FP32 ? 1 : 32};
    dim4 wstride;
    tpu_compact_stride(&wstride, 0, &wshape);

    if (dtype == DT_FP16) {
        nodechip_grad_output_reorder(grad_out_global_addr,
                                     grad_out_reordered_global_addr,
                                     ic,
                                     oc,
                                     kh,
                                     kw,
                                     idtype);
    }

    global_addr_t input_global_addr = forward_input_global_addr;
    global_addr_t weight_global_addr = dtype == DT_FP32 ? grad_out_global_addr
                                                        : grad_out_reordered_global_addr;
    global_addr_t output_global_addr = grad_weight_global_addr;

    int last_nslice = 0;
    int last_nstart = 0;
    int last_ocslice = 0;
    int last_ocstart = 0;

    bool ping_in = true;
    bool ping_out = true;
    bool load_input = true;
    bool load_weight = true;
    // 1st loop: groups
    for (int gidx = 0; gidx < groups; gidx++) {

        bool parallel_branch = false;
        int nstart = 0;
        int nend = 0;
        // 2nd loop: split N dim
        for (int nidx = 0; nidx < secs_info.nsecs; nidx++) {
            nstart = nend;
            nslice = ishape.n / secs_info.nsecs + ((ishape.n % secs_info.nsecs) > nidx);
            nend = nstart + nslice;

            if (nidx != 0 && !split_oc && !split_kernel) load_weight = false;

            int ocstart = 0;
            int ocend = 0;
            // 3nd loop: split OC dim
            for (int ocidx = 0; ocidx < secs_info.ocsecs; ocidx++) {
                // oc boundary: [ocstart, ocend)
                ocstart = ocend;
                ocslice = DIV_UP(oc, secs_info.ocsecs);
                ocslice = MIN(ALIGN(ocslice, NPU_NUM), oc - ocend);
                ocend = ocstart + ocslice;

                if (ocidx != 0 && !split_kernel) load_input = false;

                int khstart = 0;
                int khend = 0;
                // 4nd loop: split KH dim
                for (int khidx = 0; khidx < secs_info.khsecs; khidx++) {
                    khstart = khend;
                    khslice = kh / secs_info.khsecs + ((kh % secs_info.khsecs) > khidx);
                    khend = khstart + khslice;

                    int kwstart = 0;
                    int kwend = 0;
                    // 5nd loop: split KW dim
                    for (int kwidx = 0; kwidx < secs_info.kwsecs; kwidx++) {
                        kwstart = kwend;
                        kwslice = kw / secs_info.kwsecs + ((kw % secs_info.kwsecs) > kwidx);
                        kwend = kwstart + kwslice;

                        dim4 wslice_shape = {
                            1,
                            ocslice,
                            (idtype == DT_FP32 ? ic : DIV_UP(ic, 32)) * khslice * kwslice,
                            idtype == DT_FP32 ? 1 : 32};

                        if (load_weight) {
                            if (tpu_data_type_size(idtype) == 2) {
                                dim4 wslice_stride;
                                tpu_compact_stride(&wslice_stride, 0, &wslice_shape);
                                if (idtype == DT_FP16 && ic > 32 && split_kernel) {
                                    for (int icidx = 0; icidx < DIV_UP(ic, 32); icidx++) {
                                        dim4 per32ic_wslice_shape = {1, ocslice, khslice * kwslice, 32};
                                        tpu_gdma_cpy_S2L(
                                            ping_in ? waddr_ping + icidx * khslice * kwslice *32 * tpu_data_type_size(idtype)
                                                    : waddr_pong + icidx * khslice * kwslice *32 * tpu_data_type_size(idtype),
                                            weight_global_addr +
                                                ((gidx * oc + ocstart) * wstride.c +
                                                 (icidx * kh * kw + khstart * kw + kwstart) * 32) * tpu_data_type_size(idtype),
                                            &per32ic_wslice_shape,
                                            &wslice_stride,
                                            &wstride,
                                            idtype);
                                    }
                                } else {
                                    tpu_gdma_cpy_S2L(
                                        ping_in ? waddr_ping : waddr_pong,
                                        weight_global_addr +
                                            ((gidx * oc + ocstart) * wstride.c +
                                             (khstart * kw + kwstart) * (idtype == DT_FP32 ? 1 : 32)) * tpu_data_type_size(idtype),
                                        &wslice_shape,
                                        &wslice_stride,
                                        &wstride,
                                        idtype);
                                }
                            } else if (tpu_data_type_size(idtype) == 4) {
                                dim4 wslice_shape_fp32 = {ic, ocslice, khslice, kwslice};
                                dim4 wslice_stride_global = {.n = oc * kh * kw, .c = kh * kw, .h = kw, .w = 1};
                                dim4 wslice_stride_local = {.n = khslice * kwslice, .c = ic * khslice * kwslice, .h = kwslice, .w = 1};
                                tpu_gdma_cpy_S2L(
                                    ping_in ? waddr_ping : waddr_pong,
                                    weight_global_addr +
                                        ((gidx * oc + ocstart) * wslice_stride_global.c +
                                        khstart * wslice_stride_global.h +
                                        kwstart) * tpu_data_type_size(idtype),
                                    &wslice_shape_fp32,
                                    &wslice_stride_local,
                                    &wslice_stride_global,
                                    idtype);
                            }
                        }
                        load_weight = true;

                        khslice_ext = dilation.h * (khslice - 1) + 1;
                        kwslice_ext = dilation.h * (kwslice - 1) + 1;

                        padding_t pad_slice;
                        int ihstart = - pad->top + dilation.h * khstart;
                        ihslice = (oh - 1) * stride.h + khslice_ext;
                        int ihend = ihstart + ihslice;
                        pad_slice.top = ihstart < 0 ? -ihstart : 0;
                        pad_slice.bottom = ihend > ishape.h ? ihend - ishape.h : 0;
                        ihstart = ihstart < 0 ? 0 : ihstart;
                        ihend = ihend > ishape.h ? ishape.h : ihend;
                        ihslice = ihend - ihstart;

                        int iwstart = -pad->left + dilation.w * kwstart;
                        iwslice = (ow - 1) * stride.w + kwslice_ext;
                        int iwend = iwstart + iwslice;
                        pad_slice.left = iwstart < 0 ? -iwstart : 0;
                        pad_slice.right = iwend > ishape.w ? iwend - ishape.w : 0;
                        iwstart = iwstart < 0 ? 0 : iwstart;
                        iwend = iwend > ishape.w ? ishape.w : iwend;
                        iwslice = iwend - iwstart;

                        dim4 islice_shape = {nslice, ic, ihslice, iwslice};
                        dim4 oslice_shape = {nslice, ocslice, oh, ow};

                        dim4 input_shape = {nslice, ic, ihslice, iwslice};
                        if (load_input) {
                            tpu_gdma_cpy_nc_trans_S2L(
                                ping_in ? iaddr_ping : iaddr_pong,
                                input_global_addr +
                                    (nstart * istride.c +
                                     gidx * ic * istride.n +
                                     ihstart * istride.h +
                                     iwstart) * tpu_data_type_size(idtype),
                                &input_shape,
                                NULL,
                                &istride,
                                idtype);
                        }
                        load_input = true;

                        if (parallel_branch) {
                            tpu_parallel_end();
                        }
                        tpu_parallel_start();
                        parallel_branch = true;

                        dim2 kernel_slice;
                        kernel_slice.h = khslice;
                        kernel_slice.w = kwslice;
                        tpu_bdc_fp_conv2d(
                                khidx == 0 && kwidx == 0 ?
                                    ping_out ? oaddr_ping : oaddr_pong :
                                    buffer_addr,
                                ping_in ? iaddr_ping : iaddr_pong,
                                ping_in ? waddr_ping : waddr_pong,
                                0,
                                &islice_shape,
                                NULL,
                                oslice_shape.c,
                                &kernel_slice,
                                &pad_slice,
                                &stride,
                                &dilation,
                                odtype,
                                idtype,
                                false,
                                false);
                        if (khidx > 0 || kwidx > 0)
                            tpu_bdc_fp_add(
                                ping_out ? oaddr_ping : oaddr_pong,
                                ping_out ? oaddr_ping : oaddr_pong,
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
                                ping_in ? waddr_ping : waddr_pong,
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
                                    ping_out ? grad_bias_addr_ping : grad_bias_addr_pong :
                                    ping_in ? waddr_ping : waddr_pong,//buffer_addr,
                                buffer_addr,
                                &grad_bias_shape,
                                &pooling_kernel,
                                &pooling_padding,
                                &pooling_stride,
                                &pooling_dilation,
                                idtype,
                                tpu_cast(scale, idtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
                            grad_bias_shape.w = 1;
                            if (khidx != 0 || kwidx != 0)
                                tpu_bdc_fp_add(
                                    ping_out ? grad_bias_addr_ping : grad_bias_addr_pong,
                                    ping_out ? grad_bias_addr_ping : grad_bias_addr_pong,
                                    ping_in ? waddr_ping : waddr_pong,//buffer_addr,
                                    &grad_bias_shape,
                                    NULL,
                                    NULL,
                                    NULL,
                                    idtype);
                        }
                        ping_in = !ping_in;
                    }
                }
                if (nidx > 0 || ocidx > 0) {
                    /*
                    dim4 last_oslice_shape = {last_nslice, last_ocslice, oh, ow};
                    tpu_gdma_cpy_L2S(
                        output_global_addr +
                            (last_nstart * ostride.n +
                             (gidx * oc + last_ocstart) * ostride.c) * tpu_data_type_size(odtype),
                        ping_out ? oaddr_pong : oaddr_ping,
                        &last_oslice_shape,
                        &ostride,
                        NULL,
                        odtype);
                    */
                    dim4 last_oslice_shape = {last_ocslice, last_nslice, oh, ow};
                    tpu_gdma_cpy_nc_trans_L2S(
                        output_global_addr +
                            (last_nstart * ostride_nc_trans.c +
                             (gidx * oc + last_ocstart) * ostride_nc_trans.n) * tpu_data_type_size(odtype),
                        ping_out ? oaddr_pong : oaddr_ping,
                        &last_oslice_shape,
                        &ostride_nc_trans,//&ostride,
                        NULL,
                        odtype);
                    if (grad_bias_enable && nidx == 0 && ocidx > 0) {
                        bool do_cw_trans = last_ocslice > 1;
                        dim4 move_bias_shape = {1,
                                                do_cw_trans ? 1 : last_ocslice,
                                                1,
                                                do_cw_trans ? last_ocslice : 1};
                        if (do_cw_trans) {
                            tpu_bdc_cw_trans(
                                buffer_addr,
                                ping_out ? grad_bias_addr_pong : grad_bias_addr_ping,
                                &move_bias_shape,
                                idtype);
                        }
                        tpu_gdma_cpy_L2S(
                            grad_bias_global_addr +
                                last_ocstart * tpu_data_type_size(idtype),
                            do_cw_trans ? buffer_addr :
                                          ping_out ? grad_bias_addr_pong :
                                                     grad_bias_addr_ping,
                            &move_bias_shape,
                            NULL,
                            NULL,
                            idtype);
                    }
                }
                ping_out = !ping_out;
                // save current info used for moving output to global memory next loop
                last_nstart = nstart;
                last_nslice = nslice;
                last_ocstart = ocstart;
                last_ocslice = ocslice;
            }
        }
        tpu_parallel_end();
        // move last output to global memory
        /*
        dim4 last_oslice_shape = {last_nslice, last_ocslice, oh, ow};
        tpu_gdma_cpy_L2S(
            output_global_addr +
                (last_nstart * ostride.n +
                 ((groups - 1) * oc + last_ocstart) * ostride.c)
                * tpu_data_type_size(odtype),
            ping_out ? oaddr_pong : oaddr_ping,
            &last_oslice_shape,
            &ostride,
            NULL,
            odtype);
        */
        dim4 last_oslice_shape = {last_ocslice, last_nslice, oh, ow};
        tpu_gdma_cpy_nc_trans_L2S(
            output_global_addr +
                (last_nstart * ostride_nc_trans.c +
                 (gidx * oc + last_ocstart) * ostride_nc_trans.n) * tpu_data_type_size(odtype),
            ping_out ? oaddr_pong : oaddr_ping,
            &last_oslice_shape,
            &ostride_nc_trans,
            NULL,
            odtype);
        if (grad_bias_enable) {
            bool do_cw_trans = last_ocslice > 1;
            dim4 move_bias_shape = {1,
                                    do_cw_trans ? 1 : last_ocslice,
                                    1,
                                    do_cw_trans ? last_ocslice : 1};
            if (do_cw_trans) {
                tpu_bdc_cw_trans(
                    buffer_addr,
                    ping_out ? grad_bias_addr_pong : grad_bias_addr_ping,
                    &move_bias_shape,
                    idtype);
            }
            tpu_gdma_cpy_L2S(
                grad_bias_global_addr +
                    last_ocstart * tpu_data_type_size(idtype),
                do_cw_trans ? buffer_addr :
                              ping_out ? grad_bias_addr_pong :
                                         grad_bias_addr_ping,
                &move_bias_shape,
                NULL,
                NULL,
                idtype);
        }
    }
}

typedef struct{
    int ocsecs;
    int icsecs;
    int ohsecs;
    int owsecs;
} depthwise_secs_info_t;

static inline bool depthwise_split_kernel(
    int         *slice,
    int         *secs,
    int         ih_or_iw,
    int         kh_or_kw,
    int         oh_or_ow,
    int         stride,
    int         dilation,
    int         other_size,
    data_type_t dtype) {

    bool valid = false;
    int slice_new = *slice;
    do {
        if (slice_new == 1) {
            *slice = slice_new;
            --(*secs);
            return false;
        }
        slice_new = DIV_UP(*slice, *secs);

        int oslice_ext = dilation * (slice_new - 1) + 1;
        int islice = (kh_or_kw - 1) * stride + oslice_ext;

        unsigned int isize = 1 * tpu_aligned_feature_size(islice, ih_or_iw, dtype);
        unsigned int wsize = 1 * slice_new * oh_or_ow * tpu_data_type_size(dtype);
        unsigned int total_size = ALIGN(isize, BANK_SIZE) * 2 +
                                  ALIGN(wsize, BANK_SIZE) * 2 +
                                  other_size;
        valid = total_size <= (unsigned int)LOCAL_MEM_SIZE;

        ++(*secs);
    } while(!valid);
    *slice = slice_new;
    --(*secs);
    return true;
}

void depthwise_data_split(
    const int               n,
    const int               oc,
    const int               ic,
    const int               ih,
    const int               iw,
    const int               oh,
    const int               ow,
    const int               kh,
    const int               kw,
    const dim2             *stride,
    const dim2             *dilation,
    data_type_t             dtype,
    depthwise_secs_info_t  *secs_info
) {

    const int nslice = 1;
    secs_info->ocsecs = 1;
    secs_info->icsecs = 1;
    secs_info->ohsecs = 1;
    secs_info->owsecs = 1;

    int ocslice = oc;
    int icslice = ic;
    int ohslice = oh;
    int owslice = ow;
    int ihslice = ih;
    int iwslice = iw;

    icslice = MIN(ic, NPU_NUM);
    secs_info->icsecs = DIV_UP(ic, icslice);

    // depthwise operators:
    // input: [1, icslice, ihslice, iwslice], loop in n;
    // weight: [1, 1 => icslice, ohslice, owslice], first loop in oc, then loop in n;
    // output: [ocslice, icslice, kh, kw]

    // data_split order:
    // split oc => split ic => split oh => split ow, nslice is always 1 to add_result

    unsigned int isize = nslice * DIV_UP(icslice, NPU_NUM) *
                         tpu_aligned_feature_size(ihslice, iwslice, dtype);
    unsigned int wsize = nslice * DIV_UP(icslice, NPU_NUM) *
                         ohslice * owslice * tpu_data_type_size(dtype);
    unsigned int osize = ocslice * DIV_UP(icslice, NPU_NUM) *
                         tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int buffer_size = n > 1 ? osize : 0;

    unsigned int total_size = ALIGN(isize, BANK_SIZE) * 2 +
                              ALIGN(wsize, BANK_SIZE) * 2 +
                              ALIGN(osize, BANK_SIZE) * 2 +
                              ALIGN(buffer_size, BANK_SIZE);
    if (total_size <= (unsigned int)LOCAL_MEM_SIZE) return;

    unsigned int per_oc_isize = isize;
    unsigned int per_oc_wsize = wsize;
    unsigned int per_oc_osize = DIV_UP(icslice, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int per_oc_buffer_size = n > 1 ? per_oc_osize : 0;
    unsigned int per_oc_total_size = ALIGN(per_oc_isize, BANK_SIZE) * 2 +
                                     ALIGN(per_oc_wsize, BANK_SIZE) * 2 +
                                     ALIGN(per_oc_osize, BANK_SIZE) * 2 +
                                     ALIGN(per_oc_buffer_size, BANK_SIZE);
    if (per_oc_total_size <= (unsigned int)LOCAL_MEM_SIZE) {
        ocslice = ((unsigned int)LOCAL_MEM_SIZE -
                   ALIGN(per_oc_isize, BANK_SIZE) * 2 -
                   ALIGN(per_oc_wsize, BANK_SIZE) * 2) / 3 / per_oc_osize;
        secs_info->ocsecs = DIV_UP(oc, ocslice);
        return;
    }

    ocslice = 1;
    secs_info->ocsecs = oc;

    //split kernel
    bool valid = false;
    unsigned int other_size = ALIGN(per_oc_osize, BANK_SIZE) * 2 + ALIGN(per_oc_buffer_size, BANK_SIZE);
    TPUKERNEL_ASSERT(other_size <= (unsigned int)LOCAL_MEM_SIZE);

    //split oh first, then split ow
    valid = depthwise_split_kernel(
            &ohslice,
            &(secs_info->ohsecs),
            iw,
            kh,
            ow,
            stride->h,
            dilation->h,
            other_size,
            dtype);
    if (!valid) {
        int ohslice_ext = dilation->h * (ohslice - 1) + 1;
        int ihslice = (kh - 1) * stride->h + ohslice_ext;
        valid = depthwise_split_kernel(
                &owslice,
                &(secs_info->owsecs),
                ihslice,
                kw,
                ohslice,
                stride->w,
                dilation->w,
                other_size,
                dtype);
    }
    TPUKERNEL_ASSERT(valid);
}

void nodechip_conv_backward_weight_depthwise(
    global_addr_t       forward_input_global_addr,
    global_addr_t       grad_out_global_addr,
    global_addr_t       grad_weight_global_addr,
    const int           groups,
    const dim4          *forward_input_shape,
    const dim4          *grad_out_shape,
    const dim2          *forward_kernel,
    const dim2          *forward_stride,
    const dim2          *forward_dilation,
    const padding_t     *pad,
    data_type_t         dtype
 ) {

    int n = forward_input_shape->n;
    int ic = forward_input_shape->c;
    int ih = forward_input_shape->h;
    int iw = forward_input_shape->w;
    int oc = grad_out_shape->c;
    int oh = grad_out_shape->h;
    int ow = grad_out_shape->w;

    int kh = forward_kernel->h;
    int kw = forward_kernel->w;

    dim2 stride, dilation;
    stride.h = forward_dilation->h;
    stride.w = forward_dilation->w;
    dilation.h = forward_stride->h;
    dilation.w = forward_stride->w;
    //stride.h = forward_stride->h;
    //stride.w = forward_stride->w;
    //dilation.h = forward_dilation->h;
    //dilation.w = forward_dilation->w;

    dim4 ishape = {n, ic, ih, iw};
    //dim4 wshape = {n, oc, oh, ow};
    //dim4 oshape = {oc, ic, kh, kw};

    dim4 istride = {ic * ih * iw, ih * iw, iw, 1};
    dim4 wstride = {oc * oh * ow, oh * ow, ow, 1};
    dim4 ostride = {ic * kh * kw, kh * kw, kw, 1};

    depthwise_secs_info_t secs_info;
    depthwise_data_split(
        n, oc, ic,
        ih, iw, oh, ow,
        kh, kw,
        &stride,
        &dilation,
        dtype,
        &secs_info);

    //[n, ic', ih, iw] depthwise (oc' * [n, 1, oh, ow]) => [oc', ic', kh, kw]
    //[n, ic', ih, iw] depthwise ()

    // n = 1 to do add_result
    int nslice = 1;
    int ocslice = DIV_UP(oc, secs_info.ocsecs);
    int icslice = DIV_UP(ic, secs_info.icsecs);
    icslice = secs_info.icsecs > 1 ? ALIGN(icslice, NPU_NUM) : icslice;
    int ohslice = DIV_UP(oh, secs_info.ohsecs);
    int owslice = DIV_UP(ow, secs_info.owsecs);
    int ohslice_ext = dilation.h * (ohslice - 1) + 1;
    int owslice_ext = dilation.w * (owslice - 1) + 1;
    int ihslice = MIN((kh - 1) * stride.h + ohslice_ext, ih);
    int iwslice = MIN((kw - 1) * stride.w + owslice_ext, iw);

    // input: [1, icslice, ihslice, iwslice]
    // weight: [1, 1 => icslice, ohslice, owslice]
    unsigned int isize = nslice * DIV_UP(icslice, NPU_NUM) *
                         tpu_aligned_feature_size(ihslice, iwslice, dtype);
    unsigned int wsize = nslice * DIV_UP(icslice, NPU_NUM) *
                         ohslice * owslice * tpu_data_type_size(dtype);
    unsigned int osize = ocslice * DIV_UP(icslice, NPU_NUM) *
                         tpu_aligned_feature_size(kh, kw, dtype);

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
    local_addr_t waddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
    local_addr_t waddr_pong = ALIGN(waddr_ping + wsize, BANK_SIZE);
    local_addr_t oaddr_ping = ALIGN(waddr_pong + wsize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);

    local_addr_t buffer_addr = 0;
    if (n > 1 || secs_info.ohsecs > 1 || secs_info.owsecs > 1) {
        unsigned int buffer_size = osize;
        buffer_addr = ALIGN(oaddr_pong + osize, BANK_SIZE);
        TPUKERNEL_ASSERT(buffer_addr + buffer_size <= (unsigned int)LOCAL_MEM_SIZE);
    }

    global_addr_t input_global_addr = forward_input_global_addr;
    global_addr_t weight_global_addr = grad_out_global_addr;
    global_addr_t output_global_addr = grad_weight_global_addr;

    bool ping_in = true;
    bool ping_weight = true;
    bool ping_out = true;
    bool parallel_branch = false;

    int last_ocslice = 0;
    int last_ocstart = 0;
    int last_icslice = 0;
    int last_icstart = 0;

    int ocstart = 0;
    int ocend = 0;
    for (int oc_idx = 0; oc_idx < secs_info.ocsecs; oc_idx++) {
        ocstart = ocend;
        ocslice = oc / secs_info.ocsecs + ((oc % secs_info.ocsecs) > oc_idx);
        ocend = ocstart + ocslice;

        int icstart = 0;
        int icend = 0;
        for (int ic_idx = 0; ic_idx < secs_info.icsecs; ic_idx++) {
            icstart = icend;
            icslice = DIV_UP(ic, secs_info.icsecs);
            icslice = MIN(ALIGN(icslice, NPU_NUM), ic - icend);
            icend = icstart + icslice;

            dim4 oslice_shape = {ocslice, icslice, kh, kw};
            dim4 oslice_stride;
            tpu_aligned_stride(&oslice_stride, 0, &oslice_shape, dtype);

            for (int n_idx = 0; n_idx < n; n_idx++) {

                int ohstart = 0;
                int ohend = 0;
                for (int oh_idx = 0; oh_idx < secs_info.ohsecs; oh_idx++) {
                    ohstart = ohend;
                    ohslice = oh / secs_info.ohsecs + ((oh % secs_info.ohsecs) > oh_idx);
                    ohend = ohstart + ohslice;

                    int owstart = 0;
                    int owend = 0;
                    for (int ow_idx = 0; ow_idx < secs_info.owsecs; ow_idx++) {
                        owstart = owend;
                        owslice = ow / secs_info.owsecs + ((ow % secs_info.owsecs) > ow_idx);
                        owend = owstart + owslice;

                        bool add_result = (n_idx != 0 || oh_idx != 0 || ow_idx != 0);

                        for (int ocslice_idx = 0; ocslice_idx < ocslice; ocslice_idx++) {

                            ohslice_ext = dilation.h * (ohslice - 1) + 1;
                            owslice_ext = dilation.h * (owslice - 1) + 1;

                            padding_t pad_slice;
                            int ihstart = - pad->top + dilation.h * ohstart;
                            ihslice = (kh - 1) * stride.h + ohslice_ext;
                            int ihend = ihstart + ihslice;
                            pad_slice.top = ihstart < 0 ? -ihstart : 0;
                            pad_slice.bottom = ihend > ishape.h ? ihend - ishape.h : 0;
                            ihstart = ihstart < 0 ? 0 : ihstart;
                            ihend = ihend > ishape.h ? ishape.h : ihend;
                            ihslice = ihend - ihstart;

                            int iwstart = -pad->left + dilation.w * owstart;
                            iwslice = (kw - 1) * stride.w + owslice_ext;
                            int iwend = iwstart + iwslice;
                            pad_slice.left = iwstart < 0 ? -iwstart : 0;
                            pad_slice.right = iwend > ishape.w ? iwend - ishape.w : 0;
                            iwstart = iwstart < 0 ? 0 : iwstart;
                            iwend = iwend > ishape.w ? ishape.w : iwend;
                            iwslice = iwend - iwstart;

                            // load forward_input [1, ic', ih', iw']
                            dim4 islice_shape = {1, icslice, ihslice, iwslice};
                            if (ocslice_idx == 0) {
                                tpu_gdma_cpy_S2L(
                                    ping_in ? iaddr_ping : iaddr_pong,
                                    input_global_addr +
                                        (n_idx * istride.n +
                                         icstart * istride.c +
                                         ihstart * istride.h +
                                         iwstart) * tpu_data_type_size(dtype),
                                    &islice_shape,
                                    NULL,
                                    &istride,
                                    dtype);
                            }

                            // load grad_out channel bcast  [1, 1, oh', ow'] => [1, ic', oh', ow']
                            //dim4 wslice_shape = {1, 1, ohslice, owslice};
                            //dim4 wslice_stride;
                            //tpu_compact_stride(&wslice_stride, 0, &wslice_shape);
                            dim4 wslice_bcast_shape = {1, icslice, ohslice, owslice};
                            dim4 wslice_bcast_stride;
                            tpu_compact_stride(&wslice_bcast_stride, 0, &wslice_bcast_shape);
                            dim4 wstride_bcast_global = {0, 0, ow, 1};
                            tpu_gdma_channel_bcast_S2L(
                                ping_weight ? waddr_ping : waddr_pong,
                                weight_global_addr +
                                    (n_idx * wstride.n +
                                     (ocstart + ocslice_idx) * wstride.c +
                                     ohstart * wstride.h +
                                     owstart) * tpu_data_type_size(dtype),
                                &wslice_bcast_shape,
                                &wslice_bcast_stride,
                                &wstride_bcast_global,
                                dtype);

                            if (parallel_branch) {
                                tpu_parallel_end();
                            };
                            tpu_parallel_start();
                            parallel_branch = true;

                            // per ocslice_idx: [nidx, ic', ih, iw] depthwise [nidx, ic', oh, ow], add_result if nidx != 0
                            // all ocslice_idx: => [oc', ic', kh, kw]
                            dim2 kernel_slice = {ohslice, owslice};
                            tpu_bdc_fp_depthwise2d(
                                add_result ? buffer_addr + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype):
                                             ping_out ? oaddr_ping + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype):
                                                        oaddr_pong + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype),
                                ping_in ? iaddr_ping : iaddr_pong,
                                ping_weight ? waddr_ping : waddr_pong,
                                0,
                                &islice_shape,
                                &kernel_slice,
                                &pad_slice,
                                &stride,
                                &dilation,
                                dtype,
                                false);
                           ping_weight = !ping_weight;
                        }
                        if (add_result) {
                              tpu_bdc_fp_add(
                                  ping_out ? oaddr_ping : oaddr_pong,
                                  ping_out ? oaddr_ping : oaddr_pong,
                                  buffer_addr,
                                  &oslice_shape,
                                  NULL,
                                  NULL,
                                  NULL,
                                  dtype);
                        }
                        ping_in = !ping_in;
                    }
                }
            }
            if (oc_idx > 0 || ic_idx > 0) {
                // gdma L2S [oc', ic', kh, kw]
                dim4 last_oslice_shape = {last_ocslice, last_icslice, kh, kw};
                tpu_gdma_cpy_L2S(
                    output_global_addr +
                        (last_ocstart * ostride.n +
                         last_icstart * ostride.c) * tpu_data_type_size(dtype),
                    ping_out ? oaddr_pong : oaddr_ping,
                    &last_oslice_shape,
                    &ostride,
                    NULL,
                    dtype);
            }
            ping_out = !ping_out;
            last_ocstart = ocstart;
            last_ocslice = ocslice;
            last_icstart = icstart;
            last_icslice = icslice;
        }
    }
    tpu_parallel_end();
    //move lastest slice L2S [ocslice_lastest, icslice_lastest, kh, kw]
    dim4 last_oslice_shape = {last_ocslice, last_icslice, kh, kw};
    tpu_gdma_cpy_L2S(
        output_global_addr +
            (last_ocstart * ostride.n +
             last_icstart * ostride.c) * tpu_data_type_size(dtype),
        ping_out ? oaddr_pong : oaddr_ping,
        &last_oslice_shape,
        &ostride,
        NULL,
        dtype);
}

typedef struct{
    int nsecs;
    int ocsecs;
    int icsecs;
} grad_weight_1x1_secs_info_t;

void grad_weight_1x1_data_split(
    const int                    n,
    const int                    oc,
    const int                    ic,
    const int                    ih,
    const int                    iw,
    const int                    oh,
    const int                    ow,
    const int                    kh,
    const int                    kw,
    data_type_t                  dtype,
    grad_weight_1x1_secs_info_t  *secs_info
) {

    int nslice = 1;
    secs_info->nsecs = n;

    secs_info->ocsecs = 1;
    secs_info->icsecs = 1;

    int ocslice = oc;
    int icslice = ic;

    icslice = MIN(ic, NPU_NUM);
    secs_info->icsecs = DIV_UP(ic, icslice);

    unsigned int isize = nslice * DIV_UP(icslice, NPU_NUM) *
                         tpu_aligned_feature_size(ih, iw, dtype);
    unsigned int wsize = nslice * DIV_UP(icslice, NPU_NUM) *
                         oh * ow * tpu_data_type_size(dtype);
    unsigned int osize = ocslice * DIV_UP(icslice, NPU_NUM) *
                         tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int buffer_size = n > 1 ? osize : 0;

    unsigned int total_size = ALIGN(isize, BANK_SIZE) * 2 +
                              ALIGN(wsize, BANK_SIZE) * 2 +
                              ALIGN(osize, BANK_SIZE) * 2 +
                              ALIGN(buffer_size, BANK_SIZE);
    //if (total_size <= (unsigned int)LOCAL_MEM_SIZE) return;
    if (total_size <= (unsigned int)LOCAL_MEM_SIZE) {
        nslice = n;
        secs_info->nsecs = 1;
        bool first_time = true;
        while(1) {
            if (!first_time) {
                secs_info->nsecs++;
                nslice = DIV_UP(n, secs_info->nsecs);
            }
            unsigned int other_size = ALIGN(osize, BANK_SIZE) * 2;
            if (secs_info->nsecs > 1) other_size += ALIGN(osize, BANK_SIZE);
            unsigned int some_n_isize = nslice * DIV_UP(icslice, NPU_NUM) *
                                        tpu_aligned_feature_size(ih, iw, dtype);
            unsigned int some_n_wsize = nslice * DIV_UP(icslice, NPU_NUM) *
                                        oh * ow * tpu_data_type_size(dtype);
            unsigned int some_n_total_size = ALIGN(some_n_isize, BANK_SIZE) * 2 +
                                             ALIGN(some_n_wsize, BANK_SIZE) * 2 +
                                             other_size;
            if (some_n_total_size <= (unsigned int)LOCAL_MEM_SIZE) break;
            first_time = false;
        }
        return;
    }

    unsigned int per_oc_isize = isize;
    unsigned int per_oc_wsize = wsize;
    unsigned int per_oc_osize = DIV_UP(icslice, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int per_oc_buffer_size = n > 1 ? per_oc_osize : 0;
    unsigned int per_oc_total_size = ALIGN(per_oc_isize, BANK_SIZE) * 2 +
                                     ALIGN(per_oc_wsize, BANK_SIZE) * 2 +
                                     ALIGN(per_oc_osize, BANK_SIZE) * 2 +
                                     ALIGN(per_oc_buffer_size, BANK_SIZE);

    TPUKERNEL_ASSERT(per_oc_total_size <= (unsigned int)LOCAL_MEM_SIZE);
    //TODO
    // if nsecs == 1, can no need buffer, ocslice can larger
    ocslice = ((unsigned int)LOCAL_MEM_SIZE -
               ALIGN(per_oc_isize, BANK_SIZE) * 2 -
               ALIGN(per_oc_wsize, BANK_SIZE) * 2) / 3 / per_oc_osize;
    secs_info->ocsecs = DIV_UP(oc, ocslice);

    nslice = n;
    secs_info->nsecs = 1;
    bool first_time = true;
    while(1) {
        if (!first_time) {
                secs_info->nsecs++;
                nslice = DIV_UP(n, secs_info->nsecs);
        }
        unsigned int some_oc_osize = ocslice * DIV_UP(icslice, NPU_NUM) *
                                     tpu_aligned_feature_size(kh, kw, dtype);
        unsigned int other_size = ALIGN(some_oc_osize, BANK_SIZE) * 2;
        if (secs_info->nsecs > 1) other_size += ALIGN(some_oc_osize, BANK_SIZE);
        unsigned int some_n_isize = nslice * DIV_UP(icslice, NPU_NUM) *
                                    tpu_aligned_feature_size(ih, iw, dtype);
        unsigned int some_n_wsize = nslice * DIV_UP(icslice, NPU_NUM) *
                                    oh * ow * tpu_data_type_size(dtype);
        unsigned int some_n_total_size = ALIGN(some_n_isize, BANK_SIZE) * 2 +
                                         ALIGN(some_n_wsize, BANK_SIZE) * 2 +
                                         other_size;
        if (some_n_total_size <= (unsigned int)LOCAL_MEM_SIZE) break;
        first_time = false;
    }

    //(TODO) split kernel
}

void nodechip_conv_backward_weight_1x1(
    global_addr_t       forward_input_global_addr,
    global_addr_t       grad_out_global_addr,
    global_addr_t       grad_weight_global_addr,
    const int           groups,
    const dim4          *forward_input_shape,
    const dim4          *grad_out_shape,
    const dim2          *forward_kernel,
    const dim2          *forward_stride,
    const dim2          *forward_dilation,
    const padding_t     *pad,
    data_type_t         dtype
 ) {

    int n = forward_input_shape->n;
    int ic = forward_input_shape->c;
    int ih = forward_input_shape->h;
    int iw = forward_input_shape->w;
    int oc = grad_out_shape->c;
    int oh = grad_out_shape->h;
    int ow = grad_out_shape->w;

    int kh = forward_kernel->h;
    int kw = forward_kernel->w;
    TPUKERNEL_ASSERT(kh == 1);
    TPUKERNEL_ASSERT(kw == 1);

    TPUKERNEL_ASSERT(pad->top == 0);
    TPUKERNEL_ASSERT(pad->bottom == 0);
    TPUKERNEL_ASSERT(pad->left == 0);
    TPUKERNEL_ASSERT(pad->right == 0);

    TPUKERNEL_ASSERT(forward_dilation->h == 1);
    TPUKERNEL_ASSERT(forward_dilation->w == 1);

    dim2 stride = {forward_stride->h, forward_stride->w};

    dim4 istride = {ic * ih * iw, ih * iw, iw, 1};
    dim4 wstride = {oc * oh * ow, oh * ow, ow, 1};
    dim4 ostride = {ic * kh * kw, kh * kw, kw, 1};

    grad_weight_1x1_secs_info_t secs_info;
    grad_weight_1x1_data_split(
        n, oc, ic,
        ih, iw, oh, ow,
        kh, kw,
        dtype,
        &secs_info);

    //(TODO) support split hw
    int nslice = DIV_UP(n, secs_info.nsecs);
    int ocslice = DIV_UP(oc, secs_info.ocsecs);
    int icslice = DIV_UP(ic, secs_info.icsecs);
    icslice = secs_info.icsecs > 1 ? ALIGN(icslice, NPU_NUM) : icslice;

    unsigned int isize = nslice * DIV_UP(icslice, NPU_NUM) *
                         tpu_aligned_feature_size(ih, iw, dtype);
    unsigned int wsize = nslice * DIV_UP(icslice, NPU_NUM) *
                         oh * ow * tpu_data_type_size(dtype);
    unsigned int osize = ocslice * DIV_UP(icslice, NPU_NUM) *
                         tpu_aligned_feature_size(kh, kw, dtype);

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
    local_addr_t waddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
    local_addr_t waddr_pong = ALIGN(waddr_ping + wsize, BANK_SIZE);
    local_addr_t oaddr_ping = ALIGN(waddr_pong + wsize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);

    local_addr_t buffer_addr = 0;
    if (secs_info.nsecs > 1) {
        unsigned int buffer_size = osize;
        buffer_addr = ALIGN(oaddr_pong + osize, BANK_SIZE);
        TPUKERNEL_ASSERT(buffer_addr + buffer_size <= (unsigned int)LOCAL_MEM_SIZE);
    }

    global_addr_t input_global_addr = forward_input_global_addr;
    global_addr_t weight_global_addr = grad_out_global_addr;
    global_addr_t output_global_addr = grad_weight_global_addr;

    bool ping_in = true;
    bool ping_weight = true;
    bool ping_out = true;
    bool parallel_branch = false;

    int last_ocslice = 0;
    int last_ocstart = 0;
    int last_icslice = 0;
    int last_icstart = 0;

    int ocstart = 0;
    int ocend = 0;
    for (int oc_idx = 0; oc_idx < secs_info.ocsecs; oc_idx++) {
        ocstart = ocend;
        ocslice = oc / secs_info.ocsecs + ((oc % secs_info.ocsecs) > oc_idx);
        ocend = ocstart + ocslice;

        int icstart = 0;
        int icend = 0;
        for (int ic_idx = 0; ic_idx < secs_info.icsecs; ic_idx++) {
            icstart = icend;
            icslice = DIV_UP(ic, secs_info.icsecs);
            icslice = MIN(ALIGN(icslice, NPU_NUM), ic - icend);
            icend = icstart + icslice;

            dim4 oslice_shape = {ocslice, icslice, kh, kw};
            dim4 oslice_stride;
            tpu_aligned_stride(&oslice_stride, 0, &oslice_shape, dtype);

            int nstart = 0;
            int nend = 0;
            for (int n_idx = 0; n_idx < secs_info.nsecs; n_idx++) {
                nstart = nend;
                nslice = n / secs_info.nsecs + ((n % secs_info.nsecs) > n_idx);
                nend = nstart + nslice;

                // load forward input [n', ic', ih, iw]
                dim4 islice_shape = {nslice, icslice, ih, iw};
                tpu_gdma_cpy_S2L(
                    ping_in ? iaddr_ping : iaddr_pong,
                    input_global_addr +
                        (nstart * istride.n +
                         icstart * istride.c) * tpu_data_type_size(dtype),
                    &islice_shape,
                    NULL,
                    &istride,
                    dtype);

                for (int ocslice_idx = 0; ocslice_idx < ocslice; ocslice_idx++) {

                    // load grad_out [n, 1, oh, ow]
                    dim4 wslice_cbcast_shape = {nslice, icslice, oh, ow};
                    dim4 wstride_cbcast_global = {oc * oh * ow, 0, ow, 1};
                    tpu_gdma_channel_bcast_S2L(
                        ping_weight ? waddr_ping : waddr_pong,
                        weight_global_addr +
                            (nstart * wstride.n +
                             (ocstart + ocslice_idx) * wstride.c) * tpu_data_type_size(dtype),
                        &wslice_cbcast_shape,
                        NULL,
                        &wstride_cbcast_global,
                        dtype);

                    if (parallel_branch) {
                        tpu_parallel_end();
                    }
                    tpu_parallel_start();
                    parallel_branch = true;

                    dim4 slice_shape = {nslice, icslice, oh, ow};
                    dim4 A_stride;
                    tpu_aligned_stride(&A_stride, 0, &islice_shape, dtype);
                    A_stride.w = stride.w;
                    A_stride.h = islice_shape.w * stride.h;
                    // [n', ic', ih, iw] * [n', 1, oh, ow] => [n', ic', oh, ow]
                    tpu_bdc_fp_mul(
                        ping_weight ? waddr_ping : waddr_pong,
                        ping_in ? iaddr_ping : iaddr_pong,
                        ping_weight ? waddr_ping : waddr_pong,
                        &slice_shape,
                        NULL,
                        (stride.h == 1 && stride.w == 1) ? NULL : &A_stride,
                        NULL,
                        dtype);

                    // [n', ic', oh, ow] => [n', ic', 1, ow] => [n', ic', 1, 1]
                    dim4 pooling_shape = {nslice, icslice, oh, ow};
                    dim2 pooling_kernel = {oh, 1};
                    dim2 pooling_stride = {1, 1};
                    dim2 pooling_dilation = {1, 1};
                    padding_t pooling_padding = {0, 0, 0, 0};
                    scalar_t scale = {.f32 = 1.f};
                    tpu_bdc_fp_avg_pool2d(
                        //ping_in ? iaddr_ping : iaddr_pong,
                        ping_weight ? waddr_ping : waddr_pong,
                        ping_weight ? waddr_ping : waddr_pong,
                        &pooling_shape,
                        &pooling_kernel,
                        &pooling_padding,
                        &pooling_stride,
                        &pooling_dilation,
                        dtype,
                        tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));

                    pooling_shape.h = 1;
                    pooling_kernel.h = 1;
                    pooling_kernel.w = pooling_shape.w;
                    tpu_bdc_fp_avg_pool2d(
                        //n_idx > 0 ? buffer_addr + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype)
                        //          : ping_out ? oaddr_ping + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype)
                        //                     : oaddr_pong + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype),
                        ping_weight ? waddr_ping : waddr_pong,
                        ping_weight ? waddr_ping : waddr_pong,
                        &pooling_shape,
                        &pooling_kernel,
                        &pooling_padding,
                        &pooling_stride,
                        &pooling_dilation,
                        dtype,
                        tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));

                    //[n', ic', 1, 1] => [1, ic', 1, n']
                    dim4 trans_shape = {nslice, icslice, 1, 1};
                    dim4 trans_dst_stride = {1, ALIGN(nslice, tpu_eu_num(dtype)), 2, 1};
                    tpu_bdc_cpy(
                        ping_weight ? waddr_ping : waddr_pong,
                        ping_weight ? waddr_ping : waddr_pong,
                        &trans_shape,
                        &trans_dst_stride,
                        NULL,
                        dtype);

                    //[1, ic', 1, n'] => [1, ic', 1, 1]
                    pooling_shape.w = nslice;
                    pooling_shape.n = 1;
                    pooling_kernel.h = 1;
                    pooling_kernel.w = nslice;
                    tpu_bdc_fp_avg_pool2d(
                        n_idx > 0 ? buffer_addr + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype)
                                  : ping_out ? oaddr_ping + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype)
                                             : oaddr_pong + ocslice_idx * oslice_stride.c * tpu_data_type_size(dtype),
                        ping_weight ? waddr_ping : waddr_pong,
                        &pooling_shape,
                        &pooling_kernel,
                        &pooling_padding,
                        &pooling_stride,
                        &pooling_dilation,
                        dtype,
                        tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
                    ping_weight = !ping_weight;
                }
                if (n_idx > 0) {
                    tpu_bdc_fp_add(
                        ping_out ? oaddr_ping : oaddr_pong,
                        ping_out ? oaddr_ping : oaddr_pong,
                        buffer_addr,
                        &oslice_shape,
                        NULL,
                        NULL,
                        NULL,
                        dtype);
                }
                ping_in = !ping_in;
            }
            if (oc_idx > 0 || ic_idx > 0) {
                dim4 last_oslice_shape = {last_ocslice, last_icslice, kh, kw};
                tpu_gdma_cpy_L2S(
                    output_global_addr +
                        (last_ocstart * ostride.n +
                         last_icstart * ostride.c) * tpu_data_type_size(dtype),
                    ping_out ? oaddr_pong : oaddr_ping,
                    &last_oslice_shape,
                    &ostride,
                    NULL,
                    dtype);
            }
            ping_out = !ping_out;
            last_ocstart = ocstart;
            last_ocslice = ocslice;
            last_icstart = icstart;
            last_icslice = icslice;
        }
    }
    tpu_parallel_end();
    dim4 last_oslice_shape = {last_ocslice, last_icslice, kh, kw};
    tpu_gdma_cpy_L2S(
        output_global_addr +
            (last_ocstart * ostride.n +
             last_icstart * ostride.c) * tpu_data_type_size(dtype),
        ping_out ? oaddr_pong : oaddr_ping,
        &last_oslice_shape,
        &ostride,
        NULL,
        dtype);
}

/*
void nodechip_conv_backward_weight_3x3(
    global_addr_t       forward_input_global_addr,
    global_addr_t       grad_out_global_addr,
    global_addr_t       grad_weight_global_addr,
    const int           groups,
    const dim4          *forward_input_shape,
    const dim4          *grad_out_shape,
    const dim2          *forward_kernel,
    const dim2          *forward_stride,
    const dim2          *forward_dilation,
    const padding_t     *pad,
    data_type_t         dtype
 ) {

    int n = forward_input_shape->n;
    int ic = forward_input_shape->c;
    int ih = forward_input_shape->h;
    int iw = forward_input_shape->w;
    int oc = grad_out_shape->c;
    int oh = grad_out_shape->h;
    int ow = grad_out_shape->w;

    int kh = forward_kernel->h;
    int kw = forward_kernel->w;
    TPUKERNEL_ASSERT(kh == 1);
    TPUKERNEL_ASSERT(kw == 1);
    TPUKERNEL_ASSERT(forward_dilation->h == 1);
    TPUKERNEL_ASSERT(forward_dilation->w == 1);

    dim2 stride = {forward_stride->h, forward_stride->w};

    dim4 istride = {ic * ih * iw, ih * iw, iw, 1};
    dim4 wstride = {oc * oh * ow, oh * ow, ow, 1};
    dim4 ostride = {ic * kh * kw, kh * kw, kw, 1};

    //(TODO)
}
*/
typedef struct{
    int icsecs;
    int ocsecs;
} grad_weight_use_conv_secs_info_t;

static inline bool grad_weight_can_use_conv_with_no_split_hw(
    const int                          n,
    const int                          oc,
    const int                          ic,
    const int                          ih,
    const int                          iw,
    const int                          oh,
    const int                          ow,
    const int                          kh,
    const int                          kw,
    data_type_t                        dtype
) {

    unsigned int smallest_isize = DIV_UP(n, NPU_NUM) * tpu_aligned_feature_size(ih, iw, dtype);
    unsigned int smallest_wsize = (dtype == DT_FP32 ? n : ALIGN(n, 32)) * oh * ow * tpu_data_type_size(dtype);
    unsigned int smallest_osize = tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int smallest_size = ALIGN(smallest_wsize, BANK_SIZE) +
                                 ALIGN(smallest_isize, BANK_SIZE) * 2 +
                                 ALIGN(smallest_osize, BANK_SIZE) * 2;
    bool valid;
    valid = smallest_size <= (unsigned long long)LOCAL_MEM_SIZE;
    return valid;
}

//[n, ic, ih, iw] op [n, oc, oh, ow] => [oc, ic, kh, kw]
//[ic, n, ih, iw] op [oc, n, oh, ow] => [ic, oc, kh, kw]
//for FP32:
//[ic, n, ih, iw] op [1, oc, n * oh * ow, 1] => [ic, oc, kh, kw]
//for FP16:
//[ic, n, ih, iw] op [1, oc, DIV_UP(n, 32) * oh * ow, 32] => [ic, oc, kh, kw]
void grad_weight_use_conv_data_split(
    const int                          n,
    const int                          oc,
    const int                          ic,
    const int                          ih,
    const int                          iw,
    const int                          oh,
    const int                          ow,
    const int                          kh,
    const int                          kw,
    data_type_t                        dtype,
    grad_weight_use_conv_secs_info_t  *secs_info
) {

    int ocslice = oc;
    int icslice = ic;
    secs_info->ocsecs = 1;
    secs_info->icsecs = 1;

    unsigned int isize = ic * DIV_UP(n, NPU_NUM) * tpu_aligned_feature_size(ih, iw, dtype);
    unsigned int wsize = DIV_UP(oc, NPU_NUM) * (dtype == DT_FP32 ? n : ALIGN(n, 32)) * oh * ow * tpu_data_type_size(dtype);
    unsigned int osize = ic * DIV_UP(oc, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int total_size = ALIGN(wsize, BANK_SIZE) +
                              ALIGN(isize, BANK_SIZE) * 2 +
                              ALIGN(osize, BANK_SIZE) * 2;
    if (total_size <= (unsigned long long)LOCAL_MEM_SIZE) return;

    unsigned int per_ic_isize = DIV_UP(n, NPU_NUM) * tpu_aligned_feature_size(ih, iw, dtype);
    unsigned int per_ic_wsize = wsize;
    unsigned int per_ic_osize = DIV_UP(oc, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int per_ic_total_size = ALIGN(per_ic_wsize, BANK_SIZE) +
                                     ALIGN(per_ic_isize, BANK_SIZE) * 2 +
                                     ALIGN(per_ic_osize, BANK_SIZE) * 2;
    if (per_ic_total_size <= (unsigned long long)LOCAL_MEM_SIZE) {
        while(1) {
            secs_info->icsecs++;
            icslice = DIV_UP(ic, secs_info->icsecs);
            unsigned int some_ic_isize = icslice * per_ic_isize;
            unsigned int some_ic_osize = icslice * per_ic_osize;
            unsigned int some_ic_total_size = ALIGN(some_ic_isize, BANK_SIZE) * 2 +
                                              ALIGN(some_ic_osize, BANK_SIZE) * 2 +
                                              ALIGN(per_ic_wsize, BANK_SIZE);
            if (some_ic_total_size <= (unsigned long long)LOCAL_MEM_SIZE) break;
        }
        return;
    }
    icslice = 1;
    secs_info->icsecs = ic;

    unsigned int per_ic_64oc_isize = per_ic_isize;
    unsigned int per_ic_64oc_wsize = (dtype == DT_FP32 ? n : ALIGN(n, 32)) * oh * ow * tpu_data_type_size(dtype);
    unsigned int per_ic_64oc_osize = tpu_aligned_feature_size(kh, kw, dtype);
    unsigned int per_ic_64oc_size = ALIGN(per_ic_64oc_wsize, BANK_SIZE) +
                                    ALIGN(per_ic_64oc_isize, BANK_SIZE) * 2 +
                                    ALIGN(per_ic_64oc_osize, BANK_SIZE) * 2;
    TPUKERNEL_ASSERT(per_ic_64oc_size <= (unsigned long long)LOCAL_MEM_SIZE);

    while(1) {
        secs_info->ocsecs++;
        ocslice = DIV_UP(oc, secs_info->ocsecs);
        ocslice = secs_info->ocsecs > 1 ? ALIGN(ocslice, NPU_NUM) : ocslice;
        unsigned int per_ic_some_64oc_isize = per_ic_64oc_isize;
        unsigned int per_ic_some_64oc_wsize = DIV_UP(ocslice, NPU_NUM) * (dtype == DT_FP32 ? n : ALIGN(n, 32)) * oh * ow * tpu_data_type_size(dtype);
        unsigned int per_ic_some_64oc_osize = DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
        unsigned int per_ic_some_64oc_total_size = ALIGN(per_ic_some_64oc_wsize, BANK_SIZE) +
                                                   ALIGN(per_ic_some_64oc_isize, BANK_SIZE) * 2 +
                                                   ALIGN(per_ic_some_64oc_osize, BANK_SIZE) * 2;
        if (per_ic_some_64oc_total_size <= (unsigned long long)LOCAL_MEM_SIZE) break;
    }

    // split icslice second
    secs_info->icsecs = 1;
    icslice = ic;
    bool first_time = true;
    unsigned int some_ic_wsize = DIV_UP(ocslice, NPU_NUM) * (dtype == DT_FP32 ? n : ALIGN(n, 32)) * oh * ow * tpu_data_type_size(dtype);
    while(1) {
        if (!first_time) {
            secs_info->icsecs++;
            icslice = DIV_UP(ic, secs_info->icsecs);
        }
        unsigned int some_ic_isize = icslice * DIV_UP(n, NPU_NUM) * tpu_aligned_feature_size(ih, iw, dtype);
        unsigned int some_ic_osize = icslice * DIV_UP(ocslice, NPU_NUM) * tpu_aligned_feature_size(kh, kw, dtype);
        unsigned int some_ic_total_size = ALIGN(some_ic_wsize, BANK_SIZE) +
                                          ALIGN(some_ic_isize, BANK_SIZE) * 2 +
                                          ALIGN(some_ic_osize, BANK_SIZE) * 2;
        if (some_ic_total_size <= (unsigned long long)LOCAL_MEM_SIZE) break;
        first_time = false;
    }
}

// forward_input op grad_out => grad_weight
// [ic, n, ih, iw] conv [oc, n, oh, ow] => [ic, oc, kh, kw]
void nodechip_conv_backward_weight_use_conv(
    global_addr_t       forward_input_global_addr,
    global_addr_t       grad_out_global_addr,
    global_addr_t       grad_out_reordered_global_addr,
    global_addr_t       grad_weight_global_addr,
    const int           groups,
    const dim4          *forward_input_shape,
    const dim4          *grad_out_shape,
    const dim2          *forward_kernel,
    const dim2          *forward_stride,
    const dim2          *forward_dilation,
    const padding_t     *pad,
    data_type_t         dtype
 ) {

    dim2 stride, dilation;
    stride.h = forward_dilation->h;
    stride.w = forward_dilation->w;
    dilation.h = forward_stride->h;
    dilation.w = forward_stride->w;

    int n = forward_input_shape->n;
    int ic = forward_input_shape->c;
    int ih = forward_input_shape->h;
    int iw = forward_input_shape->w;
    int oc = grad_out_shape->c;
    int oh = grad_out_shape->h;
    int ow = grad_out_shape->w;
    int kh = forward_kernel->h;
    int kw = forward_kernel->w;

    dim2 kernel = {oh, ow};

    //dim4 ishape = {n, ic, ih, iw};
    //dim4 wshape = {n, oc, oh, ow};
    //dim4 oshape = {oc, ic, kh, kw};

    grad_weight_use_conv_secs_info_t secs_info;
    grad_weight_use_conv_data_split(
        n, oc, ic,
        ih, iw, oh, ow,
        kh, kw,
        dtype,
        &secs_info);

    int icslice = DIV_UP(ic, secs_info.icsecs);
    int ocslice = DIV_UP(oc, secs_info.ocsecs);
    ocslice = secs_info.ocsecs > 1 ? ALIGN(ocslice, NPU_NUM) : ocslice;
    unsigned int isize = icslice * DIV_UP(n, NPU_NUM) *
                         tpu_aligned_feature_size(ih, iw, dtype);
    unsigned int wsize = DIV_UP(ocslice, NPU_NUM) *
                         (dtype == DT_FP32 ? n : ALIGN(n, 32)) *
                         oh * ow * tpu_data_type_size(dtype);
    unsigned int osize = icslice * DIV_UP(ocslice, NPU_NUM) *
                         tpu_aligned_feature_size(kh, kw, dtype);

    local_addr_t waddr = 0;
    local_addr_t iaddr_ping = ALIGN(waddr + wsize, BANK_SIZE);
    local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
    local_addr_t oaddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);

    dim4 istride = {
        ic * ih * iw,
        ih * iw,
        iw,
        1};
    dim4 wstride = {
        oc * oh * ow,
        oh * ow,
        ow,
        1};

    dim4 ostride = {
        ic * kh * kw,
        kh * kw,
        kw,
        1};

    if (tpu_data_type_size(dtype) == 2) {
        nodechip_grad_output_reorder(grad_out_global_addr,
                                     grad_out_reordered_global_addr,
                                     n,
                                     oc,
                                     oh,
                                     ow,
                                     dtype);
    }

    global_addr_t input_global_addr = forward_input_global_addr;
    global_addr_t weight_global_addr = dtype == DT_FP32 ? grad_out_global_addr
                                                        : grad_out_reordered_global_addr;
    global_addr_t output_global_addr = grad_weight_global_addr;

    bool ping = true;

    int last_icstart = 0;
    int last_icslice = 0;

    int ocstart = 0;
    int ocend = 0;
    for (int oc_idx = 0; oc_idx < secs_info.ocsecs; oc_idx++) {
        ocstart = ocend;
        ocslice = DIV_UP(oc, secs_info.ocsecs);
        ocslice = MIN(ALIGN(ocslice, NPU_NUM), oc - ocend);
        ocend = ocstart + ocslice;

        dim4 wslice_shape = {1,
                             ocslice,
                             (dtype == DT_FP32 ? n : DIV_UP(n, 32)) * oh * ow,
                             dtype == DT_FP32 ? 1 : 32};

        dim4 wslice_stride;
        tpu_compact_stride(&wslice_stride, 0, &wslice_shape);

        //load weight
        if (tpu_data_type_size(dtype) == 2) {
            tpu_gdma_cpy_S2L(
                waddr,
                weight_global_addr +
                    ocstart * DIV_UP(n, 32) * oh * ow * 32 * tpu_data_type_size(dtype),
                &wslice_shape,
                &wslice_stride,
                NULL,
                dtype);
        } else if (tpu_data_type_size(dtype) == 4) {
            dim4 wslice_shape_fp32 = {n, ocslice, oh, ow};
            dim4 wslice_stride_local = {.n = oh * ow, .c = n * oh * ow, .h = ow, .w = 1};
            tpu_gdma_cpy_S2L(
                waddr,
                weight_global_addr +
                    ocstart * oh * ow * tpu_data_type_size(dtype),
                &wslice_shape_fp32,
                &wslice_stride_local,
                &wstride,
                dtype);
        }

        bool parallel_branch = false;
        int icstart = 0;
        int icend = 0;
        for (int ic_idx = 0; ic_idx < secs_info.icsecs; ic_idx++) {
            icstart = icend;
            icslice = ic / secs_info.icsecs + ((ic % secs_info.icsecs) > ic_idx);
            icend = icstart + icslice;

            int oh_ext = dilation.h * (oh - 1) + 1;
            int ow_ext = dilation.w * (ow - 1) + 1;

            padding_t pad_slice;
            int ihstart = - pad->top;
            int ihslice = (kh - 1) * stride.h + oh_ext;
            int ihend = ihstart + ihslice;
            pad_slice.top = ihstart < 0 ? -ihstart : 0;
            pad_slice.bottom = ihend > ih ? ihend - ih : 0;
            ihstart = ihstart < 0 ? 0 : ihstart;
            ihend = ihend > ih ? ih : ihend;
            ihslice = ihend - ihstart;

            int iwstart = - pad->left;
            int iwslice = (kw - 1) * stride.w + ow_ext;
            int iwend = iwstart + iwslice;
            pad_slice.left = iwstart < 0 ? -iwstart : 0;
            pad_slice.right = iwend > iw ? iwend - iw : 0;
            iwstart = iwstart < 0 ? 0 : iwstart;
            iwend = iwend > iw ? iw : iwend;
            iwslice = iwend - iwstart;

            //TODO need split kh/kw ?
            dim4 islice_shape = {icslice, n, ihslice, iwslice};
            tpu_gdma_cpy_nc_trans_S2L(
                ping ? iaddr_ping : iaddr_pong,
                input_global_addr +
                    icstart * istride.c * tpu_data_type_size(dtype),
                &islice_shape,
                NULL,
                &istride,
                dtype);

            if (parallel_branch) {
                tpu_parallel_end();
            }
            tpu_parallel_start();
            parallel_branch = true;

            tpu_bdc_fp_conv2d(
                ping ? oaddr_ping : oaddr_pong,
                ping ? iaddr_ping : iaddr_pong,
                waddr,
                0,
                &islice_shape,
                NULL,
                ocslice,
                &kernel,
                &pad_slice,
                &stride,
                &dilation,
                dtype,
                dtype,
                false,
                false);

            if (ic_idx > 0) {
                // move out
                dim4 last_oslice_shape = {ocslice, last_icslice, kh, kw};
                tpu_gdma_cpy_nc_trans_L2S(
                    output_global_addr +
                        (ocstart * ostride.n +
                         last_icstart * ostride.c) * tpu_data_type_size(dtype),
                    ping ? oaddr_pong : oaddr_ping,
                    &last_oslice_shape,
                    &ostride,
                    NULL,
                    dtype);
            }
            ping = !ping;
            last_icstart = icstart;
            last_icslice = icslice;
        }
        tpu_parallel_end();
        dim4 last_oslice_shape = {ocslice, last_icslice, kh, kw};
        tpu_gdma_cpy_nc_trans_L2S(
            output_global_addr +
                (ocstart * ostride.n +
                 last_icstart * ostride.c) * tpu_data_type_size(dtype),
            ping ? oaddr_pong : oaddr_ping,
            &last_oslice_shape,
            &ostride,
            NULL,
            dtype);
    }
}

/*
//[n, ic, ih, iw] => [ic, n * ih * iw]
//[n, oc, oh, ow] => [oc, n * oh * ow]
//for (kh * kw) {
//  [oc, n * oh * ow] MM2_NT [ic, n * oh * ow] => [oc, ic]
//}
//kh * kw * [oc, ic] => [oc, ic, kh, kw]
//kh * kw * [oc, ic] => [1, oc, kh * kw, ic]
void nodechip_conv_backward_weight_use_mm2(
    global_addr_t       forward_input_global_addr,
    global_addr_t       grad_out_global_addr,
    global_addr_t       grad_out_reordered_global_addr,
    global_addr_t       grad_weight_global_addr,
    const int           groups,
    const dim4          *forward_input_shape,
    const dim4          *grad_out_shape,
    const dim2          *forward_kernel,
    const dim2          *forward_stride,
    const dim2          *forward_dilation,
    const padding_t     *pad,
    data_type_t         dtype
 ) {

  //(TODO)

}
*/

void nodechip_conv_backward(
    global_addr_t       grad_out_global_addr,
    global_addr_t       input_global_addr,
    global_addr_t       weight_global_addr,
    global_addr_t       grad_input_global_addr,
    global_addr_t       grad_weight_global_addr,
    global_addr_t       grad_bias_global_addr,
    global_addr_t       buffer_global_addr,
    const dim4          *input_shape,
    const dim4          *grad_out_shape,
    const dim2          *kernel,
    const dim2          *stride,
    const dim2          *dilation,
    const padding_t     *pad,
    const int           groups,
    bool                grad_input_enable,
    bool                grad_weight_enable,
    bool                grad_bias_enable,
    data_type_t         dtype
 ) {

    TPUKERNEL_ASSERT(groups == 1);
    if (grad_input_enable) {
        nodechip_conv_backward_input(
            grad_out_global_addr,
            grad_input_global_addr,
            weight_global_addr,
            buffer_global_addr,
            groups,//groups TODO
            input_shape,
            grad_out_shape,
            kernel,
            stride,
            dilation,
            pad,
            dtype);
    }
    if (grad_weight_enable) {
        if ((input_shape->h * input_shape->w > 10000) || grad_bias_enable) {
            nodechip_conv_backward_weight(
                input_global_addr,
                grad_out_global_addr,
                buffer_global_addr,
                grad_weight_global_addr,
                grad_bias_global_addr,
                groups,//groups TODO
                input_shape,
                grad_out_shape,
                kernel,
                stride,
                dilation,
                pad,
                grad_bias_enable,
                dtype);
        } else {
            bool can_use_conv = grad_weight_can_use_conv_with_no_split_hw(
                                    input_shape->n,
                                    grad_out_shape->c, input_shape->c,
                                    input_shape->h, input_shape->w,
                                    grad_out_shape->h, grad_out_shape->w,
                                    kernel->h, kernel->w, dtype);
            if (kernel->h == 1 && kernel->w == 1 &&
                       dilation->h == 1 && dilation->w == 1 &&
                       pad->top == 0 && pad->bottom == 0 && pad->left == 0 && pad->right == 0) {
                nodechip_conv_backward_weight_1x1(
                    input_global_addr,
                    grad_out_global_addr,
                    grad_weight_global_addr,
                    groups,
                    input_shape,
                    grad_out_shape,
                    kernel,
                    stride,
                    dilation,
                    pad,
                    dtype);
            } else if (can_use_conv) {
                nodechip_conv_backward_weight_use_conv(
                    input_global_addr,
                    grad_out_global_addr,
                    buffer_global_addr,
                    grad_weight_global_addr,
                    groups,
                    input_shape,
                    grad_out_shape,
                    kernel,
                    stride,
                    dilation,
                    pad,
                    dtype);
            } else {
                nodechip_conv_backward_weight_depthwise(
                    input_global_addr,
                    grad_out_global_addr,
                    grad_weight_global_addr,
                    groups,
                    input_shape,
                    grad_out_shape,
                    kernel,
                    stride,
                    dilation,
                    pad,
                    dtype);
            }
        }
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
        api->groups,
        api->grad_input_enable == 1 ? true : false,
        api->grad_weight_enable == 1 ? true : false,
        api->grad_bias_enable == 1 ? true : false,
        tpu_type_convert(api->dtype));
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_conv_backward);
