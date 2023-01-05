#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"
#ifdef USING_CMODEL
#include "cmodel_memory.h"
#endif

typedef struct {
    int nsecs;
    int csecs;
    int ohsecs;
    int owsecs;
} avgpool_secs_info_t;

static inline bool is_local_mem_enough(
        int *size,
        int c,
        int ihw,
        int ohw,
        data_type_t dtype) {
    int c_per_npu = DIV_UP(c, NPU_NUM);
    size[0] = c_per_npu * tpu_aligned_feature_size(1, ihw, dtype);
    size[1] = c_per_npu * tpu_aligned_feature_size(1, ohw, dtype);
    int total_size = ALIGN(size[0], BANK_SIZE) * 2 +
                     ALIGN(size[1], BANK_SIZE) * 2;
    return total_size <= LOCAL_MEM_SIZE;
}

static inline bool split_oh_or_ow(
    int         *slice,
    int         *secs,
    int         stride,
    int         c,
    int         ih_or_iw,
    int         oh_or_ow,
    int         kh_or_kw_ext,
    data_type_t dtype) {

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
        // i_ext = (i - 1) * stride + 1
        // o = i_ext - k_ext + 1
        int islice = DIV_UP(slice_new - 1 + kh_or_kw_ext - 1, stride) + 1;
        valid = is_local_mem_enough(
                size,
                c,
                islice * ih_or_iw,
                slice_new * oh_or_ow,
                dtype);
        ++(*secs);
    } while(!valid);
    *slice = slice_new;
    --(*secs);
    return true;
}

static void avgpool_data_split(
    int             n,
    int             c,
    int             ih,
    int             iw,
    int             oh,
    int             ow,
    int             kh,
    int             kw,
    int             dh,
    int             dw,
    int             stride_h,
    int             stride_w,
    data_type_t     dtype,
    avgpool_secs_info_t *p_secs
  ) {

    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;

    p_secs->nsecs = n;
    p_secs->csecs = 1;
    p_secs->ohsecs = 1;
    p_secs->owsecs = 1;
    int nslice = 1;
    int cslice = c;
    int ohslice = oh;
    int owslice = ow;

    int size[2];    // {input, output}
    int total_size;
    bool valid = is_local_mem_enough(size,
                                     c,
                                     ih * iw,
                                     oh * ow,
                                     dtype);
    if (valid) {
        total_size = ALIGN(size[0], BANK_SIZE) * 2 + ALIGN(size[1], BANK_SIZE) * 2;
        nslice = LOCAL_MEM_SIZE / total_size;
        p_secs->nsecs = DIV_UP(n, nslice);
        return;
    }

    /*
      c > 64 ?
          64c can hold => noneed split hw
          64c cannot hold => c = 64 && split hw
      c <= 64 ? split hw
    */
    // split c
    if (c > NPU_NUM) {
        // (1, NPU_NUM, ih, iw)&(1, NPU_NUM, oh, ow) can hold?
        bool valid = is_local_mem_enough(size,
                                         NPU_NUM,
                                         ih * iw,
                                         oh * ow,
                                         dtype);
        if (valid) {
            total_size = ALIGN(size[0], BANK_SIZE) * 2 + ALIGN(size[1], BANK_SIZE) * 2;
            cslice = LOCAL_MEM_SIZE / total_size * NPU_NUM;
            p_secs->csecs = DIV_UP(c, cslice);
            return;
        }
    }

    cslice = MIN(c, NPU_NUM);
    p_secs->csecs = DIV_UP(c, cslice);
    int hwsecs = 1 + (size[0] + size[1]) / LOCAL_MEM_SIZE;
    p_secs->ohsecs = hwsecs;
    valid = split_oh_or_ow(
            &ohslice,
            &(p_secs->ohsecs),
            stride_h,
            cslice,
            iw,
            owslice,
            kh_ext,
            dtype);
    if (!valid) {
        // check
        int ihslice = DIV_UP(ohslice - 1 + kh_ext - 1, stride_h) + 1;
        valid = split_oh_or_ow(
                &owslice,
                &(p_secs->owsecs),
                stride_w,
                cslice,
                ihslice,
                ohslice,
                kw_ext,
                dtype);
    }
    TPUKERNEL_ASSERT(valid);
}

void nodechip_avgpool_backward(
    global_addr_t    grad_out_global_addr,
    global_addr_t    grad_input_global_addr,
    const dim4       *grad_input_shape,
    const dim4       *grad_out_shape,
    const dim2       *kernel,
    const dim2       *stride,
    const padding_t  *pad,
    bool             ceil_mode,
    bool             count_include_pad,
    const int        divisor_override
 ) {

    TPUKERNEL_ASSERT(ceil_mode == false);
    TPUKERNEL_ASSERT(count_include_pad == true);

    const data_type_t dtype = DT_FP16;

    const int dh = 1;
    const int dw = 1;
    const dim2 dilation = {dh, dw};
    const dim2 insert = {stride->h - 1, stride->w - 1};

    const int kh = kernel->h;
    const int kw = kernel->w;
    //const int pad_h_t = pad->top;
    //const int pad_h_b = pad->bottom;
    //const int pad_w_l = pad->left;
    //const int pad_w_r = pad->right;
    const int stride_h = stride->h;
    const int stride_w = stride->w;

    const int n = grad_input_shape->n;
    const int c = grad_input_shape->c;
    const int ih = grad_out_shape->h;
    const int iw = grad_out_shape->w;
    const int oh = grad_input_shape->h;
    const int ow = grad_input_shape->w;

    dim4 ishape = {n, c, ih, iw};
    dim4 oshape = {n, c, oh, ow};

    dim4 istride, ostride;
    tpu_continuous_stride(&istride, &ishape);
    tpu_continuous_stride(&ostride, &oshape);

    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;
    //int ih_ext = ih + pad_h_t + pad_h_b;
    //int iw_ext = iw + pad_w_l + pad_w_r;
    int ih_ext = (ih - 1) * stride->h + 1;
    int iw_ext = (iw - 1) * stride->w + 1;
    int pad_h0 = kh_ext - pad->top - 1;
    int pad_w0 = kw_ext - pad->left - 1;

    avgpool_secs_info_t secs_info;
    avgpool_data_split(
        n, c, ih, iw, oh, ow,
        kh, kw, dh, dw,
        stride_h, stride_w,
        dtype, &secs_info);

    int nslice = DIV_UP(n, secs_info.nsecs);
    int cslice = DIV_UP(c, secs_info.csecs);
    cslice = secs_info.csecs > 1 ? ALIGN(cslice, NPU_NUM) : cslice;
    int ohslice = DIV_UP(oh, secs_info.ohsecs);
    int owslice = DIV_UP(ow, secs_info.owsecs);
    int ihslice = MIN(DIV_UP(ohslice - 1 + kh_ext - 1,  stride->h) + 1, ishape.h);
    int iwslice = MIN(DIV_UP(owslice - 1 + kw_ext - 1,  stride->w) + 1, ishape.w);
    TPUKERNEL_DBG("nsecs=%d  nslice=%d\n", secs_info.nsecs, nslice);
    TPUKERNEL_DBG("csecs=%d  cslice=%d\n", secs_info.csecs, cslice);
    TPUKERNEL_DBG("ohsecs=%d  ohslice=%d\n", secs_info.ohsecs, ohslice);
    TPUKERNEL_DBG("owsecs=%d  owslice=%d\n", secs_info.owsecs, owslice);

    unsigned int isize = nslice * DIV_UP(cslice, NPU_NUM) *
        tpu_aligned_feature_size(ihslice, iwslice, dtype);
    unsigned int osize = nslice * DIV_UP(cslice, NPU_NUM) *
        tpu_aligned_feature_size(ohslice, owslice, dtype);

    local_addr_t iaddr_ping = 0;
    local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
    local_addr_t oaddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
    local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);
    TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);

    TPUKERNEL_DBG("in  ping local addr = 0x%5x, bank id = %d\n", iaddr_ping, iaddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("in  pong local addr = 0x%5x, bank id = %d\n", iaddr_pong, iaddr_pong / BANK_SIZE);
    TPUKERNEL_DBG("out ping local addr = 0x%5x, bank id = %d\n", oaddr_ping, oaddr_ping / BANK_SIZE);
    TPUKERNEL_DBG("out pong local addr = 0x%5x, bank id = %d\n", oaddr_pong, oaddr_pong / BANK_SIZE);

    global_addr_t input_global_addr = grad_out_global_addr;
    global_addr_t output_global_addr = grad_input_global_addr;

    bool ping = true;
    int last_nstart = 0;
    int last_nslice = 0;
    int last_cstart = 0;
    int last_cslice = 0;
    int last_ohstart = 0;
    int last_ohslice = 0;
    int last_owstart = 0;
    int last_owslice = 0;

    bool parallel_branch = false;
    int nstart = 0;
    int nend = 0;
    // 1th loop: split N dim
    for (int nidx = 0; nidx < secs_info.nsecs; nidx++) {
        nstart = nend;
        nslice = ishape.n / secs_info.nsecs + ((ishape.n % secs_info.nsecs) > nidx);
        nend = nstart + nslice;

        int cstart = 0;
        int cend = 0;
        // 2nd loop: split C dim
        for (int cidx = 0; cidx < secs_info.csecs; cidx++) {
            // c boundary: [cstart, cend)
            cstart = cend;
            cslice = MIN(cslice, c - cend);
            cend = cstart + cslice;

            int ohstart = 0;
            int ohend = 0;
            padding_t slice_pad;
            // 3th loop: split OH dim
            for (int ohidx = 0; ohidx < secs_info.ohsecs; ohidx++) {
                // oh boundary: [ohstart, ohend)
                // ih boundary: [ihstart, ihend)
                ohstart = ohend;
                ohslice = oh / secs_info.ohsecs + ((oh % secs_info.ohsecs) > ohidx);
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
                // 4th loop: split OW dim
                for (int owidx = 0; owidx < secs_info.owsecs; owidx++) {
                    // ow boundary: [owstart, owend)
                    // iw boundary: [iwstart, iwend)
                    owstart = owend;
                    owslice = ow / secs_info.owsecs + ((ow % secs_info.owsecs) > owidx);
                    owend = owstart + owslice;
                    int iwstart = owstart - pad_w0;
                    int iwend = owend - 1 - pad_w0 + kw_ext;
                    slice_pad.left = iwstart < 0 ? -iwstart : ALIGN(iwstart, stride->w) - iwstart;
                    slice_pad.right = iwend > iw_ext ? iwend - iw_ext : (iwend - 1) % stride->w;
                    iwstart = iwstart < 0 ? 0 : DIV_UP(iwstart, stride->w);
                    iwend = iwend > iw_ext ? ishape.w : DIV_UP(iwend, stride->w);
                    iwslice = iwend - iwstart;

                    dim4 islice_shape = {nslice, cslice, ihslice, iwslice};
                    //dim4 oslice_shape = {nslice, cslice, ohslice, owslice};
                    // copy input from global
                    tpu_gdma_cpy_S2L(
                        ping ? iaddr_ping : iaddr_pong,
                        input_global_addr +
                            (nstart * istride.n +
                             cstart * istride.c +
                             ihstart * istride.h +
                             iwstart) * tpu_data_type_size(dtype),
                        &islice_shape,
                        NULL,
                        &istride,
                        dtype);

                    if (parallel_branch) {
                        tpu_parallel_end();
                    }

                    tpu_parallel_start();
                    parallel_branch = true;

                    scalar_t divisor = {.f32 = 1.0f / divisor_override};
                    tpu_bdc_fp_mul_C(
                        ping ? iaddr_ping : iaddr_pong,
                        ping ? iaddr_ping : iaddr_pong,
                        tpu_cast(divisor, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                        &islice_shape,
                        NULL,
                        NULL,
                        dtype);

                    scalar_t kernel_const = {.f32 = 1.0f};
                    tpu_bdc_fp_depthwise_for_deconv2d(
                        ping ? oaddr_ping : oaddr_pong,
                        ping ? iaddr_ping : iaddr_pong,
                        tpu_cast(kernel_const, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO).u32,
                        0,
                        &islice_shape,
                        kernel,
                        &insert,
                        &slice_pad,
                        &dilation,
                        dtype,
                        dtype,
                        true,
                        false);

                    // move output to global memory
                    if (nidx > 0 || cidx > 0 || ohidx > 0 || owidx > 0) {
                        dim4 last_oslice_shape = {last_nslice, last_cslice, last_ohslice, last_owslice};
                        tpu_gdma_cpy_L2S(
                            output_global_addr +
                                (last_nstart * ostride.n +
                                 last_cstart * ostride.c +
                                 last_ohstart * ostride.h +
                                 last_owstart) * tpu_data_type_size(dtype),
                            ping ? oaddr_pong : oaddr_ping,
                            &last_oslice_shape,
                            &ostride,
                            NULL,
                            dtype);
                    }
                    ping = !ping;
                    // save current info used for moving output to global memory next loop
                    last_nstart = nstart;
                    last_nslice = nslice;
                    last_cstart = cstart;
                    last_cslice = cslice;
                    last_ohstart = ohstart;
                    last_ohslice = ohslice;
                    last_owstart = owstart;
                    last_owslice = owslice;
                }
            }
        }
    }
    tpu_parallel_end();
    // move the last output to global memory
    dim4 last_oslice_shape = {last_nslice, last_cslice, last_ohslice, last_owslice};
    tpu_gdma_cpy_L2S(
        output_global_addr +
            (last_nstart * ostride.n +
             last_cstart * ostride.c +
             last_ohstart * ostride.h +
             last_owstart) * tpu_data_type_size(dtype),
        ping ? oaddr_pong : oaddr_ping,
        &last_oslice_shape,
        &ostride,
        NULL,
        dtype);
}

void tpu_kernel_api_avgpool_backward(const void* args) {
    sg_api_avgpool_backward_t* api = (sg_api_avgpool_backward_t*)args;

    dim4 input_shape = {api->input_shape[0], api->input_shape[1],
                        api->input_shape[2], api->input_shape[3]};
    dim4 output_shape = {api->output_shape[0], api->output_shape[1],
                         api->output_shape[2], api->output_shape[3]};
    dim2 kernel = {api->kernel[0], api->kernel[1]};
    dim2 stride = {api->stride[0], api->stride[1]};
    padding_t pad = {api->pad[0], api->pad[0], api->pad[1], api->pad[1]};

    tpu_initialize();
    nodechip_avgpool_backward(
        api->grad_output_global_addr,
        api->grad_input_global_addr,
        &input_shape,
        &output_shape,
        &kernel,
        &stride,
        &pad,
        api->ceil_mode == 1 ? true : false,
        api->count_include_pad == 1 ? true : false,
        api->divisor_override);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_avgpool_backward);
