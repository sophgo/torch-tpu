#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"
#ifdef USING_CMODEL
#include "cmodel_memory.h"
#endif

/*
 * index: (1, 1, ih, iw);
 * grad_out: (n, c, oh, ow);
 * input : (n, c, ih, iw);
 * output : (n, c, oh, ow);
 * grad_input: (n, c, ih, iw);
 *
 * without pad
 * generate index with inverted order in index_addr{15, 14, 13, ..., 2, 1, 0}
 * out depthwise_for_deconv => grad_input
 * grad_input AR_SE input => input
 * input AR_MUL(n&c_broadcast) index => input
 * input max_pool => out
 * out depthwise_for_deconv => input
 * input AR_SE(n&c_broadcast) index => input
 * grad_out depthwise_for_deconv => grad_input
 * input AR_MUL grad_input => grad_input
 */

typedef struct {
    int nsecs;
    int csecs;
    int ohsecs;
    int owsecs;
} maxpool_secs_info_t;

static inline bool is_local_mem_enough(
        int *size,
        int c,
        int ihw,
        int ohw,
        bool stride_less_kernel,
        data_type_t dtype) {
    int c_per_npu = DIV_UP(c, NPU_NUM);
    size[0] = 1 * tpu_aligned_feature_size(1, stride_less_kernel ? ihw : ohw, DT_INT32);
    size[1] = c_per_npu * tpu_aligned_feature_size(1, ihw, dtype);
    size[2] = c_per_npu * tpu_aligned_feature_size(1, ohw, dtype);
    int total_size = ALIGN(size[0], BANK_SIZE) +
                     ALIGN(size[1], BANK_SIZE) * 2 * 2 +
                     ALIGN(size[2], BANK_SIZE) * 2 * 2 +
                     (stride_less_kernel ? ALIGN(size[2], BANK_SIZE) : 0);
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
    int         islice_pad,
    bool        stride_less_kernel,
    data_type_t dtype) {

    bool valid;
    int size[3];
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
        // stride_less_kernel use maxpool:
        //     o = (i - k_ext) / stride + 1
        // !stride_less_kernel usr deconv:
        //     i_ext = (i - 1) * stride + 1
        //     o = i_ext - k_ext + 1
        int islice = stride_less_kernel ?
            (slice_new - 1) * stride + kh_or_kw_ext:
            DIV_UP(slice_new - 1 + kh_or_kw_ext - 1, stride) + 1;
        valid = is_local_mem_enough(
                size,
                c,
                stride_less_kernel ?
                    (islice + islice_pad) * ih_or_iw : islice * ih_or_iw,
                slice_new * oh_or_ow,
                stride_less_kernel,
                dtype);
        ++(*secs);
    } while(!valid);
    *slice = slice_new;
    --(*secs);
    return true;
}

static void maxpool_data_split(
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
    padding_t       padding,
    data_type_t     dtype,
    maxpool_secs_info_t *p_secs
  ) {

    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;

    int pt = padding.top;
    int pb = padding.bottom;
    int pl = padding.left;
    int pr = padding.right;

    p_secs->nsecs = n;
    p_secs->csecs = 1;
    p_secs->ohsecs = 1;
    p_secs->owsecs = 1;
    int nslice = 1;
    int cslice = c;
    int ohslice = oh;
    int owslice = ow;

    bool stride_less_kernel = stride_h < kh || stride_w < kw;

    int size[3]; // {index, input/grad_in, output/grad_out}
    int index_size, io_size;
    bool valid = is_local_mem_enough(size, c,
                                     stride_less_kernel ?
                                       (ih + pt + pb) * (iw + pl + pr) : ih * iw,
                                     oh * ow, stride_less_kernel, dtype);
    if (valid) {
        index_size = ALIGN(size[0], BANK_SIZE);
        io_size = ALIGN(size[1], BANK_SIZE) * 2 * 2 + ALIGN(size[2], BANK_SIZE) * 2 * 2 + (stride_less_kernel ? ALIGN(size[2], BANK_SIZE) : 0);
        nslice = (LOCAL_MEM_SIZE - index_size) / io_size;
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
        bool valid = is_local_mem_enough(size, NPU_NUM,
                                         stride_less_kernel ?
                                           (ih + pt + pb) * (iw + pl + pr) : ih * iw,
                                         oh * ow, stride_less_kernel, dtype);
        if (valid) {
            index_size = ALIGN(size[0], BANK_SIZE);
            io_size = ALIGN(size[1], BANK_SIZE) * 2 * 2 +
                      ALIGN(size[2], BANK_SIZE) * 2 * 2 +
                      (stride_less_kernel ? ALIGN(size[2], BANK_SIZE) : 0);
            cslice = (LOCAL_MEM_SIZE - index_size) / io_size * NPU_NUM;
            p_secs->csecs = DIV_UP(c, cslice);
            return;
        }
    }

    cslice = MIN(c, NPU_NUM);
    p_secs->csecs = DIV_UP(c, cslice);
    int hwsecs = 1 + (size[1] * 2 * 2 + size[2] * 2 * 2 + (stride_less_kernel ? size[2] : 0)) / (LOCAL_MEM_SIZE - ALIGN(size[0], BANK_SIZE));
    p_secs->ohsecs = hwsecs;
    valid = split_oh_or_ow(
            &ohslice,
            &(p_secs->ohsecs),
            stride_h,
            cslice,
            iw,
            owslice,
            kh_ext,
            pt + pb,
            stride_less_kernel,
            dtype);
    if (!valid) {
        int ihslice = stride_less_kernel ?
                          (ohslice - 1) * stride_h + kh_ext :
                          DIV_UP(ohslice - 1 + kh_ext - 1, stride_h) + 1;
        valid = split_oh_or_ow(
                &owslice,
                &(p_secs->owsecs),
                stride_w,
                cslice,
                ihslice,
                ohslice,
                kw_ext,
                pl + pr,
                stride_less_kernel,
                dtype);
    }
    TPUKERNEL_ASSERT(valid);
}

void nodechip_maxpool_backward_stride_noless_kernel(
    global_addr_t    forward_in_global_addr,
    global_addr_t    forward_out_global_addr,
    global_addr_t    grad_out_global_addr,
    global_addr_t    grad_in_global_addr,
    const dim4       *grad_input_shape,
    const dim4       *grad_out_shape,
    const dim2       *kernel,
    const dim2       *stride,
    const padding_t  *padding,
    const dim2       *dilation,
    bool             ceil_mode,
    data_type_t      dtype
  ) {

    const int kh = kernel->h;
    const int kw = kernel->w;
    const int dh = dilation->h;
    const int dw = dilation->w;
    const int stride_h = stride->h;
    const int stride_w = stride->w;
    dim2 insert = {stride->h - 1, stride->w - 1};

    const int n = grad_input_shape->n;
    const int c = grad_input_shape->c;
    const int ih = grad_out_shape->h;
    const int iw = grad_out_shape->w;
    const int oh = grad_input_shape->h;
    const int ow = grad_input_shape->w;

    const dim4 ishape = {n, c, ih, iw};
    const dim4 oshape = {n, c, oh, ow};

    dim4 istride, ostride;
    tpu_continuous_stride(&istride, &ishape);
    tpu_continuous_stride(&ostride, &oshape);

    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;
    int ih_ext = (ih - 1) * stride->h + 1;
    int iw_ext = (iw - 1) * stride->w + 1;
    int pad_h0 = kh_ext - padding->top - 1;
    int pad_w0 = kw_ext - padding->left - 1;

    padding_t forward_pad = {padding->top,
                             padding->bottom,
                             padding->left,
                             padding->right};

    maxpool_secs_info_t secs_info;
    maxpool_data_split(
        n, c, ih, iw, oh, ow,
        kh, kw, dh, dw,
        stride_h, stride_w,
        forward_pad, dtype, &secs_info);

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

    unsigned int index_size = tpu_aligned_feature_size(ohslice, owslice, DT_INT32);
    unsigned int isize = nslice * DIV_UP(cslice, NPU_NUM) *
        tpu_aligned_feature_size(ihslice, iwslice, dtype);
    unsigned int osize = nslice * DIV_UP(cslice, NPU_NUM) *
        tpu_aligned_feature_size(ohslice, owslice, dtype);
    unsigned int forward_in_size = osize;
    unsigned int forward_out_size = isize;

    local_addr_t index_addr = 0;
    local_addr_t forward_in_ping = ALIGN(index_addr + index_size, BANK_SIZE);
    local_addr_t forward_in_pong = ALIGN(forward_in_ping + forward_in_size, BANK_SIZE);
    local_addr_t forward_out_ping = ALIGN(forward_in_pong + forward_in_size, BANK_SIZE);
    local_addr_t forward_out_pong = ALIGN(forward_out_ping + forward_out_size, BANK_SIZE);
    local_addr_t grad_out_ping = ALIGN(forward_out_pong + forward_out_size, BANK_SIZE);
    local_addr_t grad_out_pong = ALIGN(grad_out_ping + isize, BANK_SIZE);
    local_addr_t grad_in_ping = ALIGN(grad_out_pong + isize, BANK_SIZE);
    local_addr_t grad_in_pong = ALIGN(grad_in_ping + osize, BANK_SIZE);
    TPUKERNEL_ASSERT(grad_in_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);

    TPUKERNEL_DBG("index local addr = 0x%5x, bank id = %d\n", index_addr, index_addr / BANK_SIZE);
    TPUKERNEL_DBG("forward in ping local addr = 0x%5x, bank id = %d\n", forward_in_ping, forward_in_ping / BANK_SIZE);
    TPUKERNEL_DBG("forward in pong local addr = 0x%5x, bank id = %d\n", forward_in_pong, forward_in_pong / BANK_SIZE);
    TPUKERNEL_DBG("forward out ping local addr = 0x%5x, bank id = %d\n", forward_out_ping, forward_out_ping / BANK_SIZE);
    TPUKERNEL_DBG("forward out pong local addr = 0x%5x, bank id = %d\n", forward_out_pong, forward_out_pong / BANK_SIZE);
    TPUKERNEL_DBG("grad out ping local addr = 0x%5x, bank id = %d\n", grad_out_ping, grad_out_ping / BANK_SIZE);
    TPUKERNEL_DBG("grad out pong local addr = 0x%5x, bank id = %d\n", grad_out_pong, grad_out_pong / BANK_SIZE);
    TPUKERNEL_DBG("grad in ping local addr = 0x%5x, bank id = %d\n", grad_in_ping, grad_in_ping / BANK_SIZE);
    TPUKERNEL_DBG("grad in pong local addr = 0x%5x, bank id = %d\n", grad_in_pong, grad_in_pong / BANK_SIZE);

    global_addr_t input_global_addr = grad_out_global_addr;
    global_addr_t output_global_addr = grad_in_global_addr;

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
                    dim4 oslice_shape = {nslice, cslice, ohslice, owslice};
                    // copy forward input from global
                    tpu_gdma_cpy_S2L(
                        ping ? forward_in_ping : forward_in_pong,
                        forward_in_global_addr +
                            (nstart * ostride.n +
                             cstart * ostride.c +
                             ohstart * ostride.h +
                             owstart) * tpu_data_type_size(dtype),
                        &oslice_shape,
                        NULL,
                        &ostride,
                        dtype);

                    // copy forward output from global
                    tpu_gdma_cpy_S2L(
                        ping ? forward_out_ping : forward_out_pong,
                        forward_out_global_addr +
                            (nstart * istride.n +
                             cstart * istride.c +
                             ihstart * istride.h +
                             iwstart) * tpu_data_type_size(dtype),
                        &islice_shape,
                        NULL,
                        &istride,
                        dtype);

                    // copy grad out from global
                    tpu_gdma_cpy_S2L(
                        ping ? grad_out_ping : grad_out_pong,
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

                    // stride == kernel use depthwise deconv
                    dim4 index_shape = {1, 1, oslice_shape.h, oslice_shape.w};
                    tpu_bdc_arithmetic_sequence_bcast(
                        index_addr,
                        1,
                        oslice_shape.h * oslice_shape.w - 1,
                        -1,
                        oslice_shape.h * oslice_shape.w);

                    tpu_bdc_cast(
                        index_addr,
                        index_addr,
                        &index_shape,
                        NULL,
                        NULL,
                        dtype,
                        DT_INT32,
                        RM_HALF_AWAY_FROM_ZERO);

                    scalar_t kernel_const = {.f32 = 1.0f};
                    tpu_bdc_fp_depthwise_for_deconv2d(
                        ping ? grad_in_ping : grad_in_pong,
                        ping ? forward_out_ping : forward_out_pong,
                        tpu_cast(kernel_const, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO).u32,
                        0,
                        &islice_shape,
                        kernel,
                        &insert,
                        &slice_pad,
                        dilation,
                        dtype,
                        dtype,
                        true,
                        false);

                    scalar_t one = {.f32 = 1.0f};
                    tpu_bdc_equal(
                        ping ? forward_in_ping : forward_in_pong,
                        ping ? grad_in_ping : grad_in_pong,
                        ping ? forward_in_ping : forward_in_pong,
                        tpu_cast(one, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                        &oslice_shape,
                        NULL,
                        NULL,
                        NULL,
                        dtype,
                        dtype);

                    dim4 index_stride = {0, 0, index_shape.w, 1};
                    tpu_bdc_fp_mul(
                        ping ? forward_in_ping : forward_in_pong,
                        ping ? forward_in_ping : forward_in_pong,
                        index_addr,
                        &oslice_shape,
                        NULL,
                        NULL,
                        &index_stride,
                        dtype);

                    scalar_t pad_value = {.u32 = FP_NEG_MAX(dtype)};
                    tpu_bdc_fp_max_pool2d(
                        ping ? forward_out_ping : forward_out_pong,
                        ping ? forward_in_ping : forward_in_pong,
                        &oslice_shape,
                        kernel,
                        padding,
                        stride,
                        dilation,
                        dtype,
                        pad_value);

                    tpu_bdc_fp_depthwise_for_deconv2d(
                        ping ? forward_in_ping : forward_in_pong,
                        ping ? forward_out_ping : forward_out_pong,
                        tpu_cast(kernel_const, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO).u32,
                        0,
                        &islice_shape,
                        kernel,
                        &insert,
                        &slice_pad,
                        dilation,
                        dtype,
                        dtype,
                        true,
                        false);

                    tpu_bdc_equal(
                        ping ? forward_in_ping : forward_in_pong,
                        ping ? forward_in_ping : forward_in_pong,
                        index_addr,
                        tpu_cast(one, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                        &oslice_shape,
                        NULL,
                        NULL,
                        &index_stride,
                        dtype,
                        dtype);

                    tpu_bdc_fp_depthwise_for_deconv2d(
                        ping ? grad_in_ping : grad_in_pong,
                        ping ? grad_out_ping : grad_out_pong,
                        tpu_cast(kernel_const, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO).u32,
                        0,
                        &islice_shape,
                        kernel,
                        &insert,
                        &slice_pad,
                        dilation,
                        dtype,
                        dtype,
                        true,
                        false);

                    tpu_bdc_fp_mul(
                        ping ? grad_in_ping : grad_in_pong,
                        ping ? forward_in_ping : forward_in_pong,
                        ping ? grad_in_ping : grad_in_pong,
                        &oslice_shape,
                        NULL,
                        NULL,
                        NULL,
                        dtype);

                    // move output to global memory
                    if (nidx > 0 || cidx > 0 || ohidx > 0 || owidx > 0) {
                        dim4 last_oslice_shape = {last_nslice, last_cslice, last_ohslice, last_owslice};
                        tpu_gdma_cpy_L2S(
                            output_global_addr +
                                (last_nstart * ostride.n +
                                 last_cstart * ostride.c +
                                 last_ohstart * ostride.h +
                                 last_owstart) * tpu_data_type_size(dtype),
                            ping ? grad_in_pong : grad_in_ping,
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
        ping ? grad_in_pong : grad_in_ping,
        &last_oslice_shape,
        &ostride,
        NULL,
        dtype);
}

void nodechip_maxpool_backward_stride_less_kernel(
    global_addr_t    forward_in_global_addr,
    global_addr_t    forward_out_global_addr,
    global_addr_t    grad_out_global_addr,
    global_addr_t    grad_in_global_addr,
    const dim4       *grad_input_shape,
    const dim4       *grad_out_shape,
    const dim2       *kernel,
    const dim2       *stride,
    const padding_t  *padding,
    const dim2       *dilation,
    bool             ceil_mode,
    data_type_t      dtype
  ) {

    const int kh = kernel->h;
    const int kw = kernel->w;
    const int dh = dilation->h;
    const int dw = dilation->w;
    const int stride_h = stride->h;
    const int stride_w = stride->w;

    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;

    const int n = grad_input_shape->n;
    const int c = grad_input_shape->c;
    const int ih = grad_input_shape->h;
    const int iw = grad_input_shape->w;
    const int oh = grad_out_shape->h;
    const int ow = grad_out_shape->w;

    const dim4 ishape = {n, c, ih, iw};
    const dim4 oshape = {n, c, oh, ow};

    dim4 istride, ostride;
    tpu_continuous_stride(&istride, &ishape);
    tpu_continuous_stride(&ostride, &oshape);

    padding_t forward_pad = {padding->top,
                             padding->bottom,
                             padding->left,
                             padding->right};

    maxpool_secs_info_t secs_info;
    maxpool_data_split(
        n, c, ih, iw, oh, ow,
        kh, kw, dh, dw,
        stride_h, stride_w,
        forward_pad, dtype, &secs_info);

    int nslice = DIV_UP(n, secs_info.nsecs);
    int cslice = DIV_UP(c, secs_info.csecs);
    cslice = secs_info.csecs > 1 ? ALIGN(cslice, NPU_NUM) : cslice;
    int ohslice = DIV_UP(oh, secs_info.ohsecs);
    int owslice = DIV_UP(ow, secs_info.owsecs);
    int ihslice = MIN((ohslice - 1) * stride->h + kh_ext, ih);
    int iwslice = MIN((owslice - 1) * stride->w + kw_ext, iw);
    TPUKERNEL_DBG("nsecs=%d  nslice=%d\n", secs_info.nsecs, nslice);
    TPUKERNEL_DBG("csecs=%d  cslice=%d\n", secs_info.csecs, cslice);
    TPUKERNEL_DBG("ohsecs=%d  ohslice=%d\n", secs_info.ohsecs, ohslice);
    TPUKERNEL_DBG("owsecs=%d  owslice=%d\n", secs_info.owsecs, owslice);

    int ihslice_with_pad = ihslice + padding->top + padding->bottom;
    int iwslice_with_pad = iwslice + padding->left + padding->right;
    unsigned int index_size = tpu_aligned_feature_size(ihslice_with_pad, iwslice_with_pad, DT_INT32);
    unsigned int isize = nslice * DIV_UP(cslice, NPU_NUM) *
                         tpu_aligned_feature_size(ihslice_with_pad, iwslice_with_pad, dtype);
    unsigned int osize = nslice * DIV_UP(cslice, NPU_NUM) *
                         tpu_aligned_feature_size(ohslice, owslice, dtype);

    local_addr_t index_addr = 0;
    local_addr_t indices_addr = ALIGN(index_addr + index_size, BANK_SIZE);
    local_addr_t forward_in_ping = ALIGN(indices_addr + osize, BANK_SIZE);
    local_addr_t forward_in_pong = ALIGN(forward_in_ping + isize, BANK_SIZE);
    local_addr_t forward_out_ping = ALIGN(forward_in_pong + isize, BANK_SIZE);
    local_addr_t forward_out_pong = ALIGN(forward_out_ping + osize, BANK_SIZE);
    local_addr_t grad_out_ping = ALIGN(forward_out_pong + osize, BANK_SIZE);
    local_addr_t grad_out_pong = ALIGN(grad_out_ping + osize, BANK_SIZE);
    local_addr_t grad_in_ping = ALIGN(grad_out_pong + osize, BANK_SIZE);
    local_addr_t grad_in_pong = ALIGN(grad_in_ping + isize, BANK_SIZE);
    TPUKERNEL_ASSERT(grad_in_pong + isize <= (unsigned int)LOCAL_MEM_SIZE);

    global_addr_t input_global_addr = grad_out_global_addr;
    global_addr_t output_global_addr = grad_in_global_addr;

    bool ping = true;
    int last_nstart = 0;
    int last_nslice = 0;
    int last_cstart = 0;
    int last_cslice = 0;
    int last_ihstart = 0;
    int last_ihslice = 0;
    int last_ihend = 0;
    int last_iwstart = 0;
    int last_iwslice = 0;
    int last_iwend = 0;
    int last_slice_pad_top = 0;
    int last_slice_pad_bottom = 0;
    int last_slice_pad_left = 0;
    int last_slice_pad_right = 0;

    bool parallel_branch = false;
    int nstart = 0;
    int nend = 0;
    // 1th loop: split N dim
    for (int nidx = 0; nidx < secs_info.nsecs; nidx++) {
        nstart = nend;
        nslice = n / secs_info.nsecs + ((n % secs_info.nsecs) > nidx);
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
            // 4th loop: split OH dim
            for (int ohidx = 0; ohidx < secs_info.ohsecs; ohidx++) {
                // oh boundary: [ohstart, ohend)
                // ih boundary: [ihstart, ihend)
                ohstart = ohend;
                ohslice = oh / secs_info.ohsecs + ((oh % secs_info.ohsecs) > ohidx);
                ohend = ohstart + ohslice;

                int ihstart = ohstart * stride->h - padding->top;
                int ihend = (ohend - 1) * stride->h + kh_ext - padding->top;
                slice_pad.top = ihstart < 0 ? -ihstart : 0;
                slice_pad.bottom = ihend > ih ? ihend - ih : 0;
                ihstart = ihstart < 0 ? 0 : ihstart;
                ihend = ihend > ih ? ih : ihend;
                ihslice = ihend - ihstart;

                int owstart = 0;
                int owend = 0;
                // 5th loop: split OW dim
                for (int owidx = 0; owidx < secs_info.owsecs; owidx++) {
                    // ow boundary: [owstart, owend)
                    // iw boundary: [iwstart, iwend)
                    owstart = owend;
                    owslice = ow / secs_info.owsecs + ((ow % secs_info.owsecs) > owidx);
                    owend = owstart + owslice;
                    int iwstart = owstart * stride->w - padding->left;
                    int iwend = (owend - 1) * stride->w + kw_ext - padding->left;
                    slice_pad.left = iwstart < 0 ? -iwstart : 0;
                    slice_pad.right = iwend > iw ? iwend - iw : 0;
                    iwstart = iwstart < 0 ? 0 : iwstart;
                    iwend = iwend > iw ? iw : iwend;
                    iwslice = iwend - iwstart;

                    dim4 islice_shape = {nslice, cslice, ihslice, iwslice};
                    dim4 oslice_shape = {nslice, cslice, ohslice, owslice};
                    dim4 islice_shape_withpad = {nslice, cslice,
                                                 ihslice + slice_pad.top + slice_pad.bottom,
                                                 iwslice + slice_pad.left + slice_pad.right};

                    dim4 islice_stride_withpad;
                    tpu_aligned_stride(&islice_stride_withpad, 0, &islice_shape_withpad, dtype);

                    // move forward_input、forward_output、grad_out in
                    // copy forward input from_global
                    scalar_t neg_max = {.u32 = FP_NEG_MAX(dtype)};
                    tpu_bdc_set_C(
                        ping ? forward_in_ping : forward_in_pong,
                        neg_max,
                        &islice_shape_withpad,
                        NULL,
                        dtype);
                    tpu_gdma_cpy_S2L(
                        ping ? forward_in_ping +
                                   (slice_pad.top * islice_shape_withpad.w + slice_pad.left) * tpu_data_type_size(dtype)
                             : forward_in_pong +
                                   (slice_pad.top * islice_shape_withpad.w + slice_pad.left) * tpu_data_type_size(dtype),
                        forward_in_global_addr +
                            (nstart * istride.n +
                             cstart * istride.c +
                             ihstart * istride.h + iwstart) * tpu_data_type_size(dtype),
                        &islice_shape,
                        &islice_stride_withpad,
                        &istride,
                        dtype);

                    // copy forward output from global
                    tpu_gdma_cpy_S2L(
                        ping ? forward_out_ping : forward_out_pong,
                        forward_out_global_addr +
                            (nstart * ostride.n +
                             cstart * ostride.c +
                             ohstart * ostride.h + owstart) * tpu_data_type_size(dtype),
                        &oslice_shape,
                        NULL,
                        &ostride,
                        dtype);

                    // copy grad out from global
                    tpu_gdma_cpy_S2L(
                        ping ? grad_out_ping : grad_out_pong,
                        input_global_addr +
                            (nstart * ostride.n +
                             cstart * ostride.c +
                             ohstart * ostride.h + owstart) * tpu_data_type_size(dtype),
                        &oslice_shape,
                        NULL,
                        &ostride,
                        dtype);

                    if (parallel_branch) {
                        tpu_parallel_end();
                    }
                    tpu_parallel_start();
                    parallel_branch = true;

                    // cal
                    // gen_idx
                    scalar_t zero = {.f32 = 0.0f};
                    dim4 index_shape = {1, 1, islice_shape.h, islice_shape.w};
                    dim4 index_shape_withpad = {1, 1,
                                                islice_shape_withpad.h,
                                                islice_shape_withpad.w};

                    // gen in index_addr, cast to grad_in_ping/pong, move out index_addr
                    tpu_bdc_arithmetic_sequence_bcast(
                        index_addr,
                        1,
                        1,
                        1,
                        index_shape.h * index_shape.w);

                    tpu_bdc_cast(
                        ping ? grad_in_ping : grad_in_pong,
                        index_addr,
                        &index_shape,
                        NULL,
                        NULL,
                        dtype,
                        DT_INT32,
                        RM_HALF_AWAY_FROM_ZERO);

                    tpu_bdc_set_C(
                        index_addr,
                        tpu_cast(zero, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                        &index_shape_withpad,
                        NULL,
                        dtype);

                    if (slice_pad.left == 0 && slice_pad.right == 0) {
                        tpu_bdc_cpy(
                            index_addr +
                                slice_pad.top * index_shape_withpad.w * tpu_data_type_size(dtype),
                            ping ? grad_in_ping : grad_in_pong,
                            &index_shape,
                            NULL,
                            NULL,
                            dtype);
                    } else {
                        dim4 index_stride = {1, 1, index_shape_withpad.w, 1};
                        tpu_bdc_cpy(
                            index_addr +
                                (slice_pad.top * index_shape_withpad.w + slice_pad.left) * tpu_data_type_size(dtype),
                            ping ? grad_in_ping : grad_in_pong,
                            &index_shape,
                            &index_stride,
                            NULL,
                            dtype);
                    }

                    tpu_bdc_set_C(
                        indices_addr,
                        tpu_cast(zero, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                        &oslice_shape,
                        NULL,
                        dtype);

                    dim4 per_stride;
                    tpu_aligned_stride(&per_stride, 0, &islice_shape_withpad, dtype);
                    per_stride.h = islice_shape_withpad.w * stride_h;
                    per_stride.w = stride_w;

                    dim4 index_per_stride = {0, 0, per_stride.h, per_stride.w};

                    scalar_t one = {.f32 = 1.0f};
                    for (int index = 0; index < kh * kw; index++) {
                        local_addr_t index_offset = ((index / kw) * islice_shape_withpad.w + index % kw) * tpu_data_type_size(dtype);
                        tpu_bdc_equal(
                            ping ? grad_in_ping : grad_in_pong,
                            ping ? forward_in_ping + index_offset
                                 : forward_in_pong + index_offset,
                            ping ? forward_out_ping : forward_out_pong,
                            dtype == DT_FP32 ? one : tpu_cast(one, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                            &oslice_shape,
                            NULL,
                            &per_stride,
                            NULL,
                            dtype,
                            dtype);
                        tpu_bdc_fp_mul(
                            ping ? grad_in_ping : grad_in_pong,
                            index_addr + index_offset,
                            ping ? grad_in_ping : grad_in_pong,
                            &oslice_shape,
                            NULL,
                            &index_per_stride,
                            NULL,
                            dtype);
                        // R = A == B ? C : D
                        // indices = buffer == 0 ? indices : buffer
                        variable_t indices_var = {
                            .type = TENSOR,
                            .context.addr = indices_addr,
                        };
                        variable_t grad_in_var = {
                            .type = TENSOR,
                            .context.addr = ping ? grad_in_ping : grad_in_pong,
                        };
                        variable_t zero_var = {
                            .type = SCALAR,
                            .context.scalar.u32 = 0,
                        };
                        tpu_bdc_equal_select(
                            indices_addr,
                            &indices_var,
                            &zero_var,
                            &grad_in_var,
                            &indices_var,
                            &oslice_shape,
                            dtype,
                            dtype);
                    }//gen index
                    tpu_bdc_set_C(
                        ping ? grad_in_ping : grad_in_pong,
                        tpu_cast(zero, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                        &islice_shape_withpad,
                        NULL,
                        dtype);

                    if (ihstart < last_ihend || iwstart < last_iwend) {
                        dim4 overlap_shape = {nslice, cslice, last_ihend - ihstart, last_iwend - iwstart};
                        //dim4 last_islice_shape = {nslice, cslice, last_ihslice, last_iwslice};
                        dim4 last_islice_shape_withpad = {nslice, cslice,
                                                          last_ihslice + last_slice_pad_top + last_slice_pad_bottom,
                                                          last_iwslice + last_slice_pad_left + last_slice_pad_right};
                        dim4 last_islice_stride_withpad;
                        tpu_aligned_stride(&last_islice_stride_withpad, 0, &last_islice_shape_withpad, dtype);
                        tpu_bdc_cpy(
                            ping ? grad_in_ping + (slice_pad.top * islice_shape_withpad.w + slice_pad.left) * tpu_data_type_size(dtype)
                                 : grad_in_pong + (slice_pad.top * islice_shape_withpad.w + slice_pad.left) * tpu_data_type_size(dtype),
                            ping ? grad_in_pong + ((last_slice_pad_top + ihstart - last_ihstart) * last_islice_shape_withpad.w + last_slice_pad_left + iwstart - last_iwstart) * tpu_data_type_size(dtype)
                                 : grad_in_ping + ((last_slice_pad_top + ihstart - last_ihstart) * last_islice_shape_withpad.w + last_slice_pad_left + iwstart - last_iwstart) * tpu_data_type_size(dtype),
                            &overlap_shape,
                            &islice_stride_withpad,
                            &last_islice_stride_withpad,
                            dtype);
                    }

                    //dim4 grad_in_stride;
                    //tpu_compact_stride(&grad_in_stride, 0, &islice_shape);
                    //grad_in_stride.h = islice_shape.w * stride_h;
                    //grad_in_stride.w = stride_w;
                    for (int index = 0; index < kh * kw; index++) {
                        local_addr_t index_offset = ((index / kw) * islice_shape_withpad.w + index % kw) * tpu_data_type_size(dtype);
                        //local_addr_t index_offset_without_pad = ((index / kw) * islice_shape.w + index % kw) * tpu_data_type_size(dtype);
                        tpu_bdc_equal(
                            ping ? forward_out_ping : forward_out_pong,
                            index_addr + index_offset,
                            indices_addr,
                            tpu_cast(one, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                            &oslice_shape,
                            NULL,
                            &index_per_stride,
                            NULL,
                            dtype,
                            dtype);
                        if (dtype == (data_type_t)FP32) {
                            tpu_bdc_fp32_mac(
                                ping ? grad_in_ping + index_offset :
                                       grad_in_pong + index_offset,
                                ping ? forward_out_ping : forward_out_pong,
                                ping ? grad_out_ping : grad_out_pong,
                                &oslice_shape,
                                &per_stride,
                                NULL,
                                NULL);
                        } else {
                            tpu_bdc_fp_mul(
                                ping ? forward_out_ping : forward_out_pong,
                                ping ? forward_out_ping : forward_out_pong,
                                ping ? grad_out_ping : grad_out_pong,
                                &oslice_shape,
                                NULL,
                                NULL,
                                NULL,
                                dtype);
                            tpu_bdc_fp_add(
                                ping ? grad_in_ping + index_offset :
                                       grad_in_pong + index_offset,
                                ping ? grad_in_ping + index_offset :
                                       grad_in_pong + index_offset,
                                ping ? forward_out_ping : forward_out_pong,
                                &oslice_shape,
                                &per_stride,
                                &per_stride,
                                NULL,
                                dtype);
                        }
                    }

                    // move output
                    // move output to global memory
                    if (nidx > 0 || cidx > 0 || ohidx > 0 || owidx > 0) {
                        dim4 last_islice_shape = {last_nslice, last_cslice, last_ihslice, last_iwslice};
                        dim4 last_islice_shape_withpad = {last_nslice, last_cslice, last_ihslice + last_slice_pad_top + last_slice_pad_bottom, last_iwslice + last_slice_pad_left + last_slice_pad_right};
                        dim4 last_grad_in_stride;
                        tpu_aligned_stride(&last_grad_in_stride,
                                           0,
                                           &last_islice_shape_withpad,
                                           dtype);
                        tpu_gdma_cpy_L2S(
                            output_global_addr +
                                (last_nstart * istride.n +
                                 last_cstart * istride.c +
                                 last_ihstart * istride.h + last_iwstart) * tpu_data_type_size(dtype),
                            ping ? grad_in_pong + (last_slice_pad_top * last_islice_shape_withpad.w + last_slice_pad_left) * tpu_data_type_size(dtype)
                                 : grad_in_ping + (last_slice_pad_top * last_islice_shape_withpad.w + last_slice_pad_left) * tpu_data_type_size(dtype),
                            &last_islice_shape,
                            &istride,
                            (last_slice_pad_left == 0 && last_slice_pad_right == 0) ? NULL : &last_grad_in_stride,
                            dtype);
                    }
                    ping = !ping;
                    // save current info used for moving output to global memory next loop
                    last_nstart = nstart;
                    last_nslice = nslice;
                    last_cstart = cstart;
                    last_cslice = cslice;
                    last_ihstart = ihstart;
                    last_ihend = ihend;
                    last_ihslice = ihslice;
                    last_iwstart = iwstart;
                    last_iwslice = iwslice;
                    last_iwend = iwend;
                    last_slice_pad_top = slice_pad.top;
                    last_slice_pad_bottom = slice_pad.bottom;
                    last_slice_pad_left = slice_pad.left;
                    last_slice_pad_right = slice_pad.right;
                }
            }
        }
    }
    tpu_parallel_end();
    // move the last output to global memory
    dim4 last_islice_shape = {last_nslice, last_cslice, last_ihslice, last_iwslice};
    dim4 last_islice_shape_withpad = {last_nslice, last_cslice, last_ihslice + last_slice_pad_top + last_slice_pad_bottom, last_iwslice + last_slice_pad_left + last_slice_pad_right};
    dim4 last_grad_in_stride;
    tpu_aligned_stride(&last_grad_in_stride,
                       0,
                       &last_islice_shape_withpad,
                       dtype);
    tpu_gdma_cpy_L2S(
        output_global_addr +
            (last_nstart * istride.n +
             last_cstart * istride.c +
             last_ihstart * istride.h + last_iwstart) * tpu_data_type_size(dtype),
        ping ? grad_in_pong + (last_slice_pad_top * last_islice_shape_withpad.w + last_slice_pad_left) * tpu_data_type_size(dtype)
             : grad_in_ping + (last_slice_pad_top * last_islice_shape_withpad.w + last_slice_pad_left) * tpu_data_type_size(dtype),
        &last_islice_shape,
        &istride,
        (last_slice_pad_left == 0 && last_slice_pad_right == 0) ? NULL : &last_grad_in_stride,
        dtype);
}

void nodechip_maxpool_backward(
    global_addr_t    forward_in_global_addr,
    global_addr_t    forward_out_global_addr,
    global_addr_t    grad_out_global_addr,
    global_addr_t    grad_in_global_addr,
    const dim4       *grad_input_shape,
    const dim4       *grad_out_shape,
    const dim2       *kernel,
    const dim2       *stride,
    const padding_t  *padding,
    const dim2       *dilation,
    bool             ceil_mode,
    data_type_t      dtype
  ) {

    int kh = kernel->h;
    int kw = kernel->w;
    int stride_h = stride->h;
    int stride_w = stride->w;
    bool stride_less_kernel = stride_h < kh || stride_w < kw;

    if (stride_less_kernel) {
        nodechip_maxpool_backward_stride_less_kernel(
            forward_in_global_addr,
            forward_out_global_addr,
            grad_out_global_addr,
            grad_in_global_addr,
            grad_input_shape,
            grad_out_shape,
            kernel,
            stride,
            padding,
            dilation,
            ceil_mode,
            dtype);
    } else {
        nodechip_maxpool_backward_stride_noless_kernel(
            forward_in_global_addr,
            forward_out_global_addr,
            grad_out_global_addr,
            grad_in_global_addr,
            grad_input_shape,
            grad_out_shape,
            kernel,
            stride,
            padding,
            dilation,
            ceil_mode,
            dtype);
    }
}

void tpu_kernel_api_maxpool_backward(const void* args) {
    sg_api_maxpool_backward_t* api = (sg_api_maxpool_backward_t*)args;

    dim4 input_shape = {api->input_shape[0], api->input_shape[1],
                        api->input_shape[2], api->input_shape[3]};
    dim4 output_shape = {api->output_shape[0], api->output_shape[1],
                         api->output_shape[2], api->output_shape[3]};
    dim2 kernel = {api->kernel[0], api->kernel[1]};
    dim2 stride = {api->stride[0], api->stride[1]};
    padding_t pad = {api->pad[0], api->pad[0], api->pad[1], api->pad[1]};
    dim2 dilation = {api->dilation[0], api->dilation[1]};

    tpu_initialize();
    nodechip_maxpool_backward(
        api->forward_input_global_addr,
        api->forward_output_global_addr,
        api->grad_output_global_addr,
        api->grad_input_global_addr,
        &input_shape,
        &output_shape,
        &kernel,
        &stride,
        &pad,
        &dilation,
        api->ceil_mode == 1 ? true : false,
        tpu_type_convert(api->dtype));
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_maxpool_backward);
