#include "tpu_kernel.h"
#include <stddef.h>
#include <math.h>

void nodechip_reduce_onedim_local(
local_addr_t input_local_addr,
local_addr_t buffer_local_addr,
local_addr_t output_local_addr,
const dim4 *ishape,
int method,
bool is_reduce_h, // true: reduce h, false: reduce w
local_addr_t cast_buff, //used when reduce_l2 & dtype isn't fp32
data_type_t dtype) {
    const int eu_num = tpu_eu_num(dtype);

    if (method == REDUCE_PROD) {
        int c_per_npu = DIV_UP(ishape->c, tpu_npu_num());
        dim4 cpy_dim = {ishape->n, ishape->c, 1, ishape->w};
        dim4 src_str = {ALIGN(ishape->h * ishape->w, eu_num) * c_per_npu,
                        ALIGN(ishape->h,ishape->w),
                        ishape->w,
                        1};
        dim4 dst_str = {ALIGN(ishape->w, eu_num) * c_per_npu,
                        ALIGN(ishape->w, eu_num),
                        ishape->w,
                        1};
        if (is_reduce_h) {
            for(int i = 0; i<ishape->h;++i) {
                if (i == 0) {
                    tpu_bdc_cpy(output_local_addr,
                                input_local_addr,
                                &cpy_dim,
                                &dst_str,
                                &src_str,
                                dtype);
                }
                else {
                    tpu_bdc_fp_mul(output_local_addr,
                                   input_local_addr + i * ishape->w * tpu_data_type_size(dtype),
                                   output_local_addr,
                                   &cpy_dim,
                                   &dst_str,
                                   &src_str,
                                   &dst_str,
                                   dtype);
                }
            }
        }
        else {
        int head = pow(2, (int)log2(ishape->w));
        int tail = ishape->w - head;
        TPUKERNEL_ASSERT(head > tail);
        if (tail > 0) {
            cpy_dim.h = ishape->h;
            cpy_dim.w = tail;
            dst_str.h = head;
            dst_str.c = ALIGN(dst_str.h * cpy_dim.h, eu_num);
            dst_str.n = c_per_npu * dst_str.c;
            tpu_bdc_fp_mul(buffer_local_addr,
                           input_local_addr,
                           input_local_addr + head * tpu_data_type_size(dtype),
                           &cpy_dim,
                           &dst_str,
                           &src_str,
                           &src_str,
                           dtype);
            cpy_dim.w = head - tail;
            tpu_bdc_cpy(buffer_local_addr + tail * tpu_data_type_size(dtype),
                        input_local_addr + tail * tpu_data_type_size(dtype),
                        &cpy_dim,
                        &dst_str,
                        &src_str,
                        dtype);
            input_local_addr = buffer_local_addr;
        }

        // process head, and we keep dst stride == src stride, and then copy to output addr
        src_str.h = head;
        src_str.c = ALIGN(src_str.h * ishape->h, eu_num);
        src_str.n = c_per_npu * src_str.c;
        int left = head;
        while(left > 1) {
            dim4 prod_dim = { ishape->n, ishape->c, ishape->h, left >> 1 };
            tpu_bdc_fp_mul(buffer_local_addr,
                           head == left ? input_local_addr : buffer_local_addr,
                           (head == left ? input_local_addr : buffer_local_addr) + (left >> 1) * tpu_data_type_size(dtype),
                           &prod_dim,
                           &src_str,
                           &src_str,
                           &src_str,
                           dtype);
            left = left >> 1;
        }
        // copy data from buffer to output
        cpy_dim.h = ishape->h;
        cpy_dim.w = 1;
        dst_str.w = 1;
        dst_str.h = 1;
        dst_str.c = ALIGN(cpy_dim.h * dst_str.h, eu_num);
        dst_str.n = c_per_npu * dst_str.c;
        tpu_bdc_cpy(output_local_addr,
                    head == 1 ? input_local_addr : buffer_local_addr,
                    &cpy_dim,
                    &dst_str,
                    &src_str,
                    dtype);
        }
    }
}