#include "nodechip_fc.h"
#include "nodechip_batch_matmul.h"
#include "nodechip_batch_matmul_local.h"
#include "sg_api_struct.h"

inline static int ceiling_func(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}

void tpu_kernel_api_general_matmul(const void *args)
{
    sg_api_general_matmul_t *api = (sg_api_general_matmul_t *)args;

    tpu_initialize();
    int L_row_num = api->L_row_num; 
    int L_col_num = api->L_col_num; 
    int R_col_num = api->R_col_num; 
    int channel = ceiling_func(R_col_num, EU_NUM_16BIT * 2);
    u64 mm_cycle = L_row_num * ceiling_func(channel, NPU_NUM) * 2ull * L_col_num;
    u64 mm2_cycle = (u64)ceiling_func(L_row_num, NPU_NUM) * ceiling_func(R_col_num, 4) * ceiling_func(L_col_num, EU_NUM_16BIT);
        
    if (tpu_type_convert(api->dtype) == DT_FP16 && mm2_cycle <= mm_cycle)
        {
            int L_shape[3] = {1, L_row_num, L_col_num};
            int R_shape[3] = {1, L_col_num, R_col_num};
            if (api->R_transpose) {
                R_shape[1] = R_col_num;
                R_shape[2] = L_col_num;
            }
            nodechip_batch_matmul_float(
                api->L_global_addr,
                api->R_global_addr,
                api->bias_global_addr,
                api->Y_global_addr,
                tpu_type_convert(api->dtype),
                tpu_type_convert(api->dtype),
                L_shape,
                R_shape,
                3, 3,
                0x00, 0x00,
                0,
                api->R_transpose,
                0,
                api->have_bias,
                0,
                0);
        } else {
            nodechip_fc(
                api->L_global_addr,
                api->R_global_addr,
                api->bias_global_addr,
                api->Y_global_addr,
                api->L_row_num,
                api->L_col_num,
                api->R_col_num,
                api->R_transpose,
                api->have_bias,
                tpu_type_convert(api->dtype),
                tpu_type_convert(api->dtype),
                tpu_type_convert(api->dtype),
                0,
                0);
        }
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_general_matmul);
