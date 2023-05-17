#include "nodechip_fc.h"
#include "sg_api_struct.h"

void tpu_kernel_api_general_matmul(const void *args)
{
    sg_api_general_matmul_t *api = (sg_api_general_matmul_t *)args;

    tpu_initialize();
    general_matmul(
            api->L_global_addr,
            api->R_global_addr,
            api->bias_global_addr,
            api->Y_global_addr,
            1,
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
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_general_matmul);