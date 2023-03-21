#include "nodechip_batch_matmul.h"
#include "sg_api_struct.h"

void tpu_kernel_api_batch_matmul(const void *args)
{
    sg_api_batch_matmul_t *api = (sg_api_batch_matmul_t *)args;
    tpu_initialize();
    
    int L_shape[] = {api->batch_num, api->L_row_num, api->L_col_num};
    int R_shape[] = {api->batch_num, api->L_col_num, api->R_col_num};
    
    nodechip_batch_matmul_float(
        api->L_addr,
        api->R_addr,
        0,
        api->Y_addr,
        tpu_type_convert(api->L_dtype),
        tpu_type_convert(api->Y_dtype),
        L_shape,
        R_shape,
        3,
        3,
        0,
        0,
        api->L_trans,
        api->R_trans,
        0,
        0,
        0,
        0);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_batch_matmul);
