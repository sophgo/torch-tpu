#include <string.h>
#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_triangularize(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    int* shape,
    int dims,
    int is_upper,
    int diagonal,
    data_type_t dtype
);

int tpu_kernel_api_triangularize_multi_core(const void *args) {
//   TPUKERNEL_ASSERT_INFO(false, "not implementated");
    sg_api_triangularize_t *api = (sg_api_triangularize_t *) args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
    tpu_initialize();
    int shape[FW_MAX_SHAPE_DIMS];
    memcpy(shape, api->shape, api->dims * sizeof(int));
#ifdef BACKEND_SG2260
    int core_idx = tpu_core_index();
    if(core_idx==0){
        nodechip_triangularize(
            api->input_global_addr,
            api->output_global_addr,
            shape,
            api->dims,
            api->is_upper,
            api->diagonal,
            api->dtype
        );
    }
    tpu_poll();
  return 0;
#else
    nodechip_triangularize(
        api->input_global_addr,
        api->output_global_addr,
        shape,
        api->dims,
        api->is_upper,
        api->diagonal,
        api->dtype
    );
    tpu_poll();
    return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_triangularize_multi_core);