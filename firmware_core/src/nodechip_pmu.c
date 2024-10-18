#include "sg_api_struct.h"
#include "tpu_kernel.h"

#ifdef BACKEND_SG2260

extern void disable_tpu_perf_monitor(void);

extern void enable_tpu_perf_monitor(void);

int tpu_kernel_pmu(const void *api_buf) {
    sg_api_pmu_t *api = (sg_api_pmu_t *)api_buf;
    if(api->enable) {
        enable_tpu_perf_monitor();
    } else {
        disable_tpu_perf_monitor();
    }
    return 0;
}
#endif