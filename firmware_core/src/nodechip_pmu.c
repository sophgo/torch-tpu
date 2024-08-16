#include "sg_api_struct.h"
#include "tpu_kernel.h"

#ifdef BACKEND_SG2260

extern void disable_tpu_perf_monitor(void);

extern void enable_tpu_perf_monitor(void);

int tpu_kernel_pmu(const void *api_buf) {
    sg_api_pmu_t *api = (sg_api_pmu_t *)api_buf;
    if(api->enable) {
        scalar_t C = {.u32 = 0};
        unsigned int size = 81 * 1024 * 1024;
        tpu_gdma_system_set(tpu_core_index() * size, C, size / 4, DT_UINT32);
        tpu_sync_all();
        tpu_poll();
        // disable first, make sure the counter is reset
        // otherwise pmu data will be all zero
        disable_tpu_perf_monitor();
        enable_tpu_perf_monitor();
    } else {
        disable_tpu_perf_monitor();
    }
    return 0;
}
#endif