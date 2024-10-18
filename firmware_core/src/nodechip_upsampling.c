#include "sg_api_struct.h"
#include "tpu_kernel.h"


extern void nodechip_interp_parallel(
    global_addr_t bottom_global_addr, global_addr_t top_global_addr,
    global_addr_t buffer_global_addr,
    unsigned long long
        *buffer_size, // if not NULL, just calculate buffer_size, not compute
    int input_n, int input_c, int input_h, int input_w, int output_h,
    int output_w, int pad_bag, int pad_end, bool align_corners,
    bool half_pixel_centers, PLATFORM_SUPPORT platform_sp, data_type_t dtype);

int tpu_kernel_api_interp_multi_core(const void *args) {
  // todo: use mutli-core
  sg_api_upsampling2d_t *api = (sg_api_upsampling2d_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
  unsigned long long *buffer_size =
      api->if_getting_buffer_size ? api->buffer_size_ptr : 0x0;
  tpu_initialize();
#ifdef BACKEND_SG2260
  int core_idx = tpu_core_index();
  if(core_idx==0)
  {
    nodechip_interp_parallel(
        api->input_global_addr, api->output_global_addr, api->buffer_addr,
        buffer_size, api->shape[0], api->shape[1], api->shape[2], api->shape[3],
        api->out_shape[2], api->out_shape[3], api->pad_bag, api->pad_end,
        api->align_corners, api->half_pixel_centers, api->platform_sp,
        (data_type_t)api->dtype);
  }
  tpu_poll();
  return 0;
#else
  nodechip_interp_parallel(
      api->input_global_addr, api->output_global_addr, api->buffer_addr,
      buffer_size, api->shape[0], api->shape[1], api->shape[2], api->shape[3],
      api->out_shape[2], api->out_shape[3], api->pad_bag, api->pad_end,
      api->align_corners, api->half_pixel_centers, api->platform_sp,
      (data_type_t)api->dtype);
  tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_interp_multi_core);