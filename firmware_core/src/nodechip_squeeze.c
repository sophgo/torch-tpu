#include "sg_api_struct.h"
#include "tpu_kernel.h"

inline static void pipeline_move(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

/**
 * output = squeeze(input, dim)
 */
void nodechip_tile_1d_simulate(global_addr_t input_global_addr,
                      global_addr_t output_global_addr, const int *input_shape,
                      int input_dims, int tile_axis, int tile_num, int type,
                      data_type_t dtype) {
  TPUKERNEL_ASSERT(0 < input_dims && input_dims <= FW_MAX_SHAPE_DIMS);
  for (int i = 0; i < input_dims; i++)
    TPUKERNEL_ASSERT(input_shape[i] > 0);
  TPUKERNEL_ASSERT(0 <= tile_axis && tile_axis < input_dims);
  TPUKERNEL_ASSERT(tile_num > 0);
  u64 H_ = 1, W_ = 1;
  for (int i = 0; i < tile_axis + (type != 0); ++i) {
    H_ *= input_shape[i];
  }
  for (int i = tile_axis + (type != 0); i < input_dims; ++i) {
    W_ *= input_shape[i];
  }
  // TODO: support larger shape
  TPUKERNEL_ASSERT(H_ <= INT32_MAX);
  TPUKERNEL_ASSERT(W_ <= INT32_MAX);
  const int H = (int)H_;
  const int W = (int)W_;
  const dim4 in_stride = {H * W, W, 0, 1};
  const dim4 shape = {1, H, tile_num, W};
  tpu_gdma_cpy_S2S(input_global_addr, output_global_addr, &shape, NULL,
                   &in_stride, dtype);
}

void tpu_kernel_api_squeeze(const void *args) {
  sg_api_squeeze_t *api = (sg_api_squeeze_t *)args;

  unsigned long long length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  tpu_initialize();
  nodechip_tile_1d_simulate(api->output_global_addr, api->input_global_addr, api->shape,
                   api->dim, 0, 1, 0, (data_type_t)api->dtype);
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_squeeze);

void tpu_kernel_api_squeeze_multi_core(const void *args) {
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_squeeze_multi_core);