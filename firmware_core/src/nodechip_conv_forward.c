#include "sg_api_struct.h"
#include "tpu_kernel.h"

typedef struct nnvlc_common_spec {
  int32_t do_compress;
  int32_t do_decompress;
  int32_t bias0;
  int32_t bias1;
  int32_t zero_guard;
} nnvlc_common_spec_t;

extern void nodechip_conv_float_parallel (
global_addr_t       input_global_addr,
global_addr_t       weight_global_addr,
global_addr_t       bias_global_addr,
global_addr_t       output_global_addr,
const dim4         *ishape,
int                 groups,
int                 output_c,
const dim2         *kernel,
const dim2         *stride,
const dim2         *dilation,
const padding_t    *pad,
bool                has_bias,
bool                if_relu,
float               upper_limit,
bool                result_add,
data_type_t         idtype,
data_type_t         odtype,
bool                reshaped_bias );

extern void nodechip_conv_float_multi_core(
global_addr_t       input_global_addr,
global_addr_t       weight_global_addr,
global_addr_t       bias_global_addr,
global_addr_t       rescale_global_addr,
global_addr_t       output_global_addr,
const dim4         *ishape,
const dim4         *merge_wstride,
int                 groups,
int                 output_c,
const dim2         *kernel,
const dim2         *stride,
const dim2         *dilation,
const padding_t    *pad,
bool                has_bias,
bool                if_relu,
float               upper_limit,
bool                result_add,
bool                has_rescale,
data_type_t         idtype,
data_type_t         wdtype,
data_type_t         bdtype,
data_type_t         odtype,
bool                reshaped_bias,
int                 merge_coeff,
int                 weight_is_coeff,
int                 use_3ic_optimize,
nnvlc_common_spec_t nnvlc_param);

int tpu_kernel_api_conv2d_multi_core ( const void * args ) {
  sg_api_conv2d_t * api = ( sg_api_conv2d_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  dim4 input_shape =
  {
    api->input_shape[0], api->input_shape[1], api->input_shape[2], api->input_shape[3]
  };
  dim2 kernel = { api->kernel[0], api->kernel[1] };
  dim2 stride = { api->stride[0], api->stride[1] };
  dim2 dilation = { api->dilation[0], api->dilation[1] };
  padding_t pad = { api->pad[0], api->pad[1], api->pad[2], api->pad[3] };

  nnvlc_common_spec_t nnvlc_param = {false, false, 0, 0, false};
  tpu_initialize();
  nodechip_conv_float_multi_core (
    api->input_global_addr,
    api->weight_global_addr,
    api->bias_global_addr,
    0,
    api->output_global_addr,
    &input_shape,
    NULL,
    api->groups,
    api->output_c,
    &kernel,
    &stride,
    &dilation,
    &pad,
    api->bias_global_addr != 0,
    0,
    -1.f,
    0,
    false,
    ( data_type_t ) api->dtype,
    ( data_type_t ) api->dtype,
    DT_FP32,
    ( data_type_t ) api->dtype,
    false,
    false,
    true,
    0,
    nnvlc_param );
#ifdef BACKEND_SG2260
  tpu_sync_all();
#endif
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_conv2d_multi_core);