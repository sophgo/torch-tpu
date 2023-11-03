#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

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

void tpu_kernel_api_conv2d ( const void * args ) {
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
  tpu_initialize();
  nodechip_conv_float_parallel (
  api->input_global_addr,
  api->weight_global_addr,
  api->bias_global_addr,
  api->output_global_addr,
  &input_shape,
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
  ( data_type_t ) api->dtype,
  ( data_type_t ) api->dtype,
  false );
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_conv2d );
