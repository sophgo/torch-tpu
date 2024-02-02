#include "sg_api_struct.h"
#include "tpu_kernel.h"


void nodechip_relu_backward (
global_addr_t input_global_addr,
global_addr_t grad_output_global_addr,
global_addr_t grad_input_global_addr,
int eltnum,
data_type_t dtype )
{
  int secs = MIN ( eltnum / NPU_NUM, ( ( tpu_bank_num() / 3 ) * BANK_SIZE ) / tpu_data_type_size ( dtype ) );
  int slice = DIV_UP ( eltnum, secs * NPU_NUM );
  for ( int idx = 0; idx < slice; idx++ )
  {
    int secs_ = idx == slice - 1 ? eltnum / NPU_NUM - idx * secs : secs;
    dim4 newshape = {1, NPU_NUM, 1, secs_};
    int size = tpu_aligned_feature_size ( secs_, 1, dtype );
    local_addr_t input_local_addr = 0;
    local_addr_t grad_output_local_addr = input_local_addr + ALIGN ( size, BANK_SIZE );
    local_addr_t grad_input_local_addr = grad_output_local_addr + ALIGN ( size, BANK_SIZE );
    scalar_t scale = {.f32 = 0.f};
    variable_t input = {
      .type = TENSOR,
      .context.addr = input_local_addr,
    };
    variable_t grad_output = {
      .type = TENSOR,
      .context.addr = grad_output_local_addr,
    };
    variable_t zero = {
      .type = SCALAR,
      .context.scalar = tpu_cast ( scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO ),
    };
    tpu_gdma_cpy_S2L (
    input_local_addr,
    input_global_addr + idx * NPU_NUM * secs * tpu_data_type_size ( dtype ),
    &newshape,
    NULL,
    NULL,
    dtype );
    tpu_gdma_cpy_S2L (
    grad_output_local_addr,
    grad_output_global_addr + idx * NPU_NUM * secs * tpu_data_type_size ( dtype ),
    &newshape,
    NULL,
    NULL,
    dtype );
    tpu_bdc_greater_select (
    grad_input_local_addr,
    &input,
    &zero,
    &grad_output,
    &zero,
    &newshape,
    dtype,
    dtype );
    tpu_gdma_cpy_L2S (
    grad_input_global_addr + idx * NPU_NUM * secs * tpu_data_type_size ( dtype ),
    grad_input_local_addr,
    &newshape,
    NULL,
    NULL,
    dtype );
  }
  return;
}

void tpu_kernel_api_relu_backward ( const void *args )
{
  sg_api_relu_backward_t *api = ( sg_api_relu_backward_t * ) args;
  TPUKERNEL_ASSERT ( api->dtype == DT_FP32 || api->dtype == DT_FP16 || api->dtype == DT_BFP16 );
  int eltnum = 1;
  for ( int i = 0; i < api->dim; ++i )
  {
    eltnum *= api->shape[i];
  }
  tpu_initialize();
  nodechip_relu_backward (
  api->input_global_addr,
  api->grad_output_global_addr,
  api->grad_input_global_addr,
  eltnum,
  ( data_type_t ) api->dtype );
  tpu_poll();
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_relu_backward );
