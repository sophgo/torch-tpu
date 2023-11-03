#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

extern void nodechip_batch_matmul_float (
global_addr_t L_global_addr,
global_addr_t R_global_addr,
global_addr_t bias_global_addr,
global_addr_t Y_global_addr,
data_type_t in_dtype,
data_type_t out_dtype,
const int* L_shape,
const int* R_shape,
int L_dim,
int R_dim,
int* Y_shape,
int* Y_dim,
int L_trans,
int R_trans,
int hdim_is_batch,
int has_bias,
bool do_relu,
float relu_upper_limit );

extern void nodechip_const_binary_fp(
    global_addr_t A_global_addr,
    global_addr_t res_global_addr,
    const int* shape,
    int shape_dim,
    float B_const_val,
    int inversed,
    int binary_type,
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit);

extern void nodechip_bcbinary_fp(
    global_addr_t A_global_addr,
    global_addr_t B_global_addr,
    global_addr_t res_global_addr,
    const int* A_shape,
    const int* B_shape,
    int A_dim,
    int B_dim,
    int binary_type,
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit) ;

void nodechip_baddbmm (
    global_addr_t input1_global_addr,
    global_addr_t input1_buffer_global_addr,
    global_addr_t batch1_global_addr,
    global_addr_t batch2_global_addr,
    global_addr_t output_global_addr,
    data_type_t dtype,
    int* input1_shape,
    int* batch1_shape,
    int* batch2_shape,
    int* output_shape,
    int input1_dims,
    int batch1_dims,
    int batch2_dims,
    int output_dims,
    float alpha,
    float beta,
    int is_left_transpose,
    int is_right_transpose){
    // beta*input1 + alpha(batch1*batch2)
    if (beta != 0){
        nodechip_const_binary_fp(input1_global_addr, input1_buffer_global_addr, input1_shape, input1_dims, beta, 0, 2, dtype, 0, 0);
    }
    nodechip_batch_matmul_float(batch1_global_addr,
                                batch2_global_addr,
                                0,
                                output_global_addr,
                                dtype, dtype,
                                batch1_shape, 
                                batch2_shape,
                                batch1_dims,
                                batch2_dims,
                                output_shape,
                                &output_dims,
                                is_left_transpose,
                                is_right_transpose,
                                0, 0, 0, 0);
    nodechip_const_binary_fp(output_global_addr, output_global_addr, output_shape, output_dims, alpha, 0, 2, dtype, 0, 0);
    if (beta != 0){
        nodechip_bcbinary_fp(input1_buffer_global_addr, output_global_addr, output_global_addr, input1_shape, output_shape, input1_dims, output_dims, 0, dtype, 0, 0);
    }
}

void tpu_kernel_api_baddbmm (const void* args){
    sg_api_baddbmm_t* op_args = (sg_api_baddbmm_t*)args;
    tpu_initialize();
    int batch2_shape [] =
    {
        op_args->batch2_shape[0],
        op_args->is_right_transpose ? op_args->batch2_shape[2] : op_args->batch2_shape[1],
        op_args->is_right_transpose ? op_args->batch2_shape[1] : op_args->batch2_shape[2]
    };
    nodechip_baddbmm(
        op_args->input_global_addr,
        op_args->buffer_global_addr,
        op_args->batch1_global_addr,
        op_args->batch2_global_addr,
        op_args->output_global_addr,
        (data_type_t)op_args->dtype,
        op_args->input_shape,
        op_args->batch1_shape,
        batch2_shape,
        op_args->output_shape,
        op_args->input_dim,
        op_args->batch1_dim,
        op_args->batch2_dim,
        op_args->output_dim,
        op_args->alpha,
        op_args->beta,
        op_args->is_left_transpose,
        op_args->is_right_transpose);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_baddbmm );