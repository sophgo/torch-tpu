#include "common.h"
#include "sg_api_struct.h"
#include "common_def.h"
#include "tpu_kernel.h"

#define OP_ELTWISE_PRODUCT 0
#define OP_ELTWISE_COEF_ADD 1
#define OP_ELTWISE_MAX 2

void product_backward(
    global_addr_t inputA_global_addr,
    global_addr_t inputB_global_addr,
    global_addr_t grad_output_global_addr,
    global_addr_t grad_inputA_global_addr,
    global_addr_t grad_inputB_global_addr,
    dim4 shape,
    data_type_t dtype)
{
    int n = shape.n;
    int c = shape.c;
    int h = shape.h;
    int w = shape.w;
    int size = n * DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(h, w, dtype);
    local_addr_t grad_output_local_addr = 0;
    local_addr_t grad_inputA_local_addr = grad_output_local_addr + ALIGN(size, BANK_SIZE);
    local_addr_t grad_inputB_local_addr = grad_inputA_local_addr + ALIGN(size, BANK_SIZE);
    
    TPUKERNEL_ASSERT(3 * ALIGN(size, BANK_SIZE)<LOCAL_MEM_SIZE);
    tpu_gdma_cpy_S2L(
        grad_output_local_addr,
        grad_output_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_gdma_cpy_S2L(
        grad_inputA_local_addr,
        inputB_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_parallel_start();
    tpu_gdma_cpy_S2L(
        grad_inputB_local_addr,
        inputA_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_fp_mul(
        grad_inputA_local_addr,
        grad_inputA_local_addr,
        grad_output_local_addr,
        &shape,
        NULL,
        NULL,
        NULL,
        dtype);
    tpu_parallel_end();
    tpu_parallel_start();
    tpu_gdma_cpy_L2S(
        grad_inputA_global_addr,
        grad_inputA_local_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_fp_mul(
        grad_inputB_local_addr,
        grad_inputB_local_addr,
        grad_output_local_addr,
        &shape,
        NULL,
        NULL,
        NULL,
        dtype);
    tpu_parallel_end();
    tpu_gdma_cpy_L2S(
        grad_inputB_global_addr,
        grad_inputB_local_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    return;
}

void coeff_add_backward(
    global_addr_t grad_output_global_addr,
    global_addr_t grad_inputA_global_addr,
    global_addr_t grad_inputB_global_addr,
    dim4 shape,
    int coeff_a,
    int coeff_b,
    data_type_t dtype)
{
    int n = shape.n;
    int c = shape.c;
    int h = shape.h;
    int w = shape.w;
    scalar_t coeffa = {.s32 = coeff_a};
    scalar_t coeffb = {.f32 = (float)coeff_b/(float)coeff_a};
    int size = n * DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(h, w, dtype);
    local_addr_t grad_output_local_addr = 0;
    local_addr_t grad_input_local_addr = grad_output_local_addr + ALIGN(size, BANK_SIZE);
    tpu_gdma_cpy_S2L(
        grad_output_local_addr,
        grad_output_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_fp_mul_C(
        grad_input_local_addr,
        grad_output_local_addr,
        tpu_cast(coeffa, dtype, DT_INT32, 0),
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_gdma_cpy_L2S(
        grad_inputA_global_addr,
        grad_input_local_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    if(coeff_a == coeff_b)
    {
        tpu_gdma_system_cpy(
            grad_inputB_global_addr,
            grad_inputA_global_addr,
            n * c * h * w,
            dtype);
    }
    else
    {
        tpu_bdc_fp_mul_C(
            grad_input_local_addr,
            grad_output_local_addr,
            tpu_cast(coeffb, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
            &shape,
            NULL,
            NULL,
            dtype);
        tpu_gdma_cpy_L2S(
            grad_inputB_global_addr,
            grad_input_local_addr,
            &shape,
            NULL,
            NULL,
            dtype);
    }
    return;
}

void max_backward(
    global_addr_t inputA_global_addr,
    global_addr_t inputB_global_addr,
    global_addr_t grad_output_global_addr,
    global_addr_t grad_inputA_global_addr,
    global_addr_t grad_inputB_global_addr,
    dim4 shape,
    data_type_t dtype)
{
    int n = shape.n;
    int c = shape.c;
    int h = shape.h;
    int w = shape.w;
    int size = n * DIV_UP(c, NPU_NUM) * tpu_aligned_feature_size(h, w, dtype);
    local_addr_t inputA_local_addr = 0;
    local_addr_t inputB_local_addr = inputA_local_addr + ALIGN(size, BANK_SIZE);
    local_addr_t grad_output_local_addr = inputB_local_addr + ALIGN(size, BANK_SIZE);;
    local_addr_t grad_input_local_addr = grad_output_local_addr + ALIGN(size, BANK_SIZE);
    scalar_t scale = {.f32 = 0.f};
    variable_t inputA = {
        .type = TENSOR,
        .context.addr = inputA_local_addr,
    };
    variable_t inputB = {
        .type = TENSOR,
        .context.addr = inputB_local_addr,
    };
    variable_t grad_output = {
        .type = TENSOR,
        .context.addr = grad_output_local_addr,
    };
    variable_t zero = {
        .type = SCALAR,
        .context.scalar = tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
    };
    tpu_gdma_cpy_S2L(
        grad_output_local_addr,
        grad_output_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_gdma_cpy_S2L(
        inputA_local_addr,
        inputA_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_gdma_cpy_S2L(
        inputB_local_addr,
        inputB_global_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_less_select(
        grad_input_local_addr,
        &inputB,
        &inputA,
        &grad_output,
        &zero,
        &shape,
        dtype,
        dtype);
    tpu_gdma_cpy_L2S(
        grad_inputA_global_addr,
        grad_input_local_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    tpu_bdc_less_select(
        grad_input_local_addr,
        &inputA,
        &inputB,
        &grad_output,
        &zero,
        &shape,
        dtype,
        dtype);
    tpu_gdma_cpy_L2S(
        grad_inputB_global_addr,
        grad_input_local_addr,
        &shape,
        NULL,
        NULL,
        dtype);
    return;
}

void nodechip_eltwise_backward(
    global_addr_t inputA_global_addr,
    global_addr_t inputB_global_addr,
    global_addr_t grad_output_global_addr,
    global_addr_t grad_inputA_global_addr,
    global_addr_t grad_inputB_global_addr,
    dim4 input_shape,
    int op_code,
    int coeff_a,
    int coeff_b,
    data_type_t dtype)
{
    int n = input_shape.n;
    int c = input_shape.c;
    int h = input_shape.h;
    int w = input_shape.w;
    int total_size = n * c * h * w;
    TPUKERNEL_ASSERT(!(total_size % NPU_NUM));
    if(op_code == OP_ELTWISE_PRODUCT)
    {
        int hwsecs = MIN(total_size/NPU_NUM, ((tpu_bank_num()/3)*BANK_SIZE)/tpu_data_type_size(dtype));
        int hwslice = DIV_UP(total_size/NPU_NUM, hwsecs);
        dim4 shape = {1, NPU_NUM, 1, hwsecs};
        for(int hwidx = 0; hwidx < hwslice; hwidx++)
        {
            product_backward(
                inputA_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                inputB_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                grad_output_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                grad_inputA_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                grad_inputB_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                shape,
                dtype);
        }
    }
    else if(op_code == OP_ELTWISE_COEF_ADD)
    {
        if(coeff_a==1 && coeff_b==1)
        {
            tpu_gdma_system_cpy(
                grad_inputA_global_addr,
                grad_output_global_addr,
                total_size,
                dtype);
            tpu_gdma_system_cpy(
                grad_inputB_global_addr,
                grad_output_global_addr,
                total_size,
                dtype);
        }
        else
        {
            int hwsecs = MIN(total_size/NPU_NUM, ((tpu_bank_num()/2)*BANK_SIZE)/tpu_data_type_size(dtype)/2);
            int hwslice = DIV_UP(total_size/NPU_NUM, hwsecs);
            dim4 shape = {1, NPU_NUM, 1, hwsecs};
            for(int hwidx = 0; hwidx < hwslice; hwidx++)
            {
                coeff_add_backward(
                    grad_output_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                    grad_inputA_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                    grad_inputB_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                    shape,
                    coeff_a,
                    coeff_b,
                    dtype);
            }
        }
    }
    else if(op_code == OP_ELTWISE_MAX)
    {
        int hwsecs = MIN(total_size/NPU_NUM, ((tpu_bank_num()/4)*BANK_SIZE)/tpu_data_type_size(dtype)/2);
        int hwslice = DIV_UP(total_size/NPU_NUM, hwsecs);
        dim4 shape = {1, NPU_NUM, 1, hwsecs};
        for(int hwidx = 0; hwidx < hwslice; hwidx++)
        {
            max_backward(
                inputA_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                inputB_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                grad_output_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                grad_inputA_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                grad_inputB_global_addr + hwidx * NPU_NUM * hwsecs * tpu_data_type_size(dtype),
                shape,
                dtype);
        }
    }
    return;
}

void tpu_kernel_api_eltwise_backward(const void *args)
{
    sg_api_eltwise_backward_t *api = (sg_api_eltwise_backward_t *)args;
    dim4 shape = {api->shape[0], api->shape[1], api->shape[2], api->shape[3]};

    data_type_t forward_idtype = api->idtype;
    data_type_t forward_odtype = api->odtype;
    TPUKERNEL_ASSERT(forward_idtype == forward_odtype);
    data_type_t dtype = forward_idtype;

    int op_code = api->op_code;
    TPUKERNEL_ASSERT(op_code==0||op_code==1||op_code==2);
    tpu_initialize();
    nodechip_eltwise_backward(
            api->inputA_global_addr,
            api->inputB_global_addr,
            api->grad_output_global_addr,
            api->grad_inputA_global_addr,
            api->grad_inputB_global_addr,
            shape,
            api->op_code,
            api->coeff_a,
            api->coeff_b,
            dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_eltwise_backward);
