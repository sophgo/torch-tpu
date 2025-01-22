#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "kernel_utils_func.h"



void tpu_bdc_check_inf_nan(local_addr_t dst_addr, local_addr_t found_inf_addr, local_addr_t src_addr,
                        local_addr_t work0_addr, local_addr_t work1_addr,
                        local_addr_t work2_addr,local_addr_t work3_addr,
                        const dim4 *shape, data_type_t dtype )
{
    // 1. nan's result in dst_addr
    tpu_bdc_fp_isnan(work3_addr, src_addr, work0_addr, work1_addr, work2_addr, shape, dtype);
    // 2. inf's result in work2_addr
    tpu_bdc_fp_isinf(work2_addr, src_addr, work0_addr, shape, dtype);
    // 3. nan | inf 's result in work0_addr
    tpu_bdc_or(work0_addr, work3_addr, work2_addr, shape, NULL, NULL, NULL, DT_UINT8);
    // 4. is 1 in work0_addr, result save in dst_addr
    dim4 pool_shape = {.n = shape->n, .c = shape->c, .h = shape->h, .w = shape->w};
    if (pool_shape.w > 65535){
        for (int i = 2; i < shape->w; i++){
            if (pool_shape.w % i == 0 && pool_shape.w / i <= 65535){
                pool_shape.w = pool_shape.w / i;
                pool_shape.h = pool_shape.h * i;
                break;
            }
        }
    }
    dim2 kernel = {.h = pool_shape.h, .w = pool_shape.w};
    padding_t pad = {.top = 0, .bottom = 0, .left = 0, .right = 0};
    scalar_t pad_val = {.u32 = 0};
    scalar_t one_scalar = {.u32 = 1};
    dim2 stride = {.h = 1, .w = 1};
    dim2 dilation = {.h = 1, .w = 1};
    // tpu_bdc_int8_max_pool2d(dst_addr, work0_addr, &pool_shape, &kernel, &pad, &stride, &dilation, DT_UINT8, pad_val); // bug: this outputs {1, c, 1, 1}, only first element valid
    tpu_bdc_int8_asym_quant_conv2d_kernel_const(
        dst_addr,
        work0_addr,
        one_scalar,
        pad_val,
        pad_val,
        &pool_shape,
        NULL,
        1,
        &kernel,
        &pad,
        &stride,
        &dilation,
        DT_UINT16, // output addr has at least 2 bytes; if has problem consider overflow
        DT_UINT8,
        DT_UINT8,
        DT_UINT8,
        0);
    dim4 found_inf_shape = {.n = 1, .c = 1, .h = 1, .w = 1};
    tpu_bdc_min_C(dst_addr, dst_addr, one_scalar, &found_inf_shape, NULL, NULL, DT_UINT16);
    tpu_bdc_or(found_inf_addr, dst_addr, found_inf_addr, &found_inf_shape, NULL, NULL, NULL, DT_UINT8);
}

void nodechip_inf_check_and_unscale(
    global_addr_t       input_global_addr,
    global_addr_t       found_inf_global_addr,
    int                 length,
    float               inv_scale,
    data_type_t         idtype,
    data_type_t         found_inf_dtype)
{
    const int dsize = tpu_data_type_size ( idtype );
    int wmax = DIV_UP ( length, NPU_NUM );
    local_addr_t input_local_addrs[2];
    local_addr_t work0_local_addr, work1_local_addr, work2_local_addr, work3_local_addr, dst_addr, found_inf_dst_addr;
    local_addr_t next = 0;
    while ( true )
    {
        next = 0;
        int size = tpu_aligned_feature_size ( 1, wmax, idtype );
        input_local_addrs[0] = next; next += size;
        input_local_addrs[1] = next; next += size;
        work0_local_addr = next; next += size;
        work1_local_addr = next; next += size;
        work2_local_addr = next; next += size;
        work3_local_addr = next; next += size;
        dst_addr = next; next += tpu_aligned_feature_size (1, 1, found_inf_dtype);
        found_inf_dst_addr = next; next += tpu_aligned_feature_size (1, 1, found_inf_dtype);
        if ( ( int ) next <= LOCAL_MEM_SIZE )
        {
        break;
        }
        else
        {
        if ( wmax > 1 )
        {
            wmax /= 2;
            continue;
        }
        else
        {
            TPUKERNEL_ASSERT ( false );
        }
        }
    }

    // 1. cpy found_inf s -> l
    dim4 found_inf_shape = {.n = 1, .c = 1, .h = 1, .w = 1};
    tpu_gdma_cpy_S2L(found_inf_dst_addr, found_inf_global_addr, &found_inf_shape, NULL, NULL, found_inf_dtype);
    tpu_bdc_cast(found_inf_dst_addr, found_inf_dst_addr, &found_inf_shape, NULL, NULL, DT_UINT8, found_inf_dtype, RM_HALF_TO_EVEN);

    // 2. start check and unscale
    int todo = length;
    int done = 0;
    dim4 shape = { .n = 1, .h = 1 };
    int index = 0;
    bool l2s = false;
    dim4 l2s_shape;
    global_addr_t l2s_global_addr = 0;
    local_addr_t l2s_local_addr = 0;
    while ( todo != 0 )
    {
        if ( todo > NPU_NUM ) { shape.c = NPU_NUM; shape.w = MIN ( todo / NPU_NUM, wmax ); }
        else { shape.c = todo; shape.w = 1; }
        tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + done * dsize, &shape, NULL, NULL, idtype );
        if ( tpu_is_parallel_state() ) { tpu_parallel_end(); }
        tpu_parallel_start();
        if ( l2s ) { tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, idtype ); }
        // start//
        tpu_bdc_check_inf_nan(dst_addr, found_inf_dst_addr, input_local_addrs[index], work0_local_addr, work1_local_addr, work2_local_addr, work3_local_addr, &shape, idtype);
        scalar_t scalar;
        if (idtype == DT_FP32){
            scalar.f32 = inv_scale;
        }else if (idtype != DT_FP32){
            scalar_t s_f32 = {.f32 = inv_scale};
            scalar = tpu_fp_cast(s_f32, idtype, DT_FP32, RM_HALF_TO_EVEN);
        }
        tpu_bdc_fp_mul_C(input_local_addrs[index], input_local_addrs[index], scalar, &shape, NULL, NULL, idtype);
        // end //
        l2s = true;
        l2s_global_addr = input_global_addr + done * dsize;
        l2s_local_addr = input_local_addrs[index];
        l2s_shape = shape;
        todo -= shape.c * shape.w;
        done += shape.c * shape.w;
        index = 1 - index;
    }
    if ( tpu_is_parallel_state() )
    {
        tpu_parallel_end();
    }
    if ( l2s )
    {
        tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, idtype );
    }

    // 3. save check inf result
    tpu_gdma_cpy_L2S(found_inf_global_addr, found_inf_dst_addr, &found_inf_shape, NULL, NULL, DT_UINT8);
    tpu_poll();

    tpu_invalidate_cache ( found_inf_global_addr, 64 );
    void * tmp = (void *)tpu_global_mem_addr ( found_inf_global_addr );
    float tmp_found_inf = (float)(*( uint8_t* ) tmp);
    if (found_inf_dtype == DT_FP32)
    { 
        *(float*)tmp = tmp_found_inf;
    }
    else if (found_inf_dtype == DT_BFP16 || found_inf_dtype == DT_FP16)
    {
        scalar_t s = {.f32 = tmp_found_inf};
        scalar_t s_fp16 = tpu_fp_cast(s, found_inf_dtype, DT_FP32, RM_HALF_TO_EVEN);
        *(unsigned short *)tmp = *(unsigned short*)(&s_fp16.f16);
    }else{
        TPUKERNEL_ASSERT ( false );
    }
    tpu_flush_cache ( found_inf_global_addr, 64 );
}

static inline void nodechip_find_inf(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    data_type_t dtype
) {
    local_addr_t input_local_addr = 0;
    local_addr_t output_local_addr = LOCAL_MEM_SIZE / 2;
    dim4 input_shape = {1, 1, 1, 8};
    dim4 input_stride = {1, 1, 1, 64 / tpu_data_type_size(dtype)};
    dim2 kernel = {1, 8};
    padding_t zero_pad = {0, 0, 0, 0};
    dim2 oneone = {1, 1};
    scalar_t zero_fp32 = {.f32 = 0.f };
    tpu_gdma_cpy_S2L(input_local_addr, input_global_addr, &input_shape, NULL, &input_stride, dtype);
    tpu_bdc_fp_max_pool2d(output_local_addr,
                          input_local_addr,
                          &input_shape,
                          &kernel,
                          &zero_pad,
                          &oneone,
                          &oneone,
                          dtype,
                          zero_fp32);
    input_shape.w = 1;
    tpu_gdma_cpy_S2L(input_local_addr, output_global_addr, &input_shape, NULL, NULL, dtype);
    tpu_bdc_max(output_local_addr, input_local_addr, output_local_addr, &input_shape, NULL, NULL, NULL, dtype);
    tpu_gdma_cpy_L2S(output_global_addr, output_local_addr, &input_shape, NULL, NULL, dtype);

}

static inline void nodechip_clear_buffer(
    global_addr_t buffer_global_addr,
    data_type_t dtype
) {
    dim4 input_shape = {1, 1, 1, 8};
    dim4 input_stride = {1, 1, 1, 64 / tpu_data_type_size(dtype)};
    scalar_t zero_scalar = {.u32 = 0};
    tpu_gdma_set_C_system(buffer_global_addr, zero_scalar, &input_shape, &input_stride, dtype);
}

int tpu_kernel_api_inf_check_and_unscale(const void *args){
    sg_api_inf_check_unscale_t* api = (sg_api_inf_check_unscale_t*)args;
    int length = 1;
    for (int i = 0; i < api->dim; i++ ){ length *= api->shape[i]; }

    tpu_initialize();
    nodechip_inf_check_and_unscale(
        api->input_global_addr,
        api->found_inf_global_addr,
        length,
        api->inv_scale,
        (data_type_t)api->idtype,
        (data_type_t)api->found_inf_dtype);
    tpu_poll();
    return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_inf_check_and_unscale);

#ifdef BACKEND_SG2260
int tpu_kernel_api_inf_check_and_unscale_multi_core(const void *args){
    sg_api_inf_check_unscale_multi_core_t* api = (sg_api_inf_check_unscale_multi_core_t*)args;
    int length = 1;
    for (int i = 0; i < api->dim; i++ ){ length *= api->shape[i]; }

    tpu_initialize();
    int core_num = tpu_core_num();
    int core_idx = tpu_core_index();
    int length_slice = DIV_UP(length, core_num);
    int length_secs = DIV_UP(length, length_slice);
    TPUKERNEL_ASSERT(length_secs <= core_num);
    int cur_length_slice = length_slice;
    if (core_idx == length_secs - 1)
        cur_length_slice = length - length_slice * (length_secs - 1);
    // sometimes buffer memory is not empty. clear the buffer
    if (core_idx == 0) {
        nodechip_clear_buffer(api->found_inf_buffer_global_addr, api->found_inf_dtype);
        if (api->need_clear_found_inf){
            scalar_t zero_scalar = {.u32 = 0};
            dim4 shape = {1, 1, 1, 1};
            tpu_gdma_set_C_system(api->found_inf_global_addr, zero_scalar,&shape, NULL, api->found_inf_dtype);
        }
    }
    tpu_sync_all();

    if (core_idx * length_slice < length) {
        nodechip_inf_check_and_unscale(
            api->input_global_addr + (unsigned long long) length_slice * core_idx * tpu_data_type_size(api->idtype),
            api->found_inf_buffer_global_addr + core_idx * 64,
            cur_length_slice,
            api->inv_scale,
            (data_type_t)api->idtype,
            (data_type_t)api->found_inf_dtype
            );
    }
    tpu_sync_all();
    if (core_idx == 0) {
        nodechip_find_inf(api->found_inf_buffer_global_addr, api->found_inf_global_addr, (data_type_t)api->found_inf_dtype);
    }
    tpu_poll();
    return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_inf_check_and_unscale_multi_core);
#endif