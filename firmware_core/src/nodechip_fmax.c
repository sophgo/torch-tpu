#include "sg_api_struct.h"
#include "tpu_kernel.h"

void isNan(
    local_addr_t output_local_addr,
    local_addr_t input_local_addr,
    local_addr_t work0_local_addr,
    local_addr_t work1_local_addr,
    const dim4 *shape,
    const dim4 *stride,
    data_type_t dtype
);

void replaceWithNotNan(
    local_addr_t output_local_addr,
    local_addr_t input_local_addr,
    local_addr_t other_local_addr,
    local_addr_t mask_local_addr,
    local_addr_t work0_local_addr,
    const dim4 *shape,
    const dim4 *stride,
    data_type_t dtype
){
    scalar_t  C= {.u8 = 1};
    tpu_bdc_set_C(work0_local_addr,C,shape,stride,DT_UINT8);
    variable_t src0 = {.type = TENSOR, .context = {.addr = mask_local_addr}};
    variable_t src1 = {.type = TENSOR, .context = {.addr = work0_local_addr}};
    variable_t src2 = {.type = TENSOR, .context = {.addr = other_local_addr}};
    variable_t src3 = {.type = TENSOR, .context = {.addr = input_local_addr}};
    tpu_bdc_equal_select(output_local_addr,&src0,&src1, &src2,&src3,shape,DT_UINT8,dtype);
}

void nodechip_fmaxc (
global_addr_t input_global_addr,
float scalar,
global_addr_t output_global_addr,
int length,
data_type_t dtype )
{
  if(length==0) return;
  int npu_num=tpu_npu_num();
  int bank_num=tpu_bank_num();
  int bank_size = tpu_local_mem_size_per_npu()/bank_num;
  int tensor_num=2+2; // 2 inputs, 2 outputs
  int coeff_bank_num=0; // 0 coeff
  int tensor_size = (bank_num-coeff_bank_num)/tensor_num * bank_size;
  TPUKERNEL_ASSERT(tensor_size>0);

  local_addr_t input_local_addrs[2]={0, tensor_size};
  local_addr_t output_local_addrs[2]={2*tensor_size, 3*tensor_size};
  int dtype_size = tpu_data_type_size(dtype);
  int tensor_w = DIV_UP(MIN(length, tensor_size*npu_num/dtype_size), npu_num);

  int todo = length;
  int done = 0;
  dim4 shape = { .n = 1, .h = 1 };
  int index = 0;
  bool l2s = false;
  dim4 l2s_shape;
  global_addr_t l2s_global_addr = 0;
  local_addr_t l2s_local_addr = 0;
  
  scalar_t value={.f32=scalar};

  while ( todo != 0 )
  {
    if ( todo > NPU_NUM )
    {
      shape.c = NPU_NUM;
      shape.w = MIN ( todo / NPU_NUM, tensor_w );
    }
    else
    {
      shape.c = todo;
      shape.w = 1;
    }
    tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + done * dtype_size, &shape, NULL, NULL, dtype );
    if ( tpu_is_parallel_state() )
    {
      tpu_parallel_end();
    }
    tpu_parallel_start();
    if ( l2s )
    {
      tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }
    __uint32_t scalar_uint32= *(__uint32_t*) &scalar;
    if (((scalar_uint32 & 0x7f800000) == 0x7f800000) && ((scalar_uint32 & 0x7fffffff) != 0x7f800000)){
        tpu_bdc_cpy(output_local_addrs[index],input_local_addrs[index],&shape,NULL,NULL,dtype);
            
    } else{
        tpu_bdc_max_C(output_local_addrs[index],input_local_addrs[index],value,&shape,NULL,NULL,dtype);
    }
    

    l2s = true;
    l2s_global_addr = output_global_addr + done * dtype_size;
    l2s_local_addr = output_local_addrs[index];
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
    tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
  }
}



void tpu_kernel_api_fmaxc(const void *args) {
    sg_api_fmaxc_t *api = (sg_api_fmaxc_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                     api->dtype == DT_BFP16 || api->dtype == DT_INT32);

    int length = 1;
    for(int i = 0; i < api->dim; ++i){
        length *= api->shape[i];
    }

    tpu_initialize();
    nodechip_fmaxc(api->input_global_addr,
                         api->scalar,
                         api->output_global_addr,
                         length,
                         (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_fmaxc);

void tpu_kernel_api_fmaxc_multi_core(const void *args) {
    sg_api_fmaxc_t *api = (sg_api_fmaxc_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                     api->dtype == DT_BFP16 || api->dtype == DT_INT32);

    int length = 1;
    for(int i = 0; i < api->dim; ++i){
        length *= api->shape[i];
    }
    tpu_initialize();

    int core_num = tpu_core_num();
    int core_idx = tpu_core_index();

    int length_slice = DIV_UP(length, core_num);
    int length_secs = DIV_UP(length, length_slice);
    TPUKERNEL_ASSERT(length_secs <= core_num);
    int cur_length_slice = length_slice;
    if (core_idx == length_secs - 1) {
        cur_length_slice = length - length_slice * (length_secs - 1);
    }
    nodechip_fmaxc(api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                         api->scalar,
                         api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                         cur_length_slice,
                         (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_fmaxc_multi_core);


void nodechip_fmax (
    global_addr_t input_global_addr,
    global_addr_t other_global_addr,
    global_addr_t output_global_addr,
    int length,
    data_type_t dtype )
{
    if(length==0) return;
    int npu_num=tpu_npu_num();
    int bank_num=tpu_bank_num();
    int bank_size = tpu_local_mem_size_per_npu()/bank_num;
    int tensor_num=2+2+2+3; // 2 inputs, 2 other, 2 outputs, 3 buffers
    int coeff_bank_num=0; // 0 coeff
    int tensor_size = (bank_num-coeff_bank_num)/tensor_num * bank_size;
    TPUKERNEL_ASSERT(tensor_size>0);

    local_addr_t input_local_addrs[2]={0, tensor_size};
    local_addr_t other_local_addrs[2]={2*tensor_size, 3*tensor_size};
    local_addr_t output_local_addrs[2]={4*tensor_size, 5*tensor_size};
    local_addr_t work_local_addrs[3]={6*tensor_size,7*tensor_size,8*tensor_size};

    int dtype_size = tpu_data_type_size(dtype);
    int tensor_w = DIV_UP(MIN(length, tensor_size*npu_num/dtype_size), npu_num);

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
        if ( todo > NPU_NUM )
        {
        shape.c = NPU_NUM;
        shape.w = MIN ( todo / NPU_NUM, tensor_w );
        }
        else
        {
        shape.c = todo;
        shape.w = 1;
        }
        tpu_gdma_cpy_S2L ( input_local_addrs[index], input_global_addr + done * dtype_size, &shape, NULL, NULL, dtype );
        tpu_gdma_cpy_S2L ( other_local_addrs[index], other_global_addr + done * dtype_size, &shape, NULL, NULL, dtype );
        if ( tpu_is_parallel_state() )
        {
        tpu_parallel_end();
        }
        tpu_parallel_start();
        if ( l2s )
        {
        tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
        }
        
        tpu_bdc_max(output_local_addrs[index],input_local_addrs[index],other_local_addrs[index],&shape,NULL,NULL,NULL,dtype);
        isNan(work_local_addrs[0],input_local_addrs[index],work_local_addrs[1],work_local_addrs[2],&shape,NULL,dtype);
        replaceWithNotNan(work_local_addrs[2],output_local_addrs[index],other_local_addrs[index],work_local_addrs[0],work_local_addrs[1],&shape,NULL,dtype);
        isNan(work_local_addrs[0],other_local_addrs[index],work_local_addrs[1],output_local_addrs[index],&shape,NULL,dtype);
        replaceWithNotNan(output_local_addrs[index],work_local_addrs[2],input_local_addrs[index],work_local_addrs[0],work_local_addrs[1],&shape,NULL,dtype);
        
        l2s = true;
        l2s_global_addr = output_global_addr + done * dtype_size;
        l2s_local_addr = output_local_addrs[index];
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
        tpu_gdma_cpy_L2S ( l2s_global_addr, l2s_local_addr, &l2s_shape, NULL, NULL, dtype );
    }
}

void tpu_kernel_api_fmax(const void *args) {
    sg_api_fmax_t *api = (sg_api_fmax_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                     api->dtype == DT_BFP16 || api->dtype == DT_INT32);

    int length = 1;
    for(int i = 0; i < api->dim; ++i){
        length *= api->shape[i];
    }

    tpu_initialize();
    nodechip_fmax(api->input_global_addr,
                         api->other_global_addr,
                         api->output_global_addr,
                         length,
                         (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_fmax);

void tpu_kernel_api_fmax_multi_core(const void *args) {
    sg_api_fmax_t *api = (sg_api_fmax_t*)args;
    TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                     api->dtype == DT_BFP16 || api->dtype == DT_INT32);

    int length = 1;
    for(int i = 0; i < api->dim; ++i){
        length *= api->shape[i];
    }
    tpu_initialize();

    int core_num = tpu_core_num();
    int core_idx = tpu_core_index();

    int length_slice = DIV_UP(length, core_num);
    int length_secs = DIV_UP(length, length_slice);
    TPUKERNEL_ASSERT(length_secs <= core_num);
    int cur_length_slice = length_slice;
    if (core_idx == length_secs - 1) {
        cur_length_slice = length - length_slice * (length_secs - 1);
    }
    nodechip_fmax(api->input_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                         api->other_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                         api->output_global_addr + (length_slice * core_idx) * tpu_data_type_size(api->dtype),
                         cur_length_slice,
                         (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_fmax_multi_core);

void nodechip_fmax_bcast(
global_addr_t input_global_addr,
global_addr_t other_global_addr,
global_addr_t output_global_addr,
const dim4   *input_shape,
const dim4   *other_shape,
const dim4   *output_shape,
data_type_t   dtype) {
    dim4 input_global_stride, other_global_stride, output_global_stride;
    tpu_continuous_stride(&input_global_stride,  input_shape);
    tpu_continuous_stride(&other_global_stride,  other_shape);
    tpu_continuous_stride(&output_global_stride, output_shape);

    const bool input_bcast[4] = {
        input_shape->n != output_shape->n,
        input_shape->c != output_shape->c,
        input_shape->h != output_shape->h,
        input_shape->w != output_shape->w
    };
    const bool other_bcast[4] = {
        other_shape->n != output_shape->n,
        other_shape->c != output_shape->c,
        other_shape->h != output_shape->h,
        other_shape->w != output_shape->w
    };
    bool input_bcast_all = false, other_bcast_all = false;
    for(int i = 0; i < 4; ++i) {
        input_bcast_all = input_bcast_all || input_bcast[i];
        other_bcast_all = other_bcast_all || other_bcast[i];
    }

    const int c_per_npu = DIV_UP ( output_shape->c, NPU_NUM );
    int hmax = output_shape->h, nmax = output_shape->n, cmax = c_per_npu * NPU_NUM;
    local_addr_t output_addr, input_addr, other_addr,work0_addr,work1_addr,work2_addr;
    while(true) {
        output_addr = 0;
        int output_size = tpu_aligned_feature_size(hmax, output_shape->w, dtype) * DIV_UP(cmax, NPU_NUM) * nmax;
        input_addr = output_addr + output_size;
        int input_size = output_size;
        other_addr = input_addr + input_size;
        int other_size = output_size;
        work0_addr = other_addr + output_size;
        work1_addr = work0_addr + output_size;
        work2_addr = work1_addr + output_size;
        int total_size = work2_addr + other_size;
        if(total_size <= LOCAL_MEM_SIZE) {
            break;
        }
        else {
            if(cmax > NPU_NUM) {
                if(cmax % NPU_NUM == 0) {
                    cmax -= NPU_NUM;
                }
                else {
                    cmax -= (cmax % NPU_NUM);
                }
                continue;
            }
            else if(nmax > 1) {
                nmax /= 2;
                continue;
            }
            else if(hmax > 1) {
                hmax /= 2;
                continue;
            }
            else {
                TPUKERNEL_ASSERT ( false );
            }
        }
    }
    dim4 shape = { .w = output_shape->w };
    dim4 input_local_shape, other_local_shape;
    dim4 input_local_stride, other_local_stride;
    int ctodo = output_shape->c, cdone = 0;
    while(ctodo > 0) {
        shape.c = MIN(ctodo, cmax);
        int ntodo = output_shape->n, ndone = 0;
        while(ntodo > 0) {
            shape.n = MIN(ntodo, nmax);
            int htodo = output_shape->h, hdone = 0;
            while(htodo > 0) {
                shape.h = MIN (htodo, hmax);
                // Move input from global memory to local memory
                tpu_aligned_stride (&input_local_stride,
                                    0,
                                    &shape,
                                    dtype);
                input_local_shape.n = input_bcast[0] ? 1 : shape.n;
                input_local_shape.c = input_bcast[1] ? 1 : shape.c;
                input_local_shape.h = input_bcast[2] ? 1 : shape.h;
                input_local_shape.w = input_bcast[3] ? 1 : shape.w;
                global_addr_t input_global_addr_gdma = input_global_addr +
                                ((input_bcast[0] ? 0 : ndone) * input_global_stride.n +
                                 (input_bcast[1] ? 0 : cdone) * input_global_stride.c +
                                 (input_bcast[2] ? 0 : hdone) * input_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_S2L(input_addr,
                                 input_global_addr_gdma,
                                 &input_local_shape,
                                 &input_local_stride,
                                 &input_global_stride,
                                 dtype);
                
                // Move other from global memory to local memory
                tpu_aligned_stride(&other_local_stride, 0, &shape, dtype);
                other_local_shape.n = other_bcast[0] ? 1 : shape.n;
                other_local_shape.c = other_bcast[1] ? 1 : shape.c;
                other_local_shape.h = other_bcast[2] ? 1 : shape.h;
                other_local_shape.w = other_bcast[3] ? 1 : shape.w;
                global_addr_t other_global_addr_gdma = other_global_addr +
                                ((other_bcast[0] ? 0 : ndone) * other_global_stride.n +
                                 (other_bcast[1] ? 0 : cdone) * other_global_stride.c +
                                 (other_bcast[2] ? 0 : hdone) * other_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_S2L(other_addr,
                                 other_global_addr_gdma,
                                 &other_local_shape,
                                 &other_local_stride,
                                 &other_global_stride,
                                 dtype);
                
                // Broadcast input if needed
                if(input_bcast[1]) {
                    input_local_shape.c = NPU_NUM;
                    tpu_bdc_npu_bcast(input_addr, input_addr, &input_local_shape, dtype);
                }
                if(input_bcast[0] || input_bcast[2] || input_bcast[3] || (input_bcast[1] && shape.c > NPU_NUM)) {
                    dim4 input_bcast_stride;
                    input_bcast_stride.n = input_bcast[0] ? 0 : input_local_stride.n;
                    input_bcast_stride.c = input_bcast[1] ? 0 : input_local_stride.c;
                    input_bcast_stride.h = input_bcast[2] ? 0 : input_local_stride.h;
                    input_bcast_stride.w = input_bcast[3] ? 0 : input_local_stride.w;
                    tpu_bdc_cpy(input_addr, input_addr, &shape, NULL, &input_bcast_stride, dtype);
                }
                
                // Broadcast other if needed
                if(other_bcast[1]) {
                    other_local_shape.c = NPU_NUM;
                    tpu_bdc_npu_bcast(other_addr, other_addr, &other_local_shape, dtype);
                }
                if(other_bcast[0] || other_bcast[2] || other_bcast[3] || (other_bcast[1] && shape.c > NPU_NUM)) {
                    dim4 other_bcast_stride;
                    other_bcast_stride.n = other_bcast[0] ? 0 : other_local_stride.n;
                    other_bcast_stride.c = other_bcast[1] ? 0 : other_local_stride.c;
                    other_bcast_stride.h = other_bcast[2] ? 0 : other_local_stride.h;
                    other_bcast_stride.w = other_bcast[3] ? 0 : other_local_stride.w;
                    tpu_bdc_cpy ( other_addr, other_addr, &shape, NULL, &other_bcast_stride, dtype );
                }
                
                
                // Select
                tpu_bdc_max(output_addr, input_addr, other_addr, &shape, NULL, NULL, NULL, dtype);
                isNan(work0_addr,input_addr,work1_addr,work2_addr,&shape,NULL,dtype);
                replaceWithNotNan(work2_addr,output_addr,other_addr,work0_addr,work1_addr,&shape,NULL,dtype);
                isNan(work0_addr,other_addr,work1_addr,output_addr,&shape,NULL,dtype);
                replaceWithNotNan(output_addr,work2_addr,input_addr,work0_addr,work1_addr,&shape,NULL,dtype);

                // Move out from local memory to global memory
                global_addr_t output_global_addr_gdma = output_global_addr +
                                (ndone * output_global_stride.n +
                                cdone * output_global_stride.c +
                                hdone * output_global_stride.h) * tpu_data_type_size(dtype);
                tpu_gdma_cpy_L2S(output_global_addr_gdma,
                                 output_addr,
                                 &shape,
                                 &output_global_stride,
                                 NULL,
                                 dtype);
                htodo -= shape.h;
                hdone += shape.h;
            }
            ntodo -= shape.n;
            ndone += shape.n;
        }
        ctodo -= shape.c;
        cdone += shape.c;
    }
}


void tpu_kernel_api_fmax_bcast(const void *args) {
    sg_api_fmax_bcast_t *api = (sg_api_fmax_bcast_t*)args;
    TPUKERNEL_ASSERT(api->output_dim > 0 && api->output_dim <= 4);

    TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                     api->dtype == DT_BFP16 || api->dtype == DT_INT32);

    dim4 input_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 other_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    dim4 output_shape = { .n = 1, .c = 1, .h = 1, .w = 1 };
    
    if(api->output_dim>= 1) {
        if(api->input_dim>=1) input_shape.w = api->input_shape[api->input_dim-1];
        if(api->other_dim>=1) other_shape.w = api->other_shape[api->other_dim-1];
        output_shape.w = input_shape.w > other_shape.w ? input_shape.w : other_shape.w;
    }
    if(api->output_dim >= 2) {
        if(api->input_dim>=2) input_shape.h = api->input_shape[api->input_dim-2];
        if(api->other_dim>=2) other_shape.h = api->other_shape[api->other_dim-2];
        output_shape.h = input_shape.h > other_shape.h ? input_shape.h : other_shape.h;
    }
    if(api->output_dim >= 3) {
        if(api->input_dim>=3) input_shape.c = api->input_shape[api->input_dim-3];
        if(api->other_dim>=3) other_shape.c = api->other_shape[api->other_dim-3];
        output_shape.c = input_shape.c > other_shape.c ? input_shape.c : other_shape.c;
    }
    if(api->output_dim >= 4) {
        if(api->input_dim>=4) input_shape.n = api->input_shape[api->input_dim-4];
        if(api->other_dim>=4) other_shape.n = api->other_shape[api->other_dim-4];
        output_shape.n = input_shape.n > other_shape.n ? input_shape.n : other_shape.n;
    }

    tpu_initialize();
    nodechip_fmax_bcast(api->input_global_addr,
                         api->other_global_addr,
                         api->output_global_addr,
                         &input_shape,
                               &other_shape,
                               &output_shape,
                         (data_type_t)api->dtype);
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_fmax_bcast);


