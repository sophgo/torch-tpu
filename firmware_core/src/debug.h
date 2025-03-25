#pragma once
#include "tpu_kernel.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
// assert
#include <assert.h>
// #include "sg_fp16.h"
// https://wiki.sophgo.com/pages/viewpage.action?pageId=158406905
#ifdef __cplusplus
extern "C" {
#endif

#define NBYTE_ALIGN   0
#define LINE_ALIGN    1
#define COMPACT_ALIGN 2

// static void weight_reorder_multicore()
typedef global_addr_t gaddr_t;
typedef local_addr_t  laddr_t;
typedef data_type_t   dtype_t;
#define unuse(x)      (void)(x)

#define scaler_cast(x, dtype) tpu_cast( (scalar_t){.f32=x}, dtype, DT_FP32, RM_HALF_TO_EVEN)

#ifndef pipeline_move
#define pipeline_move(array, num) do { \
  for (int i = (int)num - 1; i > 0; i--) { \
    array[i] = array[i - 1];\
  }\
} while(0)
#endif

typedef struct {
    gaddr_t     addr;
    dim4        shape;
    dim4        stride;
    data_type_t dtype;
} gtensor_t;

typedef struct {
    laddr_t     addr;
    dim4        shape;
    dim4        stride;
    data_type_t dtype;
} ltensor_t;

#define init_lmem(ltensor) (ltensor).addr = 0;

#define ltensor_size(ltensor) \
    ((ltensor).shape.n * (ltensor).stride.n * tpu_data_type_size((ltensor).dtype))

#define malloc_after(ltensor1, ltensor2, ALIGN_SIZE) \
    do {                                              \
        (ltensor1).addr      = ALIGN((ltensor2).addr + ltensor_size(ltensor2), ALIGN_SIZE);                       \
        size_t total_size    = (ltensor1).addr + ltensor_size(ltensor1);                                          \
        if((int)total_size >= LOCAL_MEM_SIZE){                                                                          \
            printf("Local memory overflow, current size: %zu, total size: %u\n", total_size, LOCAL_MEM_SIZE);       \
            printf("error tensor name: %s\n", #ltensor1);                                                         \
            assert(0);                                                                                             \
        }                                                                                                         \
    } while (0)

#define check_info(cond) \
    do { \
        if(cond){ \
            printf("may error: '%s'; please check the code in %s:%d!\n", #cond, __FILE__, __LINE__); \
        } \
    } while (0)

#define get_addr_bank_id(addr) ((addr) / BANK_SIZE)

#define malloc_after_with_align(ltensor1, ltensor2, ltensor3) \
    do {                                                                  \
        laddr_t tmp_addr     = ALIGN((ltensor2).addr + ltensor_size(ltensor2), 64);                               \
        laddr_t end_addr     = (ltensor3).addr + ltensor_size(ltensor3);                                          \
        if( get_addr_bank_id(end_addr) == get_addr_bank_id(tmp_addr) ){                                           \
            tmp_addr        += ALIGN(end_addr, BANK_SIZE);                                                        \
        }                                                                                                         \
        (ltensor1).addr      = tmp_addr;                                                                          \
        size_t total_size    = (ltensor1).addr + ltensor_size(ltensor1);                                          \
        if((int)total_size > LOCAL_MEM_SIZE){                                                                          \
            printf("Local memory overflow, current size: %zu, total size: %u\n", total_size, LOCAL_MEM_SIZE);     \
            printf("error tensor name: %s\n", #ltensor1);                                                      \
            assert(0);                                                                                             \
        }                                                                                   \
    } while (0)

#define only_if(cond, flag) \
        if(cond && flag){ \
            flag = false; \
            continue;     \
        }

#define reuse_stage(cond, flag) only_if(cond, flag)

#define malloc_after_default(ltensor1, ltensor2) malloc_after(ltensor1, ltensor2, 64)

#define get_var_name(var) #var

#define inner_loop(idx, total, step) \
    for(int idx = 0; idx < total; idx += step)

#define PARALLEL_CHECK \
    bool __parallel_flag = tpu_is_parallel_state(); \
    if(__parallel_flag){ \
        tpu_parallel_end(); \
    }

#define PARALLEL_RECORY \
    if(__parallel_flag){ \
        tpu_parallel_start(); \
    }

static inline void stride_assign(ltensor_t *ltensor, int align_method){
    // align_method: 0: align, 1: line align, 2: compact align
    switch(align_method){
        case 0:
            ltensor->stride.w = 1;
            ltensor->stride.h = ltensor->shape.w;
            ltensor->stride.c = ALIGN(ltensor->shape.h * ltensor->stride.h, NPU_NUM / tpu_data_type_size(ltensor->dtype));
            ltensor->stride.n = DIV_UP(ltensor->shape.c , NPU_NUM) * ltensor->stride.c;
            break;
        case 1:
            ltensor->stride.w = 1;
            ltensor->stride.h = ALIGN(ltensor->shape.w, 64 / tpu_data_type_size(ltensor->dtype));
            ltensor->stride.c = ltensor->shape.h * ltensor->stride.h;
            ltensor->stride.n = DIV_UP(ltensor->shape.c, 64) * ltensor->stride.c;
            break;
        case 2:
            ltensor->stride.w = 1;
            ltensor->stride.h = ltensor->shape.w;
            ltensor->stride.c = ltensor->shape.h * ltensor->stride.h;
            ltensor->stride.n = DIV_UP(ltensor->shape.c, 64) * ltensor->stride.c;
            break;
        default:
            break;
    }
}

static inline gaddr_t get_gtensor_idx_addr(gtensor_t *gtensor, dim4 slice){
    size_t offset = slice.n * gtensor->stride.n * tpu_data_type_size(gtensor->dtype) +
                    slice.c * gtensor->stride.c * tpu_data_type_size(gtensor->dtype) +
                    slice.h * gtensor->stride.h * tpu_data_type_size(gtensor->dtype) +
                    slice.w * gtensor->stride.w * tpu_data_type_size(gtensor->dtype);
    return gtensor->addr + offset;
}

// [0,64, xxx,xxx]
static inline laddr_t get_ltensor_idx_addr(ltensor_t *ltensor, dim4 slice){
    int npu_start      = ltensor->addr / LOCAL_MEM_SIZE;
    laddr_t basic_addr = ltensor->addr % LOCAL_MEM_SIZE;
    int total_c_slice  = slice.c+npu_start;
    return basic_addr + slice.n * ltensor->stride.n * tpu_data_type_size(ltensor->dtype) +
                       ( total_c_slice / NPU_NUM) * ltensor->stride.c * tpu_data_type_size(ltensor->dtype) +
                       (total_c_slice % NPU_NUM) * LOCAL_MEM_SIZE +
                       slice.h * ltensor->stride.h * tpu_data_type_size(ltensor->dtype) +
                       slice.w * ltensor->stride.w * tpu_data_type_size(ltensor->dtype);
}

static inline gtensor_t get_gtensor_slice(gtensor_t *gtensor, dim4 slice, dim4 shape){
    gtensor_t slice_tensor = *gtensor;
    slice_tensor.addr      = get_gtensor_idx_addr(gtensor, slice);
    slice_tensor.shape     = shape;
    return slice_tensor;
}

static inline ltensor_t get_ltensor_slice(ltensor_t *ltensor, dim4 slice, dim4 shape){
    ltensor_t slice_tensor = *ltensor;
    slice_tensor.addr      = get_ltensor_idx_addr(ltensor, slice);
    slice_tensor.shape     = shape;
    return slice_tensor;
}

static inline void global_stride(gtensor_t *gtensor){
    gtensor->stride.w = 1;
    gtensor->stride.h = gtensor->shape.w;
    gtensor->stride.c = gtensor->shape.h * gtensor->stride.h;
    gtensor->stride.n = gtensor->shape.c * gtensor->stride.c;
}

static inline ltensor_t init_ltensor_stride( dim4 shape, data_type_t dtype, int align_method){
    ltensor_t ltensor = {0, shape, {0}, dtype};
    stride_assign(&ltensor, align_method);
    return ltensor;
}

static inline ltensor_t init_ltensor(dim4 shape, data_type_t dtype) {// we can add more func later with va_list
    return init_ltensor_stride(shape, dtype, NBYTE_ALIGN);
}

static inline gtensor_t init_gtensor(global_addr_t addr, dim4 shape, data_type_t dtype){
    gtensor_t gtensor = {addr, shape, {0}, dtype};
    global_stride(&gtensor);
    return gtensor;
}

static inline ltensor_t get_ltensor_with_reshape(ltensor_t* ltensor, dim4 shape){
    // reshape ltensor -> res_tensor, we need to check, the process is valid and safe
    // how to check? todo! currrent the caller should make sure the shape is valid
    ltensor_t res_tensor = *ltensor;
    res_tensor.shape     = shape;
    stride_assign(&res_tensor, NBYTE_ALIGN);
    return res_tensor;
}


#define show_tensor(ltensor) \
    do { \
        printf(">>>>>>>>>>>>>>>>>> show %s\n", get_var_name(ltensor)); \
        printf("core_id: %d\n", tpu_core_index()); \
        printf("shape:  %d %d %d %d\n", ltensor.shape.n, ltensor.shape.c, ltensor.shape.h, ltensor.shape.w); \
        printf("stride: %d %d %d %d\n", ltensor.stride.n, ltensor.stride.c, ltensor.stride.h, ltensor.stride.w); \
        printf("dtype: %d\n", ltensor.dtype); \
        printf("addr:  %u\n", (unsigned int) ltensor.addr); \
        printf(">>>>>>>>>>>>>>>>>> end\n"); \
    } while (0)


static inline void global_reshape(gtensor_t *gtensor, dim4 new_shape){
    gtensor->shape = new_shape;
    global_stride(gtensor);
}

static inline bool check_gtensor(gtensor_t gtensor0, gtensor_t gtensor1){

    return gtensor0.addr == gtensor1.addr &&
           gtensor0.shape.n == gtensor1.shape.n &&
           gtensor0.shape.c == gtensor1.shape.c &&
           gtensor0.shape.h == gtensor1.shape.h &&
           gtensor0.shape.w == gtensor1.shape.w &&
           gtensor0.stride.n == gtensor1.stride.n &&
           gtensor0.stride.c == gtensor1.stride.c &&
           gtensor0.stride.h == gtensor1.stride.h &&
           gtensor0.stride.w == gtensor1.stride.w &&
           gtensor0.dtype == gtensor1.dtype;
}
// if you want to open dump function, you should open in TPU1686 and copy so file into tputrain
void logfordebug(const char* fmt, ...);
void dl(laddr_t addr, dim4 shape, dim4 stride, int size, const char* fname);
void dg(gaddr_t addr, dim4 shape, int size, const char* fname);
void dlm(laddr_t addr, dim4 shape, dim4 stride, int size, int len, const char* fname, ...);
void dgm(gaddr_t addr, dim4 shape, int size, int len, const char* fname, ...);

void dump_local_data_into_file(local_addr_t addr, dim4* stride, dim4* shape, int size, const char* fname);
void dump_local_data_into_file_with_idx(local_addr_t addr, dim4* stride, dim4* shape, int size, const char* fname, int idx);
void dls(local_addr_t addr, int n, int c, int h, int w, int dflag, int layout, const char* fname);
void dlsn(local_addr_t addr, int n, int c, int h, int w, int dflag, int layout, const char* fname, int idx);
void dlsnn(local_addr_t addr, int n, int c, int h, int w, int dflag, int layout, const char* fname, int num, ...);
void show_npu_data(local_addr_t addr, int num, int flag);
void global2file(global_addr_t addr, int size, const char* fname);
void global2file_with_shape(global_addr_t addr, int n, int c, int h, int w, int nstride, int cstride, int hstride, int wstride, int size, const char* fname);
void dgsn(global_addr_t addr, int n, int c, int h, int w, int nstride, int cstride, int hstride, int wstride, int size, const char* fname, int idx);
void dgssn(global_addr_t addr, int n, int c, int h, int w, int size, const char* fname, int idx);

void dltensor(ltensor_t* ltensor, const char* name);
void dltensor_with_idx(ltensor_t* ltensor, const char* name, int idx);
void dltensor_with_2idx(ltensor_t* ltensor, const char* name, int idx1, int idx2);

void dltensor_s(ltensor_t* ltensor, dim4 shape,  const char* name);
void dltensor_s_with_idx(ltensor_t* ltensor, dim4 shape, const char* name, int idx);
void dltensor_s_with_2idx(ltensor_t* ltensor, dim4 shape, const char* name, int idx1, int idx2);

void dgtensor(gtensor_t* gtensor, const char* name);
void dgtensor_with_idx(gtensor_t* gtensor, const char* name, int idx);
void dgtensor_with_2idx(gtensor_t* gtensor, const char* name, int idx1, int idx2);

void dgtensor_s(gtensor_t* gtensor, dim4 shape,  const char* name);
void dgtensor_s_with_idx(gtensor_t* gtensor, dim4 shape, const char* name, int idx);
void dgtensor_s_with_2idx(gtensor_t* gtensor, dim4 shape, const char* name, int idx1, int idx2);
#ifdef __cplusplus
}
#endif
