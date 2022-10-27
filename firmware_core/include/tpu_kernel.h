#ifndef _TPU_H_
#define _TPU_H_
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include "tpu_defs.h"
typedef void (*tpu_kernel_func_t)(const void *);
void tpu_register_kernel_func(const char *name, tpu_kernel_func_t func);
void tpu_dump_registered_kernel_funcs();
#define TPUKERNEL_FUNC_REGISTER(func)                               \
__attribute__((constructor)) void tpu_kernel_register_##func() {   \
    tpu_register_kernel_func(#func, func);                         \
}
#if (defined(USING_FW_SIMULATION))
#define TPUKERNEL_LOG(format, ...)
#define TPUKERNEL_ASSERT(assertion) do {                              \
    if (!(assertion)) {                                              \
        TPUKERNEL_LOG("%s:%d: %s: Assertion \"%s\" failed.\n",        \
                     __FILE__, __LINE__,  __FUNCTION__, #assertion); \
        while (1);                                                   \
    }                                                                \
} while(0)
#elif (defined(USING_CMODEL))
#include <stdio.h>
#include <stdlib.h>
extern void print_trace();
extern int get_atomic_cmodel_assert_enable();
#define TPUKERNEL_LOG(format, ...) printf(format, ##__VA_ARGS__)
#define TPUKERNEL_ASSERT(assertion) do {                                 \
    if (get_atomic_cmodel_assert_enable()) {                             \
        if (!(assertion)) {                                              \
            TPUKERNEL_LOG("%s:%d: %s: Assertion \"%s\" failed.\n",       \
                         __FILE__, __LINE__,  __FUNCTION__, #assertion); \
            print_trace();                                               \
            exit(233);                                                   \
        }                                                                \
    }                                                                    \
} while(0)
#else
extern void fw_log(char *, ...);
//#define TPUKERNEL_LOG(format, ...) fw_log((char *)format, ##__VA_ARGS__)
#include <stdio.h>
#include <stdlib.h>
#define TPUKERNEL_LOG(format, ...) printf(format, ##__VA_ARGS__)
#define TPUKERNEL_ASSERT(assertion) do {                              \
    if (!(assertion)) {                                              \
        TPUKERNEL_LOG("%s:%d: %s: Assertion \"%s\" failed.\n",        \
                     __FILE__, __LINE__,  __FUNCTION__, #assertion); \
        while (1);                                                   \
    }                                                                \
} while(0)
#endif
#define TPUKERNEL_ERR(fmt, args...)              \
do {                                            \
    TPUKERNEL_LOG("[ERR] " fmt, ##args);         \
    TPUKERNEL_ASSERT(0);                         \
} while(0)
#if defined (USING_FW_DEBUG)
#define TPUKERNEL_DBG(fmt, args...) TPUKERNEL_LOG("[DBG] " fmt, ##args)
#else
#define TPUKERNEL_DBG(fmt, args...)
#endif
#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif
#ifndef MIN
#define MIN(x, y) (((x)) < ((y)) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) (((x)) > ((y)) ? (x) : (y))
#endif
#ifndef ALIGN_MASK
#define ALIGN_MASK(x, mask) (((x) + (mask)) & ~(mask))
#endif
#ifndef ALIGN
#define ALIGN(x, a) ALIGN_MASK(x, (__typeof__(x))(a) - 1)
#endif
#ifndef ALIGN_MASK_DOWN
#define ALIGN_MASK_DOWN(x, mask) ((x) & ~(mask))
#endif
#ifndef ALIGN_DOWN
#define ALIGN_DOWN(x, a) ALIGN_MASK_DOWN(x, (__typeof__(x))(a)-1)
#endif
#ifndef DIV_UP
#define DIV_UP(a, b) ((a) == 0 ? 0 : ((a) - 1) / (b) + 1)
#endif
#ifndef NPU_NUM
#define NPU_NUM tpu_npu_num()
#endif
#ifndef LOCAL_MEM_SIZE
#define LOCAL_MEM_SIZE tpu_local_mem_size_per_npu()
#endif
#ifndef LOCAL_MEM_BANKS
#define LOCAL_MEM_BANKS tpu_bank_num()
#endif
#ifndef L2_SRAM_SIZE
#define L2_SRAM_SIZE tpu_l2_sram_size()
#endif
#ifndef BANK_SIZE
#define BANK_SIZE  (LOCAL_MEM_SIZE / LOCAL_MEM_BANKS)
#endif
/*
 * Example:
 *     scalar_t fp_one = {.u32 = FP_ONE(dtype)};
 */
#define FP_ONE(dtype) \
    (dtype == DT_FP32 ? 0x3f800000 : (dtype == DT_FP16 ? 0x3c00 : 0x3f80))
#define FP_NEG_ONE(dtype) \
    (dtype == DT_FP32 ? 0xbf800000 : (dtype == DT_FP16 ? 0xbc00 : 0xbf80))
#define FP_MAX(dtype) \
    (dtype == DT_FP32 ? 0x7f7fffff : (dtype == DT_FP16 ? 0x7bff : 0x7f7f))
#define FP_NEG_MAX(dtype) \
    (dtype == DT_FP32 ? 0xff7fffff : (dtype == DT_FP16 ? 0xfbff : 0xff7f))
void tpu_set_id_node(void *node);
void tpu_get_id_node(void *node);
void tpu_set_parallel_id_node(void *bd_node, void* gdma_node);
void tpu_get_parallel_id_node(void *bd_node, void* gdma_node);
void tpu_enable_check_id_node();
void tpu_disable_check_id_node();
typedef struct {
    int n, c, h, w;
} dim4;
typedef struct {
    int h, w;
} dim2;
typedef struct {
    int top, bottom, left, right;
} padding_t;
typedef struct {
    int start, end;
} range_t;
typedef union {
    unsigned short bits;
    struct {
        unsigned short frac : 10; // mantissa
        unsigned short exp  : 5;  // exponent
        unsigned short sign : 1;  // sign
    } format;
} float16;
typedef union {
    unsigned short bits;
    struct {
        unsigned short frac : 7;  // mantissa
        unsigned short exp  : 8;  // exponent
        unsigned short sign : 1;  // sign
    } format;
} bfloat16;
typedef union {
    char           s8;
    unsigned char  u8;
    short          s16;
    unsigned short u16;
    float16        f16;
    bfloat16       bf16;
    int            s32;
    unsigned int   u32;
    float          f32;
} scalar_t;
typedef enum {
    TENSOR,
    SCALAR,
    VECTOR
} var_type_t;
typedef union {
    scalar_t      scalar;
    local_addr_t  addr;
} var_context_t;
typedef struct {
    var_type_t     type;
    var_context_t  context;
} variable_t;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// COMMON FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_initialize();

void tpu_poll();

void tpu_hau_poll();

void tpu_parallel_start();

void tpu_parallel_end();

bool tpu_is_parallel_state();

int tpu_npu_num();

int tpu_bank_num();

int tpu_eu_num(data_type_t dtype);

int tpu_local_mem_size_per_npu();

int tpu_l2_sram_size();

unsigned long long tpu_l2_sram_get_start_addr();

unsigned int tpu_local_mem_get_start_addr();

void *tpu_global_mem_addr(global_addr_t addr);

void *tpu_local_mem_addr(int start_idx, local_addr_t addr);

void *tpu_local_mem_addr_unified(local_addr_t addr);

void *tpu_l2_sram_addr(l2_sram_addr_t addr);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// UTILS FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int tpu_npu_index(local_addr_t addr);

int tpu_bank_index(local_addr_t addr);

int tpu_channle_num_per_npu(int start_idx, int num_channels);

int tpu_aligned_feature_size(int h, int w, data_type_t dtype);

void tpu_aligned_stride(
    dim4        *stride,
    int          start_idx,
    const dim4  *shape,
    data_type_t  dtype);

void tpu_compact_stride(dim4 *stride, int start_idx, const dim4 *shape);

void tpu_line_aligned_stride(
    dim4        *stride,
    int          start_idx,
    const dim4  *shape,
    data_type_t  dtype);

void tpu_continuous_stride(dim4 *stride, const dim4  *shape);

bool tpu_is_data_type_signed(data_type_t dtype);

bool tpu_is_data_type_int(data_type_t dtype);

bool tpu_is_data_type_signed_int(data_type_t dtype);

bool tpu_is_data_type_unsigned_int(data_type_t dtype);

bool tpu_is_data_type_fp(data_type_t dtype);

int tpu_data_type_size(data_type_t dtype);

scalar_t tpu_int_cast(
    scalar_t     src,
    data_type_t  dst_dtype,
    data_type_t  src_dtype);

scalar_t tpu_fp_cast(
    scalar_t         src,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode);

scalar_t tpu_fp_to_int_cast(
    scalar_t         src,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode);

scalar_t tpu_int_to_fp_cast(
    scalar_t         src,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode);

scalar_t tpu_cast(
    scalar_t         src,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode);

range_t tpu_bank_range(local_addr_t addr, int size);

bool tpu_range_overlapped(const range_t *r0, const range_t *r1);

bool tpu_any_range_overlapped(const range_t *ranges, int num);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// GDMA FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_gdma_cpy_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_cpy_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_cpy_L2L(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_gdma_cpy_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_cpy_nc_trans_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_cpy_nc_trans_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_cpy_nc_trans_L2L(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *dst_shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_gdma_cpy_nc_trans_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_cpy_cw_trans_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_cpy_cw_trans_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_cpy_cw_trans_L2L(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *dst_shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_gdma_cpy_cw_trans_S2S(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

unsigned int tpu_gdma_get_filter_num();

void tpu_gdma_mask_select_L2S(
    global_addr_t  dst_addr,
    local_addr_t   src_addr,
    addr_t         mask_addr,
    int            mask_in_lmem,
    const dim4    *shape,
    data_type_t    data_dtype,
    data_type_t    mask_dtype);

void tpu_gdma_mask_select_S2S(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    addr_t         mask_addr,
    int            mask_in_lmem,
    const dim4    *shape,
    data_type_t    data_dtype,
    data_type_t    mask_dtype);

void tpu_gdma_nonzero_L2S(
    global_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *shape,
    data_type_t    data_dtype,
    unsigned int   base_idx);

void tpu_gdma_nonzero_S2S(
    global_addr_t  dst_addr,
    global_addr_t  src_addr,
    const dim4    *shape,
    data_type_t    data_dtype,
    unsigned int   base_idx);

void tpu_gdma_compact_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    data_type_t    dtype);

void tpu_gdma_compact_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *shape,
    data_type_t    dtype);

void tpu_gdma_compact_nc_trans_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *dst_shape,
    data_type_t    dtype);

void tpu_gdma_compact_nc_trans_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    const dim4    *dst_shape,
    data_type_t    dtype);

void tpu_gdma_set_C_system(
    system_addr_t  dst_addr,
    scalar_t       C,
    const dim4    *shape,
    const dim4    *dst_stride,
    data_type_t    dtype);

void tpu_gdma_set_C_local(
    local_addr_t  dst_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    data_type_t   dtype);

void tpu_gdma_matrix_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    int            rows,
    int            cols,
    int            cols_per_channel,
    int            row_stride,
    data_type_t    dtype);

void tpu_gdma_matrix_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    int            rows,
    int            cols,
    int            cols_per_channel,
    int            row_stride,
    data_type_t    dtype);

void tpu_gdma_matrix_trans_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    int            src_rows,
    int            src_cols,
    int            dst_cols_per_channel,
    int            src_row_stride,
    data_type_t    dtype);

void tpu_gdma_matrix_trans_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    int            src_rows,
    int            src_cols,
    int            src_cols_per_channel,
    int            dst_row_stride,
    data_type_t    dtype);

void tpu_gdma_vector_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    int            len,
    int            len_per_channel,
    data_type_t    dtype);

void tpu_gdma_vector_L2S(
    system_addr_t  dst_addr,
    local_addr_t   src_addr,
    int            len,
    int            len_per_channel,
    data_type_t    dtype);

void tpu_gdma_channel_bcast_S2L(
    local_addr_t   dst_addr,
    system_addr_t  src_addr,
    const dim4    *shape,
    const dim4    *dst_stride,
    const dim4    *src_stride,
    data_type_t    dtype);

void tpu_gdma_channel_bcast_L2L(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_gdma_h_gather_S2L(
    local_addr_t   output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype);

void tpu_gdma_h_gather_L2S(
    system_addr_t  output_addr,
    local_addr_t   param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype);

void tpu_gdma_h_gather_L2L(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    addr_t        index_addr,
    bool          index_is_local,
    scalar_t      C,
    const dim4   *shape,
    int           param_h,
    const dim4   *output_stride,
    const dim4   *param_stride,
    const dim4   *index_stride,
    data_type_t   dtype);

void tpu_gdma_h_gather_S2S(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    scalar_t       C,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype);

void tpu_gdma_h_scatter_S2L(
    local_addr_t   output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype);

void tpu_gdma_h_scatter_L2S(
    system_addr_t  output_addr,
    local_addr_t   param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype);

void tpu_gdma_h_scatter_L2L(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    addr_t        index_addr,
    bool          index_is_local,
    const dim4   *shape,
    int           param_h,
    const dim4   *output_stride,
    const dim4   *param_stride,
    const dim4   *index_stride,
    data_type_t   dtype);

void tpu_gdma_h_scatter_S2S(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    addr_t         index_addr,
    bool           index_is_local,
    const dim4    *shape,
    int            param_h,
    const dim4    *output_stride,
    const dim4    *param_stride,
    const dim4    *index_stride,
    data_type_t    dtype);

void tpu_gdma_system_cpy(
    system_addr_t  dst_addr,
    system_addr_t  src_addr,
    unsigned int   count,
    data_type_t    dtype);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC SELECT FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_greater_select(
    local_addr_t       dst_addr,
    const variable_t  *src0,
    const variable_t  *src1,
    const variable_t  *src2,
    const variable_t  *src3,
    const dim4        *shape,
    data_type_t        src0_src1_dtype,
    data_type_t        dst_dtype);

void tpu_bdc_less_select(
    local_addr_t       dst_addr,
    const variable_t  *src0,
    const variable_t  *src1,
    const variable_t  *src2,
    const variable_t  *src3,
    const dim4        *shape,
    data_type_t        src0_src1_dtype,
    data_type_t        dst_dtype);

void tpu_bdc_equal_select(
    local_addr_t       dst_addr,
    const variable_t  *src0,
    const variable_t  *src1,
    const variable_t  *src2,
    const variable_t  *src3,
    const dim4        *shape,
    data_type_t        src0_src1_dtype,
    data_type_t        dst_dtype);

void tpu_bdc_maximum_greater_select(
    local_addr_t       dst0_addr,
    local_addr_t       dst1_addr,
    const variable_t  *src0,
    const variable_t  *src1,
    const variable_t  *src2,
    const variable_t  *src3,
    const dim4        *shape,
    data_type_t        dst0_dtype,
    data_type_t        dst1_dtype);

void tpu_bdc_minimum_less_select(
    local_addr_t       dst0_addr,
    local_addr_t       dst1_addr,
    const variable_t  *src0,
    const variable_t  *src1,
    const variable_t  *src2,
    const variable_t  *src3,
    const dim4        *shape,
    data_type_t        dst0_dtype,
    data_type_t        dst1_dtype);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC FLOATING-POINT MATRIX FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_fp32_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    local_addr_t  bias_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    int           left_cols_per_channel,
    int           right_cols_per_channel,
    bool          has_bias,
    bool          result_add);

void tpu_bdc_fp32_mm_L_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    local_addr_t  bias_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    int           left_cols_per_channel,
    int           right_cols_per_channel,
    bool          has_bias,
    bool          result_add);

void tpu_bdc_fp32_mm_L_const(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    local_addr_t  bias_addr,
    float         C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    int           right_cols_per_channel,
    bool          has_bias,
    bool          result_add);

void tpu_bdc_fp_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype,
    bool          result_add);

void tpu_bdc_fp_mm_R_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype);

void tpu_bdc_fp_mm_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_right_dtype,
    bool          result_add);

void tpu_bdc_fp_mm_L_const(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype,
    bool          result_add);

void tpu_bdc_fp_mm_R_const(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   output_dtype,
    data_type_t   left_C_dtype,
    bool          result_add);

void tpu_bdc_fp_mm_L_const_R_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype);

void tpu_bdc_fp_mm_L_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   right_C_dtype,
    bool          result_add);

void tpu_bdc_fp_mm_R_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   output_dtype,
    data_type_t   left_C_dtype,
    bool          result_add);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC INTEGER MATRIX FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_int_mm(
    local_addr_t     output_addr,
    local_addr_t     left_addr,
    local_addr_t     right_addr,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              left_cols_per_channel,
    int              right_cols_per_channel,
    data_type_t      left_dtype,
    data_type_t      right_dtype,
    char             shift,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_mm_L_trans(
    local_addr_t     output_addr,
    local_addr_t     left_addr,
    local_addr_t     right_addr,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              left_cols_per_channel,
    int              right_cols_per_channel,
    data_type_t      left_dtype,
    data_type_t      right_dtype,
    char             shift,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_mm_L_const(
    local_addr_t     output_addr,
    local_addr_t     right_addr,
    scalar_t         C,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              right_cols_per_channel,
    data_type_t      C_dtype,
    data_type_t      right_dtype,
    char             shift,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_pcs_mm(
    local_addr_t     output_addr,
    local_addr_t     left_addr,
    local_addr_t     right_addr,
    local_addr_t     shift_addr,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              left_cols_per_channel,
    int              right_cols_per_channel,
    data_type_t      left_dtype,
    data_type_t      right_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_pcs_mm_L_trans(
    local_addr_t     output_addr,
    local_addr_t     left_addr,
    local_addr_t     right_addr,
    local_addr_t     shift_addr,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              left_cols_per_channel,
    int              right_cols_per_channel,
    data_type_t      left_dtype,
    data_type_t      right_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_pcs_mm_L_const(
    local_addr_t     output_addr,
    local_addr_t     right_addr,
    local_addr_t     shift_addr,
    scalar_t         C,
    int              left_rows,
    int              left_cols,
    int              right_cols,
    int              right_cols_per_channel,
    data_type_t      C_dtype,
    data_type_t      right_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int8_mm(
    local_addr_t   output_addr,
    local_addr_t   left_addr,
    local_addr_t   right_addr,
    local_addr_t   bias_addr,
    int            left_rows,
    int            left_cols,
    int            right_cols,
    int            left_cols_per_channel,
    int            right_cols_per_channel,
    data_type_t    output_dtype,
    data_type_t    left_dtype,
    data_type_t    right_dtype,
    data_type_t    bias_dtype,
    unsigned char  lshift,
    unsigned char  rshift,
    bool           has_bias,
    bool           result_add,
    bool           result_relu);

void tpu_bdc_int8_mm_L_trans(
    local_addr_t   output_addr,
    local_addr_t   left_addr,
    local_addr_t   right_addr,
    local_addr_t   bias_addr,
    int            left_rows,
    int            left_cols,
    int            right_cols,
    int            left_cols_per_channel,
    int            right_cols_per_channel,
    data_type_t    output_dtype,
    data_type_t    left_dtype,
    data_type_t    right_dtype,
    data_type_t    bias_dtype,
    unsigned char  lshift,
    unsigned char  rshift,
    bool           has_bias,
    bool           result_add,
    bool           result_relu);

void tpu_bdc_int8_mm_L_const(
    local_addr_t   output_addr,
    local_addr_t   right_addr,
    local_addr_t   bias_addr,
    scalar_t       C,
    int            left_rows,
    int            left_cols,
    int            right_cols,
    int            right_cols_per_channel,
    data_type_t    output_dtype,
    data_type_t    C_dtype,
    data_type_t    right_dtype,
    data_type_t    bias_dtype,
    unsigned char  lshift,
    unsigned char  rshift,
    bool           has_bias,
    bool           result_add,
    bool           result_relu);

void tpu_bdc_int8_zp_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_zp_mm_R_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype);

void tpu_bdc_int8_zp_mm_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_zp_mm_L_const(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_zp_mm_R_const(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   left_dtype,
    data_type_t   C_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_zp_mm_L_const_R_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype);

void tpu_bdc_int8_zp_mm_L_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_zp_mm_R_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    scalar_t      C,
    scalar_t      zp_val,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   C_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_pc_zp_mm(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_pc_zp_mm_R_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype);

void tpu_bdc_int8_pc_zp_mm_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_pc_zp_mm_L_const(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_pc_zp_mm_R_const(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_cols,
    data_type_t   left_dtype,
    data_type_t   C_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_pc_zp_mm_L_const_R_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype);

void tpu_bdc_int8_pc_zp_mm_L_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  right_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   C_dtype,
    data_type_t   right_zp_dtype,
    bool          result_add);

void tpu_bdc_int8_pc_zp_mm_R_const_all_trans(
    local_addr_t  output_addr,
    local_addr_t  left_addr,
    local_addr_t  zp_addr,
    scalar_t      C,
    int           left_rows,
    int           left_cols,
    int           right_rows,
    data_type_t   left_dtype,
    data_type_t   C_zp_dtype,
    bool          result_add);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC FLOATING-POINT NN FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_fp_bias(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  bias_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp_add_bias_sqr(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  bias_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp_add_C_sqr(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp_sub_bias_sqr(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  bias_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp_sub_C_sqr(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp_scale(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  scale_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp_scale_bias(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  scale_addr,
    local_addr_t  bias_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp_scale_bias_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      scale,
    scalar_t      bias,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp_conv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              has_bias,
    bool              result_add);

void tpu_bdc_fp_conv2d_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              has_bias,
    bool              result_add);

void tpu_bdc_fp_depthwise_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              has_bias);

void tpu_bdc_fp_conv2d_kernel_const(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      bias_addr,
    scalar_t          C,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    bool              has_bias,
    bool              result_add);

void tpu_bdc_fp_max_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          pad_val);

void tpu_bdc_fp_ins_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    const dim2       *ins,
    data_type_t       dtype,
    scalar_t          scale);

void tpu_bdc_fp_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          scale);

void tpu_bdc_fp_ins_depthwise(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *ins,
    const dim2       *dilation,
    data_type_t       dtype,
    bool              has_bias);

void tpu_bdc_fp_depthwise2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    bool              has_bias);

void tpu_bdc_fp_roi_max_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val);

void tpu_bdc_fp_roi_avg_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val,
    scalar_t      scale);

void tpu_bdc_fp_roi_depthwise2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  weight_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC INTEGER NN FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_int8_asym_quant_conv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    scalar_t          kzp_val,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add);

void tpu_bdc_int8_asym_pc_quant_conv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      kzp_addr,
    local_addr_t      pad_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add);

void tpu_bdc_int8_asym_quant_conv2d_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    scalar_t          kzp_val,
    scalar_t          pad_val,
    scalar_t          insert_val,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add);

void tpu_bdc_int8_asym_pc_quant_conv2d_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      kzp_addr,
    local_addr_t      pad_insert_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       kzp_dtype,
    bool              result_add);

void tpu_bdc_int8_sym_quant_conv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu);

void tpu_bdc_int8_sym_quant_conv2d_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    const dim4       *input_shape,
    const dim4       *input_stride,
    int               output_c,
    const dim2       *kernel,
    const dim2       *insert,
    const padding_t  *padding,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu);

void tpu_bdc_int8_max_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       dtype,
    scalar_t          pad_val);

void tpu_bdc_int8_ins_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *ins,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    unsigned char     scale,
    unsigned char     rshift);

void tpu_bdc_int8_avg_pool2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    unsigned char     scale,
    unsigned char     rshift);

void tpu_bdc_int8_depthwise2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode);

void tpu_bdc_int8_depthwise2d_kernel_const(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      bias_addr,
    scalar_t          C,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode);

void tpu_bdc_int8_pc_pad_depthwise2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    local_addr_t      pad_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *stride,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode);

void tpu_bdc_int8_pc_pad_depthwise_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    local_addr_t      pad_addr,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *insert,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode);

void tpu_bdc_int8_depthwise_for_deconv2d(
    local_addr_t      output_addr,
    local_addr_t      input_addr,
    local_addr_t      weight_addr,
    local_addr_t      bias_addr,
    scalar_t          pad_val,
    const dim4       *input_shape,
    const dim2       *kernel,
    const padding_t  *padding,
    const dim2       *insert,
    const dim2       *dilation,
    data_type_t       output_dtype,
    data_type_t       input_dtype,
    data_type_t       weight_dtype,
    data_type_t       bias_dtype,
    unsigned char     rshift,
    bool              has_bias,
    bool              result_relu,
    rounding_mode_t   rounding_mode);

void tpu_bdc_int8_roi_max_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   dtype,
    scalar_t      except_val);

void tpu_bdc_int8_roi_avg_pool2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   output_dtype,
    data_type_t   input_dtype,
    scalar_t      except_val,
    scalar_t      scale);

void tpu_bdc_int8_roi_depthwise2d(
    local_addr_t  output_addr,
    local_addr_t  input_addr,
    local_addr_t  weight_addr,
    local_addr_t  roi_addr,
    const dim4   *input_shape,
    int           output_w,
    const dim2   *kernel,
    data_type_t   output_dtype,
    data_type_t   input_dtype,
    data_type_t   weight_dtype,
    scalar_t      except_val);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// QUANTIZATION FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_int_requant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    int              multiplier,
    char             shift,
    scalar_t         offset,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_pc_requant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_fp32_requant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    float            scale,
    float            offset,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  dst_rounding_mode,
    rounding_mode_t  src_rounding_mode);

void tpu_bdc_fp32_pc_requant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  dst_rounding_mode,
    rounding_mode_t  src_rounding_mode);

void tpu_bdc_int_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    scalar_t         offset,
    int              multiplier,
    char             shift,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_pc_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_fp32_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    scalar_t         offset,
    float            scale,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_fp32_pc_dequant(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     quant_addr,
    const dim4      *shape,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC BINARY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_and(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_and_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_or(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_or_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_xor(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_xor_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_min(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_min_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_max(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_max_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_arithmetic_shift(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    const dim4      *shift_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      shift_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_arithmetic_shift_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    char             C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_logical_shift(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    const dim4      *shift_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      shift_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_logical_shift_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    char             C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  rounding_mode);

void tpu_bdc_greater(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_greater_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_less(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_less_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_equal_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_greater_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_greater_equal_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_less_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_less_equal_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_not_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_not_equal_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    scalar_t      true_val,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_vc_and(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dtype);

void tpu_bdc_vc_or(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dtype);

void tpu_bdc_vc_xor(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dtype);

void tpu_bdc_vc_min(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dtype);

void tpu_bdc_vc_max(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dtype);

void tpu_bdc_vc_greater(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_vc_less(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_vc_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_vc_greater_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_vc_less_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_vc_not_equal(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    scalar_t      true_val,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC FLOATING-POINT BINARY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_fp_add(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_fp_add_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_fp_sub(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_fp_sub_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_fp_C_sub(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_fp_mul(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_fp_mul_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_fp32_div(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride);

void tpu_bdc_fp32_div_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride);

void tpu_bdc_fp32_C_div(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride);

void tpu_bdc_fp32_tunable_div(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    int           num_iter);

void tpu_bdc_fp32_tunable_div_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    int           num_iter);

void tpu_bdc_fp32_tunable_C_div(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    int           num_iter);

void tpu_bdc_fp32_mac(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride);

void tpu_bdc_fp32_mac_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    float         C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride);

void tpu_bdc_fp_diff_abs(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src0_stride,
    const dim4   *src1_stride,
    data_type_t   dtype);

void tpu_bdc_fp_diff_abs_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_fp32_pow(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  log_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_pow_C(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  log_coeff_addr,
    local_addr_t  exp_table_addr,
    float         C,
    const dim4   *shape);

void tpu_bdc_fp32_C_pow(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  exp_table_addr,
    float         C,
    const dim4   *shape);

void tpu_bdc_fp_vc_add(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dtype);

void tpu_bdc_fp_vc_sub(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dtype);

void tpu_bdc_fp_vc_mul(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dtype);

void tpu_bdc_fp32_vc_div(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC INTEGER BINARY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_int_add(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      dst_dtype,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_add_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_pcs_add(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    local_addr_t     shift_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      dst_dtype,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_pcs_add_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_sub(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      dst_dtype,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_sub_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_C_sub(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_pcs_sub(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    local_addr_t     shift_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      dst_dtype,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_pcs_sub_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_pcs_C_sub(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_mul(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      dst_dtype,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_mul_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    char             shift,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_pcs_mul(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    local_addr_t     shift_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      dst_dtype,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_pcs_mul_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    local_addr_t     shift_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    rounding_mode_t  rounding_mode,
    bool             saturation);

void tpu_bdc_int_min_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dtype,
    char             shift,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_max_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dtype,
    char             shift,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int8_mac(
    local_addr_t     dst_addr,
    local_addr_t     src0_addr,
    local_addr_t     src1_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src0_stride,
    const dim4      *src1_stride,
    data_type_t      src0_dtype,
    data_type_t      src1_dtype,
    unsigned char    lshift,
    unsigned char    rshift,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int8_mac_C(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    scalar_t         C,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      src_dtype,
    data_type_t      C_dtype,
    unsigned char    lshift,
    unsigned char    rshift,
    rounding_mode_t  rounding_mode);

void tpu_bdc_int_vc_add(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src0_dtype,
    data_type_t   src1_dtype,
    bool          saturation);

void tpu_bdc_int_vc_sub(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src0_dtype,
    data_type_t   src1_dtype,
    bool          saturation);

void tpu_bdc_int_vc_mul(
    local_addr_t  dst_addr,
    local_addr_t  src0_addr,
    local_addr_t  src1_addr,
    int           src0_len,
    int           src1_len,
    int           src0_len_per_channel,
    int           src1_len_per_channel,
    data_type_t   dst_dtype,
    data_type_t   src0_dtype,
    data_type_t   src1_dtype,
    bool          saturation);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC COMMON FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_cpy(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_cpy_cross_npu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_npu_bcast(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_set_C(
    local_addr_t  dst_addr,
    scalar_t      C,
    const dim4   *shape,
    const dim4   *dst_stride,
    data_type_t   dtype);

void tpu_bdc_cw_trans(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_wc_trans(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC CAST & ROUNDING FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_cast(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dst_dtype,
    data_type_t      src_dtype,
    rounding_mode_t  mode);

void tpu_bdc_fp_round(
    local_addr_t     dst_addr,
    local_addr_t     src_addr,
    const dim4      *shape,
    const dim4      *dst_stride,
    const dim4      *src_stride,
    data_type_t      dtype,
    rounding_mode_t  mode);

void tpu_bdc_fp_floor(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_fp_ceil(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// BDC UNARY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_abs(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_not(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_neg(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_fp32_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride);

void tpu_bdc_fp32_tunable_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    int           num_iter);

void tpu_bdc_fp32_compensate_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    const dim4   *shape,
    const dim4   *src_stride,
    int           num_comp);

void tpu_bdc_fp32_tunable_compensate_reciprocal(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    const dim4   *shape,
    const dim4   *src_stride,
    int           num_iter,
    int           num_comp);

void tpu_bdc_fp32_rsqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape);

void tpu_bdc_fp32_tunable_rsqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    int           num_iter);

void tpu_bdc_fp32_sqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape);

void tpu_bdc_fp32_tunable_sqrt(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    int           num_iter);

void tpu_bdc_fp32_exp(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_expm1(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_log(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_log1p(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_logx(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    float         x);

void tpu_bdc_sign(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp32_sin(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_cos(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_tan(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_cot(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_arcsin(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_arccos(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_arcsinh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_arccosh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);

void tpu_bdc_fp32_arctanh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape);
#if 0
void tpu_bdc_first_zero_bit_index(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    bit_width_t   dst_witdh,
    bit_width_t   src_witdh);

void tpu_bdc_first_one_bit_index(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    bit_width_t   dst_witdh,
    bit_width_t   src_witdh);
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// SPECIAL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_fp_taylor(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  coeff_addr,
    const dim4   *shape,
    int           num,
    data_type_t   dtype);

void tpu_bdc_table_lookup(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  table_addr,
    const dim4   *shape,
    int           len,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_fp_exponent_part(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    data_type_t   dst_dtype,
    data_type_t   src_dtype);

void tpu_bdc_npu_bcast_from_static(
    local_addr_t   dst_addr,
    static_addr_t  src_addr,
    int            npu_num,
    int            len,
    data_type_t    dtype);

void tpu_bdc_npu_distribute_from_static(
    local_addr_t   dst_addr,
    static_addr_t  src_addr,
    int            len,
    data_type_t    dtype);

void tpu_bdc_arithmetic_sequence_bcast(
    local_addr_t  dst_addr,
    int           npu_num,
    int           start,
    int           step,
    int           num);

void tpu_bdc_arithmetic_sequence_distribute(
    local_addr_t  dst_addr,
    int           start,
    int           step,
    int           num);

void tpu_bdc_arithmetic_sequence_general(
    local_addr_t  dst_addr,
    local_addr_t  buffer_addr, // size = sizeof(int32)
    int           npu_num,
    int           start,
    int           step,
    int           num);

void tpu_bdc_load_fp32_exp_coeff(local_addr_t coeff_addr);

void tpu_bdc_load_fp32_exp_table(local_addr_t table_addr);

void tpu_bdc_load_fp32_log_coeff(local_addr_t coeff_addr);

void tpu_bdc_load_fp32_erf_coeff(local_addr_t coeff_addr);

void tpu_bdc_load_fp32_sin_coeff(local_addr_t coeff_addr);

void tpu_bdc_load_fp32_cos_coeff(local_addr_t coeff_addr);

void tpu_bdc_load_fp32_tan_coeff(local_addr_t coeff_addr);

void tpu_bdc_load_fp32_arcsin_coeff(local_addr_t coeff_addr);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// SCATTER & GATHER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_w_gather(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype);

void tpu_bdc_w_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          fill_const);

void tpu_bdc_w_scatter(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype);

void tpu_bdc_hw_gather(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_h,
    int           param_w,
    data_type_t   dtype);

void tpu_bdc_hw_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_h,
    int           param_w,
    data_type_t   dtype,
    bool          fill_const);

void tpu_bdc_hw_scatter(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_h,
    int           param_w,
    data_type_t   dtype);

void tpu_bdc_batch_bcast_w_gather(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated);

void tpu_bdc_batch_bcast_w_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated,
    bool          fill_const);

void tpu_bdc_batch_bcast_w_scatter(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated);

void tpu_bdc_batch_bcast_w_mask_select(
    local_addr_t  output_addr,
    local_addr_t  count_addr,
    local_addr_t  param_addr,
    local_addr_t  mask_addr,
    const dim4   *shape,
    data_type_t   dtype,
    data_type_t   mask_dtype,
    bool          is_param_repeated);

void tpu_bdc_batch_bcast_h_gather(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_h,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated);

void tpu_bdc_batch_bcast_h_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_h,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated,
    bool          fill_const);

void tpu_bdc_batch_bcast_h_scatter(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_h,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated);

void tpu_bdc_4bank_w_gather(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated);

void tpu_bdc_4bank_w_gather_exception(
    local_addr_t  output_addr,
    local_addr_t  param_addr,
    local_addr_t  index_addr,
    scalar_t      C,
    const dim4   *shape,
    int           param_w,
    data_type_t   dtype,
    data_type_t   index_dtype,
    bool          is_param_repeated,
    bool          fill_const);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// ACTIVE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_bdc_relu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape,
    const dim4   *dst_stride,
    const dim4   *src_stride,
    data_type_t   dtype);

void tpu_bdc_prelu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    scalar_t      alpha,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp32_elu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    float         alpha,
    const dim4   *shape);

void tpu_bdc_fp32_sigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_hsigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    const dim4   *shape,
    float         scope,
    float         offset);

void tpu_bdc_fp_isfinite(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    const dim4   *shape,
    data_type_t   dtype);

void tpu_bdc_fp32_sinh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_cosh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_tanh(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_softplus(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  log_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape,
    float         beta);

void tpu_bdc_fp32_softsign(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    const dim4   *shape);

void tpu_bdc_fp32_erf(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_erfc(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_gelu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  work2_addr,
    local_addr_t  work3_addr,
    local_addr_t  exp_coeff_addr,
    local_addr_t  erf_coeff_addr,
    local_addr_t  exp_table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_gelu_fast(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_mish(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_swish(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    float         beta,
    const dim4   *shape);

void tpu_bdc_fp32_silu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_selu(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    const dim4   *shape);

void tpu_bdc_fp32_log_sigmoid(
    local_addr_t  dst_addr,
    local_addr_t  src_addr,
    local_addr_t  work0_addr,
    local_addr_t  work1_addr,
    local_addr_t  coeff_addr,
    local_addr_t  table_addr,
    local_addr_t  ln_coeff_addr,
    const dim4   *shape);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// HAU FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void tpu_hau_sort(
    system_addr_t  output_addr,
    system_addr_t  input_addr,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype);

void tpu_hau_sort_natural_index(
    system_addr_t  output_data_addr,
    system_addr_t  output_idx_addr,
    system_addr_t  input_addr,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype);

void tpu_hau_sort_specific_index(
    system_addr_t  output_data_addr,
    system_addr_t  output_idx_addr,
    system_addr_t  input_data_addr,
    system_addr_t  input_idx_addr,
    int            len,
    int            K,
    bool           descended,
    data_type_t    dtype);

void tpu_hau_hard_nms(
    system_addr_t  output_addr,
    system_addr_t  input_addr,
    int            box_num,
    int            keep_num);

void tpu_hau_soft_nms(
    system_addr_t  output_addr,
    system_addr_t  iou_addr,
    system_addr_t  score_addr,
    float          threshold,
    int            box_num,
    int            keep_num);

void tpu_hau_line_gather(
    system_addr_t  output_addr,
    system_addr_t  param_addr,
    system_addr_t  index_addr,
    scalar_t       C,
    int            line_num,
    int            line_len,
    int            index_len,
    int            start,
    int            end,
    data_type_t    dtype,
    bool           fill_const);

void tpu_invalidate_cache(system_addr_t address,
                          unsigned long long size);

void tpu_flush_cache(system_addr_t address,
                     unsigned long long size);

#ifdef __cplusplus
}
#endif
#endif /* _TPU_H_ */
