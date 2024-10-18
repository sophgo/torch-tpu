#include "sg_api_struct.h"
#include "tpu_kernel.h"
 
 
#if 0
#define USE_PIPELINE
 
#define POOL_MAX_KSIZE (1 << 16)
#define ROUND_MODE RM_HALF_AWAY_FROM_ZERO
#define RSQRT_NUM_ITER 4
#define MEMORY_BASE 0x100000000UL
#define MEMORY_SIZE (16UL * 1024 * 1024 * 1024)
// #define DEBUG_GN
/* common utils */
 
extern void nodechip_scale_forward(global_addr_t bottom_global_addr,
                                   global_addr_t scale_global_addr,
                                   global_addr_t bias_global_addr,
                                   global_addr_t top_global_addr, int bottom_n,
                                   int bottom_c, int bottom_h, int bottom_w,
                                   int axis,     // scale begin axis
                                   int axis_num, // scale axis num
                                   int has_bias, int if_relu,
                                   float relu_upper_limit,
                                   int merge_weight_bias, // (scale, bias)
                                   data_type_t dtype);
 
// eltwise binary and broadcast binary
extern void
nodechip_bcbinary_fp(global_addr_t A_global_addr, global_addr_t B_global_addr,
                     global_addr_t res_global_addr, const int *A_shape,
                     const int *B_shape, int A_dim, int B_dim, int binary_type,
                     data_type_t dtype, int if_relu, float relu_upper_limit);
 
typedef enum {
  SWPL_LOAD = 0,
  SWPL_COMPUTE = 1,
  SWPL_STORE = 2,
} SWPL_STAGE;
 
#define SWPL_BEGIN(idxes, fulls)                                               \
  int stage_idx = 0, draning_idx = 0;                                          \
  while (idxes[SWPL_STORE][0] < fulls[0]) {
 
#define SWPL_END }
 
#define SWPL_PROC_BEGIN tpu_parallel_start();
#define SWPL_PROC_END tpu_parallel_end();
 
#define SWPL_TRST_BEGIN                                                        \
  ++stage_idx;                                                                 \
  if (draning_idx < 1) {
 
#define SWPL_TRST_END                                                          \
  }                                                                            \
  ++draning_idx;
 
#define SWPL_IDX_ARR_INC(idxes, slices, fulls, rank)                           \
  IDX_ARR_INC(idxes[0], slices, fulls, rank)
 
#define SWPL_PINGPONG_ARR_DECL(pingpong_arr, local_addrs, num, mem_sizes)      \
  local_addr_t pingpong_arr[2][num];                                           \
  for (int i = 0; i < num; ++i) {                                              \
    pingpong_arr[0][i] = local_addrs[i];                                       \
    pingpong_arr[1][i] = local_addrs[i];                                       \
    if (mem_sizes[i] > 0)                                                      \
      pingpong_arr[1][i] += mem_sizes[i];                                      \
  }
 
#define SWPL_STORE_BEGIN                                                       \
  if (stage_idx > 1) {                                                         \
    const int $STAGE = SWPL_STORE;                                             \
    const int $PINGPONG = stage_idx & 1;                                       \
    UNUSED($STAGE);                                                            \
    UNUSED($PINGPONG);
#define SWPL_STORE_END }
 
#define SWPL_LOAD_BEGIN                                                        \
  if (draning_idx < 1) {                                                       \
    const int $STAGE = SWPL_LOAD;                                              \
    const int $PINGPONG = stage_idx & 1;                                       \
    UNUSED($STAGE);                                                            \
    UNUSED($PINGPONG);
#define SWPL_LOAD_END }
 
#define SWPL_COMPUTE_BEGIN                                                     \
  if (stage_idx > 0 && draning_idx < 2) {                                      \
    const int $STAGE = SWPL_COMPUTE;                                           \
    const int $PINGPONG = (stage_idx - 1) & 1;                                 \
    UNUSED($STAGE);                                                            \
    UNUSED($PINGPONG);
#define SWPL_COMPUTE_END }
 
#define SWPL_STORE_BEGIN                                                       \
  if (stage_idx > 1) {                                                         \
    const int $STAGE = SWPL_STORE;                                             \
    const int $PINGPONG = stage_idx & 1;                                       \
    UNUSED($STAGE);                                                            \
    UNUSED($PINGPONG);
#define SWPL_STORE_END }
 
#define SWPL_STAGE_ARR_MOVE(array, rank)                                       \
  for (int i = 2; i > 0; i--) {                                                \
    for (int j = 0; j < rank; ++j) {                                           \
      array[i][j] = array[i - 1][j];                                           \
    }                                                                          \
  }
 
static bool multiloop_incr_idxes(int *idxes, const int *slices,
                                 const int *fulls, int rank) {
  for (int j = rank - 1; j >= 0; --j) {
    idxes[j] += slices[j];
    if (idxes[j] < fulls[j])
      return true;
    if (j == 0)
      return false;
    idxes[j] = 0;
  }
  return false;
}
 
#define IDX_ARR_INC(idxes, slices, fulls, rank)                                \
  if (multiloop_incr_idxes((int *)idxes, (int *)slices, (int *)fulls, rank))   \
    continue;
 
typedef __uint128_t uint128_t;
 
int is_normal_memory(const void *addr) {
  return (unsigned long)addr >= MEMORY_BASE &&
         (unsigned long)addr < MEMORY_SIZE;
}
 
void *memset_normal(void *s, int c, size_t n);
 
void *memset_io(void *b, int c, size_t n) {
  unsigned long s = (unsigned long)b;
  uint128_t _tmp;
  unsigned long tmp;
 
  tmp = (unsigned long)&_tmp;
  memset_normal(&_tmp, c, sizeof(_tmp));
 
  while (n) {
    if ((s & (16 - 1)) == 0 && n >= 16) {
      *(volatile uint128_t *)s = _tmp;
      s += 16;
      n -= 16;
    } else if ((s & (8 - 1)) == 0 && n >= 8) {
      *(volatile uint64_t *)s = *((volatile uint64_t *)tmp);
      s += 8;
      n -= 8;
 
    } else if ((s & (4 - 1)) == 0 && n >= 4) {
      *(volatile uint32_t *)s = *((volatile uint32_t *)tmp);
      s += 4;
      n -= 4;
    } else {
      *(volatile uint8_t *)s = c;
      s += 1;
      n -= 1;
    }
  }
  return b;
}
 
void *memset(void *s, int c, size_t n) {
  if (is_normal_memory(s))
    return memset_normal(s, c, n);
  else
    return memset_io(s, c, n);
}
 
#define SWPL_STAGE_ARR_DECL(type, arr, rank)                                   \
  type arr[3][rank];                                                           \
  memset(arr, 0, sizeof(arr));
 
#define IDX_ARR_INC(idxes, slices, fulls, rank)                                \
  if (multiloop_incr_idxes((int *)idxes, (int *)slices, (int *)fulls, rank))   \
    continue;
 
typedef struct {
  uint32_t slices[FW_MAX_SHAPE_DIMS];
  uint32_t fulls[FW_MAX_SHAPE_DIMS];
  int rank;
} swpl_split_info_t;
 
typedef struct {
  local_addr_t *addrs;
  int *mem_sizes;
  int num;
} swpl_lmem_scheme_t;
 
/**
 * \brief software pipeline load functor
 * \param system_addrs: type: const system_addr_t*
 * \param local_addrs: type: const local_addr_t*
 * \param idxs: type: const uint32_t*
 * \param slice_info: type: const swpl_split_info_t*
 * \param fw_param: type: const void*
 **/
typedef void (*swpl_load_func_t)(const system_addr_t *system_addrs,
                                 const local_addr_t *local_addrs,
                                 const uint32_t *idxs,
                                 const swpl_split_info_t *slice_info,
                                 const void *fw_param);
 
/**
 * \brief software pipeline compute functor
 * \param local_addrs: type: const local_addr_t*
 * \param idxs: type: const uint32_t*
 * \param slice_info: type: const swpl_split_info_t*
 * \param fw_param: type: const void*
 * \param bw_param: type: void*
 **/
typedef void (*swpl_compute_func_t)(const local_addr_t *local_addrs,
                                    const uint32_t *idxs,
                                    const swpl_split_info_t *slice_info,
                                    const void *fw_param, void *bw_param);
 
/**
 * \brief software pipeline store functor
 * \param system_addrs: type: const system_addr_t*
 * \param local_addrs: type: const local_addr_t*
 * \param idxs: type: const uint32_t*
 * \param slice_info: type: const swpl_split_info_t*
 * \param fw_param: type: const void*
 **/
typedef void (*swpl_store_func_t)(const system_addr_t *system_addrs,
                                  const local_addr_t *local_addrs,
                                  const uint32_t *idxs,
                                  const swpl_split_info_t *slice_info,
                                  const void *fw_param);
 
typedef void (*loop_3stage_func_t)(const swpl_load_func_t load_func,
                                   const swpl_compute_func_t compute_func,
                                   const swpl_store_func_t store_func,
                                   const system_addr_t *system_addrs,
                                   const swpl_lmem_scheme_t *lmem_scheme,
                                   const swpl_split_info_t *slice_info,
                                   const void *fw_param, void *bw_param);
 
typedef enum {
  BINARY_ADD = 0,
  BINARY_SUB = 1,
  BINARY_MUL = 2,
  BINARY_DIV = 3,
  BINARY_MAX = 4,
  BINARY_MIN = 10000,
  BINARY_GT = 10001,
  BINARY_GE = 10002,
  BINARY_LT = 10003,
  BINARY_LE = 10004,
  BINARY_EQ = 10005,
  BINARY_NE = 10006,
  BINARY_SQUARED_DIFF = 10007,
  BINARY_FLOOR_MOD = 10008,
  BINARY_FLOOR_DIV = 10009
} sg_binary_type_t;
 
static
void software_pipeline_3stage(const swpl_load_func_t load_func,
                              const swpl_compute_func_t compute_func,
                              const swpl_store_func_t store_func,
                              const system_addr_t *system_addrs,
                              const swpl_lmem_scheme_t *lmem_scheme,
                              const swpl_split_info_t *slice_info,
                              const void *fw_param, void *bw_param) {
  const local_addr_t *local_addrs = lmem_scheme->addrs;
  const int *mem_sizes = lmem_scheme->mem_sizes;
  const int num = lmem_scheme->num;
 
  const uint32_t *slices = slice_info->slices;
  const uint32_t *fulls = slice_info->fulls;
  const int rank = slice_info->rank;
  TPUKERNEL_ASSERT(rank > 0);
 
  local_addr_t pl_local_addrs[2][num];
  for (int i = 0; i < num; ++i) {
    pl_local_addrs[0][i] = local_addrs[i];
    pl_local_addrs[1][i] = local_addrs[i];
    if (mem_sizes[i] > 0)
      pl_local_addrs[1][i] += mem_sizes[i];
  }
 
  SWPL_STAGE_ARR_DECL(uint32_t, idxes, rank)
  SWPL_BEGIN(idxes, fulls)
  SWPL_PROC_BEGIN
  SWPL_STORE_BEGIN
  store_func(system_addrs, pl_local_addrs[$PINGPONG], idxes[$STAGE], slice_info,
             fw_param);
  SWPL_STORE_END
 
  SWPL_COMPUTE_BEGIN
  compute_func(pl_local_addrs[$PINGPONG], idxes[$STAGE], slice_info, fw_param,
               bw_param);
  SWPL_COMPUTE_END
 
  SWPL_LOAD_BEGIN
  load_func(system_addrs, pl_local_addrs[$PINGPONG], idxes[$STAGE], slice_info,
            fw_param);
  SWPL_LOAD_END
  SWPL_PROC_END
  SWPL_STAGE_ARR_MOVE(idxes, rank)
  SWPL_TRST_BEGIN
  SWPL_IDX_ARR_INC(idxes, slices, fulls, rank)
  SWPL_TRST_END
  SWPL_END
}
 
static
void plain_loop_3stage(const swpl_load_func_t load_func,
                       const swpl_compute_func_t compute_func,
                       const swpl_store_func_t store_func,
                       const system_addr_t *system_addrs,
                       const swpl_lmem_scheme_t *lmem_scheme,
                       const swpl_split_info_t *slice_info,
                       const void *fw_param, void *bw_param) {
  const local_addr_t *local_addrs = lmem_scheme->addrs;
 
  const uint32_t *slices = slice_info->slices;
  const uint32_t *fulls = slice_info->fulls;
  const int rank = slice_info->rank;
  TPUKERNEL_ASSERT(rank > 0);
 
  uint32_t idxes[rank];
  memset(idxes, 0, sizeof(idxes));
  while (idxes[0] < fulls[0]) {
    load_func(system_addrs, local_addrs, idxes, slice_info, fw_param);
    compute_func(local_addrs, idxes, slice_info, fw_param, bw_param);
    store_func(system_addrs, local_addrs, idxes, slice_info, fw_param);
    IDX_ARR_INC(idxes, slices, fulls, rank)
  }
}
 
static inline int _sign(int x) { return x > 0 ? +1 : -1; }
static inline int _abs(int x) { return x >= 0 ? x : -x; }
 
static inline void fp_sum2d(local_addr_t dst_addr, local_addr_t src_addr,
                            const dim4 *shape, const dim2 *kernel,
                            data_type_t dtype) {
  const scalar_t C = {.f32 = 1.0f};
  const padding_t padding = {0, 0, 0, 0};
  const dim2 stride = {1, 1};
  const dim2 dilation = {1, 1};
  tpu_bdc_fp_avg_pool2d(dst_addr, // 64-byte aligned
                        src_addr, // 64-byte aligned
                        shape, kernel, &padding, &stride, &dilation, dtype,
                        tpu_cast(C, dtype, DT_FP32, ROUND_MODE));
}
 
static inline void fp_glb_avg2d(local_addr_t dst_addr, local_addr_t src_addr,
                                const dim4 *shape, data_type_t dtype) {
  const scalar_t C = {.f32 = 1.0f / (shape->h * shape->w)};
  const dim2 kernel = {shape->h, shape->w};
  const padding_t padding = {0, 0, 0, 0};
  const dim2 stride = {1, 1};
  const dim2 dilation = {1, 1};
  tpu_bdc_fp_avg_pool2d(dst_addr, // 64-byte aligned
                        src_addr, // 64-byte aligned
                        shape, &kernel, &padding, &stride, &dilation, dtype,
                        tpu_cast(C, dtype, DT_FP32, ROUND_MODE));
}
 
static inline void inplace_fp_add_C(local_addr_t addr, const scalar_t C,
                                    const dim4 *shape, const dim4 *stride,
                                    data_type_t dtype) {
  tpu_bdc_fp_add_C(addr, addr, C, shape, stride, stride, dtype);
}
 
static inline void inplace_fp_mul_C(local_addr_t addr, const scalar_t C,
                                    const dim4 *shape, const dim4 *stride,
                                    data_type_t dtype) {
  tpu_bdc_fp_mul_C(addr, addr, C, shape, stride, stride, dtype);
}
 
static inline void inplace_fp_add(local_addr_t addr, local_addr_t addr2,
                                  const dim4 *shape, const dim4 *stride,
                                  const dim4 *stride2, data_type_t dtype) {
  tpu_bdc_fp_add(addr, addr, addr2, shape, stride, stride, stride2, dtype);
}
 
static inline void inplace_fp_sub(local_addr_t addr, local_addr_t addr2,
                                  const dim4 *shape, const dim4 *stride,
                                  const dim4 *stride2, data_type_t dtype) {
  tpu_bdc_fp_sub(addr, addr, addr2, shape, stride, stride, stride2, dtype);
}
 
static inline void inplace_fp_mul(local_addr_t addr, local_addr_t addr2,
                                  const dim4 *shape, const dim4 *stride,
                                  const dim4 *stride2, data_type_t dtype) {
  tpu_bdc_fp_mul(addr, addr, addr2, shape, stride, stride, stride2, dtype);
}
 
static inline void inplace_fp_rsqrt(local_addr_t addr, local_addr_t buffer_addr,
                                    const dim4 *shape, data_type_t dtype) {
#ifdef __bm1686__
  UNUSED(buffer_addr);
  tpu_bdc_fp_rsqrt(addr, addr, shape, dtype);
#else // __bm1684x__
  const local_addr_t fp32_addr = dtype == DT_FP32 ? addr : buffer_addr;
  if (dtype != DT_FP32) {
    tpu_bdc_cast(buffer_addr, addr, shape, NULL, NULL, DT_FP32, dtype,
                 ROUND_MODE);
  }
  tpu_bdc_fp32_tunable_rsqrt(fp32_addr, fp32_addr, shape, RSQRT_NUM_ITER);
  if (dtype != DT_FP32) {
    tpu_bdc_cast(addr, buffer_addr, shape, NULL, NULL, dtype, DT_FP32,
                 ROUND_MODE);
  }
#endif
}
 
/*  */
 
#define GN_HAVE_WEIGHT 0x01
#define GN_HAVE_BIAS 0x02
#define GN_NEED_MEAN 0x04
#define GN_NEED_RSTD 0x08
 
typedef enum {
  GROUP_NORM_MODE,
  LAYER_NORM_MODE,
  RMS_NORM_MODE,
} gn_mode_t;
 
typedef struct {
  global_addr_t input_addr;
  global_addr_t weight_addr;
  global_addr_t bias_addr;
  global_addr_t output_addr;
  global_addr_t mean_addr;
  global_addr_t rstd_addr;
} glb_scheme_t;
 
typedef struct {
  local_addr_t input_addr;
  local_addr_t weight_addr;
  local_addr_t bias_addr;
  local_addr_t buffer_addr;
  local_addr_t output_addr;
  local_addr_t mean_addr;
  local_addr_t rstd_addr;
} loc_scheme_t;
 
typedef struct {
  int input_mem_sz;
  int weight_mem_sz;
  int bias_mem_sz;
  int buffer_mem_sz;
  int output_mem_sz;
  int mean_mem_sz;
  int rstd_mem_sz;
} loc_mem_sz_t;
 
/**
 *  @author: shunrong.qian
 *  @brief automatic assign local address, not concerning bank conflict
 *  @param local_addrs: input as index map, output as real local address
 *  @param mem_sizes: input as weights, output as real memory size
 *  @param num: length of `local_addrs` and `mem_sizes`
 *  @param scale: divide local mem into `scale * LOCAL_MEM_BANKS` blocks
 *  @param use_swpl: whether to use software pipeline
 **/
static inline void auto_partition(local_addr_t *local_addrs, int *mem_sizes,
                                  const int num, int scale, bool use_swpl) {
  float weights_[num];
  /**
   * invar: [=========][==========]
   *         real_rank  extra_rank
   **/
  int invar[num];
  float sum_w = 0;
  int real_rank = 0;
  for (int i = 0; i < num; ++i) {
    if ((int)local_addrs[i] == i) {
      invar[real_rank] = i;
      int weight = _abs(mem_sizes[i]);
      if (mem_sizes[i] > 0) {
        weight *= 2;
      }
      weights_[real_rank] = (float)weight;
      sum_w += (float)weight;
      ++real_rank;
    }
  }
  if (sum_w == 0)
    return;
  int ng = scale * LOCAL_MEM_BANKS;
  TPUKERNEL_ASSERT_INFO(real_rank <= ng, "auto partition failed!\n");
  int extra_rank = 0;
  for (int i = 0; i < num; ++i) {
    if ((int)local_addrs[i] >= num) { // set invalid index to -1
      local_addrs[i] = -1;
      mem_sizes[i] = 0;
    } else if ((int)local_addrs[i] != i) {
      invar[real_rank + extra_rank] = i;
      ++extra_rank;
    }
  }
  for (int i = 0; i < real_rank; ++i) {
    weights_[i] /= sum_w;
    weights_[i] *= ng;
  }
  int mem_sizes_[num];
  for (int i = 0; i < real_rank; ++i) {
    mem_sizes_[i] = 1;
  }
  sum_w = 0;
  for (int i = 0; i < real_rank; ++i) {
    weights_[i] -= 1;
    weights_[i] = MAX(weights_[i], 0);
    sum_w += weights_[i];
  }
  ng -= real_rank;
  for (int i = 0; i < real_rank; ++i) {
    //mem_sizes_[i] += (int)floorf(ng * weights_[i] / sum_w);
  }
  const int mem_unit = BANK_SIZE / scale;
  for (int i = 0; i < real_rank; ++i) {
    mem_sizes_[i] *= mem_unit;
  }
  // assign local mem addrs
  int running_addr = 0;
  for (int i = 0; i < real_rank; ++i) {
    const int j = invar[i];
    local_addrs[j] = running_addr;
    mem_sizes[j] = _sign(mem_sizes[j]);
    mem_sizes[j] *= mem_sizes_[i];
    running_addr += mem_sizes_[i];
  }
  for (int i = real_rank; i < real_rank + extra_rank; ++i) {
    const int j = invar[i];
    const int idx = local_addrs[j];
    local_addrs[j] = local_addrs[idx];
    mem_sizes[j] = _sign(mem_sizes[j]);
    mem_sizes[j] *= _abs(mem_sizes[idx]);
  }
  if (use_swpl) {
    for (int i = 0; i < num; ++i) {
      if (mem_sizes[i] > 0) {
        mem_sizes[i] /= 2;
      }
    }
  }
}
 
static inline void
layer_norm_scale_bias_local(local_addr_t weight_addr, // aligned
                            local_addr_t bias_addr,   // aligned
                            local_addr_t output_addr, // aligned
                            int channel, int height, int ex_affine,
                            data_type_t dtype) {
  const dim4 cbstr = {0, 0, 1, 0};
  const dim4 ishape = {1, channel, height, 1};
  const dim4 ishape_m = {1, MIN(channel, NPU_NUM), height, 1};
  if (ex_affine & 1) {
    // xc * weight
    tpu_bdc_npu_bcast(weight_addr, weight_addr, &ishape_m, dtype);
    inplace_fp_mul(output_addr, weight_addr, &ishape, NULL, &cbstr, dtype);
  }
  if (ex_affine & 2) {
    // xc * weight + bias
    tpu_bdc_npu_bcast(bias_addr, bias_addr, &ishape_m, dtype);
    inplace_fp_add(output_addr, bias_addr, &ishape, NULL, &cbstr, dtype);
  }
}
 
/************************ c split ************************/
 
static inline void lmem_alloc_c_split(int channel, int height, int ex_affine,
                                      data_type_t dtype, loc_scheme_t *sm,
                                      loc_mem_sz_t *sz, bool use_swpl) {
  const int eu_num = tpu_eu_num(dtype);
  const int type_len = tpu_data_type_size(dtype);
  const int num = sizeof(loc_scheme_t) / sizeof(local_addr_t);
  // init `sm`
  for (int i = 0; i < num; ++i) {
    *((int *)sm + i) = i;
  }
  // invalidate weight
  if (!(ex_affine & GN_HAVE_WEIGHT)) {
    sm->weight_addr = -1;
  }
  // invalidate bias
  if (!(ex_affine & GN_HAVE_BIAS)) {
    sm->bias_addr = -1;
  }
  // init `sz`
  sz->input_mem_sz =
      DIV_UP(channel, NPU_NUM) * ALIGN(height, eu_num) * type_len;
  sz->output_mem_sz = sz->input_mem_sz;
  sz->weight_mem_sz = DIV_UP(channel, NPU_NUM) * eu_num * type_len;
  sz->bias_mem_sz = sz->weight_mem_sz;
  sz->mean_mem_sz = ALIGN(height, eu_num) * type_len;
  sz->rstd_mem_sz = sz->mean_mem_sz;
  if (!(ex_affine & GN_NEED_MEAN)) {
    sm->rstd_addr = sm->mean_addr;
  }
  /**
   * @brief: A trick on memory usage.
   * @author: shunrong.qian
   * For bm1684x, in fp16/bf16 cases, one needs 2x input/output memory size to
   *store rsqrt results, so at the same time buffer mem and output mem could be
   *used to do this for reducing memory usage. For memory continuity, buffer mem
   *and output mem should be placed adjacently.
   **/
  sz->buffer_mem_sz = sz->input_mem_sz;
  auto_partition((local_addr_t *)sm, (int *)sz, num, 1, use_swpl);
}
 
static int calc_slice_c_split(int channel, int height, data_type_t dtype,
                              int ex_affine, const loc_mem_sz_t *m,
                              int *p_cslice) {
  /**
   * 1. DIV_UP(cslice, NPU_NUM) * ALIGN(hslice, eu_num(dtype)) * WIDTH(dtype) <=
   *mem_sz_0
   * 2. DIV_UP(cslice, NPU_NUM) * eu_num(dtype) * WIDTH(dtype) <= mem_sz_1
   * 3. ALIGN(hslice, eu_num(dtype)) * WIDTH(dtype) <= mem_sz_2
   **/
 
  const int eu_num = tpu_eu_num(dtype);
  const int type_len = tpu_data_type_size(dtype);
  int max_hslice = MIN(height, POOL_MAX_KSIZE);
  if (ex_affine & (GN_HAVE_WEIGHT | GN_HAVE_BIAS)) {
    int mem_sz_2 = INT32_MAX;
    if (ex_affine & GN_HAVE_WEIGHT)
      mem_sz_2 = MIN(mem_sz_2, m->weight_mem_sz);
    if (ex_affine & GN_HAVE_BIAS)
      mem_sz_2 = MIN(mem_sz_2, m->bias_mem_sz);
    max_hslice = MIN(max_hslice, ALIGN_DOWN(mem_sz_2 / type_len, eu_num));
  }
  if (max_hslice < height)
    return -1;
  int max_kslice = DIV_UP(channel, NPU_NUM);
  if (ex_affine & (GN_NEED_MEAN | GN_NEED_RSTD)) {
    int mem_sz_1 = INT32_MAX;
    if (ex_affine & GN_NEED_MEAN)
      mem_sz_1 = MIN(mem_sz_1, m->mean_mem_sz);
    if (ex_affine & GN_NEED_RSTD)
      mem_sz_1 = MIN(mem_sz_1, m->rstd_mem_sz);
    max_kslice = MIN(max_kslice, mem_sz_1 / type_len / eu_num);
  }
  if (max_kslice <= 0)
    return -1;
  const int mem_sz_0 =
      MIN(MIN(m->input_mem_sz, m->output_mem_sz), m->buffer_mem_sz);
  int kslice = MIN(max_kslice, mem_sz_0 / type_len / ALIGN(height, eu_num));
  if (kslice <= 0)
    return -1;
  *p_cslice = MIN(kslice * NPU_NUM, channel);
  return 0;
}
 
typedef struct {
  gn_mode_t mode;
  int height;
  int ex_affine;
  float eps;
  data_type_t dtype;
  dim4 *p_glb_istr;
  int rn;
  int rc;
  int rh;
} c_split_fw_param_t;
 
static inline void
gn_load__ch_split_c_split(const system_addr_t *system_addrs,
                          const local_addr_t *local_addrs, const uint32_t *idxs,
                          const swpl_split_info_t *slice_info,
                          const void *fw_param) {
  const glb_scheme_t *g = (glb_scheme_t *)system_addrs;
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const c_split_fw_param_t *p = (c_split_fw_param_t *)fw_param;
  const int cidx = idxs[0];
  const int cslice = slice_info->slices[0];
  const int channel = slice_info->fulls[0];
  const int real_cslice = MIN(channel - cidx, cslice);
  const int type_len = tpu_data_type_size(p->dtype);
  const dim4 ishape = {1, real_cslice, p->height, 1};
  tpu_gdma_cpy_S2L(l->input_addr,
                   g->input_addr + (cidx * p->p_glb_istr->c) * type_len,
                   &ishape, NULL, p->p_glb_istr, p->dtype);
  if (p->mode == LAYER_NORM_MODE || p->mode == RMS_NORM_MODE) {
    const dim4 wshape = {1, 1, p->height, 1};
    if (p->ex_affine & GN_HAVE_WEIGHT) {
      tpu_gdma_cpy_S2L(l->weight_addr, g->weight_addr, &wshape, NULL, NULL,
                       p->dtype);
    }
    if (p->ex_affine & GN_HAVE_BIAS) {
      tpu_gdma_cpy_S2L(l->bias_addr, g->bias_addr, &wshape, NULL, NULL,
                       p->dtype);
    }
  }
}
 
static inline void group_norm_2d_local(
    local_addr_t input_addr,  // aligned, but could be overwritten
    local_addr_t weight_addr, // aligned
    local_addr_t bias_addr,   // aligned
    local_addr_t output_addr, // aligned
    local_addr_t mean_addr,   // aligned
    local_addr_t invstd_addr, // aligned
    local_addr_t buffer_addr, int channel, int height, float eps, int ex_affine,
    data_type_t dtype, gn_mode_t mode) {
  const int eu_num = tpu_eu_num(dtype);
  const dim4 ishape = {1, channel, height, 1};
  const dim4 mshape = {1, channel, 1, 1};
  const dim4 hbstr = {0, eu_num, 0, 0};
 
  if (mode != RMS_NORM_MODE) {
    // mean
    fp_glb_avg2d(mean_addr, input_addr, &ishape, dtype);
 
    // x - mean
    inplace_fp_sub(input_addr, mean_addr, &ishape, NULL, &hbstr, dtype);
  }
 
  // (x - mean)^2. x^2 for RMSNorm
  tpu_bdc_fp_square(buffer_addr, input_addr, &ishape, NULL, NULL, dtype);
 
  // var
  fp_glb_avg2d(invstd_addr, buffer_addr, &ishape, dtype);
 
  // var + eps
  const scalar_t C = {.f32 = eps};
  inplace_fp_add_C(invstd_addr, tpu_cast(C, dtype, DT_FP32, ROUND_MODE),
                   &mshape, NULL, dtype);
 
  // invstd := 1 / sqrt(var + eps)
  inplace_fp_rsqrt(invstd_addr, buffer_addr, &mshape, dtype);
 
  // xc := (x - mean) * invstd. x * invstd for RMSNorm
  tpu_bdc_fp_mul(output_addr, input_addr, invstd_addr, &ishape, NULL, NULL,
                 &hbstr, dtype);
}
 
static inline void gn_compute_c_split(const local_addr_t *local_addrs,
                                      const uint32_t *idxs,
                                      const swpl_split_info_t *slice_info,
                                      const void *fw_param, void *bw_param) {
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const c_split_fw_param_t *p = (c_split_fw_param_t *)fw_param;
  const int cidx = idxs[0];
  const int cslice = slice_info->slices[0];
  const int channel = slice_info->fulls[0];
  const int real_cslice = MIN(channel - cidx, cslice);
  group_norm_2d_local(l->input_addr, l->weight_addr, l->bias_addr,
                      l->output_addr, l->mean_addr, l->rstd_addr,
                      l->buffer_addr, real_cslice, p->height, p->eps,
                      p->ex_affine, p->dtype, p->mode);
  if (p->mode == LAYER_NORM_MODE || p->mode == RMS_NORM_MODE) {
    layer_norm_scale_bias_local(l->weight_addr, l->bias_addr, l->output_addr,
                                real_cslice, p->height, p->ex_affine, p->dtype);
  }
}
 
static inline void gn_store__ch_split_c_split(
    const system_addr_t *system_addrs, const local_addr_t *local_addrs,
    const uint32_t *idxs, const swpl_split_info_t *slice_info,
    const void *fw_param) {
  const glb_scheme_t *g = (glb_scheme_t *)system_addrs;
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const c_split_fw_param_t *p = (c_split_fw_param_t *)fw_param;
  const int cidx = idxs[0];
  const int cslice = slice_info->slices[0];
  const int channel = slice_info->fulls[0];
  const int real_cslice = MIN(channel - cidx, cslice);
  const int type_len = tpu_data_type_size(p->dtype);
  const dim4 ishape = {1, real_cslice, p->height, 1};
  tpu_gdma_cpy_L2S(g->output_addr + (cidx * p->p_glb_istr->c) * type_len,
                   l->output_addr, &ishape, p->p_glb_istr, NULL, p->dtype);
  const dim4 mshape = {1, real_cslice, 1, 1};
  if (p->ex_affine & GN_NEED_MEAN) {
    tpu_gdma_cpy_L2S(g->mean_addr + cidx * type_len, l->mean_addr, &mshape,
                     NULL, NULL, p->dtype);
  }
  if (p->ex_affine & GN_NEED_RSTD) {
    tpu_gdma_cpy_L2S(g->rstd_addr + cidx * type_len, l->rstd_addr, &mshape,
                     NULL, NULL, p->dtype);
  }
}
 
static void group_norm__c_split(const global_addr_t *global_addrs,
                                const swpl_lmem_scheme_t *lmem_scheme,
                                const c_split_fw_param_t *fw_param, int channel,
                                int cslice, gn_mode_t mode) {
  swpl_split_info_t slice_info;
  slice_info.rank = 1;
  slice_info.fulls[0] = channel;
  slice_info.slices[0] = cslice;
 
  loop_3stage_func_t loop_3stage_func = plain_loop_3stage;
#ifdef USE_PIPELINE
  loop_3stage_func = software_pipeline_3stage;
#endif
 
  loop_3stage_func(gn_load__ch_split_c_split, gn_compute_c_split,
                   gn_store__ch_split_c_split, global_addrs, lmem_scheme,
                   &slice_info, fw_param, NULL);
 
  if (mode == GROUP_NORM_MODE) {
    const glb_scheme_t *g = (glb_scheme_t *)global_addrs;
    const c_split_fw_param_t *p = (c_split_fw_param_t *)fw_param;
 
    if (p->ex_affine & GN_HAVE_WEIGHT && p->ex_affine & GN_HAVE_BIAS) {
#ifdef DEBUG_GN
      TPUKERNEL_DBG("++++++++++++++++++++++++I'm "
                    "here++++++++++++++++++++++++++++c split calling scale\n");
#endif
      nodechip_scale_forward(g->output_addr, g->weight_addr, g->bias_addr,
                             g->output_addr, p->rn, p->rc, p->rh, 1, 1, 1, true,
                             false, 0, false, p->dtype);
    } else {
      int io_shape[3] = {p->rn, p->rc, p->rh};
      int wb_shape[3] = {1, p->rc, 1};
      if (p->ex_affine & GN_HAVE_WEIGHT) {
        nodechip_bcbinary_fp(g->output_addr, g->weight_addr, g->output_addr,
                             io_shape, wb_shape, 3, 3, BINARY_MUL, p->dtype,
                             false, 0);
      }
      if (p->ex_affine & GN_HAVE_BIAS) {
        nodechip_bcbinary_fp(g->output_addr, g->bias_addr, g->output_addr,
                             io_shape, wb_shape, 3, 3, BINARY_ADD, p->dtype,
                             false, 0);
      }
    }
  }
}
 
/************************ c-h split ************************/
 
static inline void standardize_2d_local(
    local_addr_t input_addr,  // aligned, but could be overwritten
    local_addr_t output_addr, // aligned
    local_addr_t mean_addr,   // aligned
    local_addr_t rstd_addr,   // aligned
    int channel, int height, data_type_t dtype, gn_mode_t mode) {
  const int eu_num = tpu_eu_num(dtype);
  const dim4 ishape = {1, channel, height, 1};
  const dim4 hbstr = {0, eu_num, 0, 0};
 
  if (mode != RMS_NORM_MODE) {
    // x - mean
    inplace_fp_sub(input_addr, mean_addr, &ishape, NULL, &hbstr, dtype);
  }
 
  // xc := (x - mean) * rstd
  tpu_bdc_fp_mul(output_addr, input_addr, rstd_addr, &ishape, NULL, NULL,
                 &hbstr, dtype);
}
 
static inline void lmem_alloc__ch_split(int channel, int height, int ex_affine,
                                        data_type_t dtype, loc_scheme_t *sm,
                                        loc_mem_sz_t *sz, bool use_swpl) {
  const int eu_num = tpu_eu_num(dtype);
  const int type_len = tpu_data_type_size(dtype);
  const int num = sizeof(loc_scheme_t) / sizeof(local_addr_t);
  // init `sm`
  for (int i = 0; i < num; ++i) {
    *((int *)sm + i) = i;
  }
  // invalidate weight
  if (!(ex_affine & GN_HAVE_WEIGHT)) {
    sm->weight_addr = -1;
  }
  // invalidate bias
  if (!(ex_affine & GN_HAVE_BIAS)) {
    sm->bias_addr = -1;
  }
  // init `sz`
  sz->input_mem_sz =
      DIV_UP(channel, NPU_NUM) * ALIGN(height, eu_num) * type_len;
  sz->output_mem_sz = sz->input_mem_sz;
  sz->weight_mem_sz = DIV_UP(channel, NPU_NUM) * eu_num * type_len;
  sz->bias_mem_sz = sz->weight_mem_sz;
  sz->mean_mem_sz =
      -ALIGN(height, eu_num) * type_len; // force not to be pingpong array
  sz->rstd_mem_sz = sz->mean_mem_sz;
  /**
   * @brief: A trick on memory usage.
   * @author: shunrong.qian
   * For bm1684x, in fp16/bf16 cases, one needs 2x input/output memory size to
   *store rsqrt results, so at the same time buffer mem and output mem could be
   *used to do this for reducing memory usage. For memory continuity, buffer mem
   *and output mem should be placed adjacently.
   **/
  sz->buffer_mem_sz = sz->input_mem_sz;
  auto_partition((local_addr_t *)sm, (int *)sz, num, 1, use_swpl);
}
 
static inline int calc_slice__ch_split(int channel, int height,
                                       data_type_t dtype, int ex_affine,
                                       const loc_mem_sz_t *m, int *p_cslice,
                                       int *p_hslice) {
  /**
   * 1. DIV_UP(cslice, NPU_NUM) * pslice * eu_num(dtype) * WIDTH(dtype) <=
   *mem_sz_0
   * 2. DIV_UP(cslice, NPU_NUM) * eu_num(dtype) * WIDTH(dtype) <= mem_sz_1
   *(mean/rstd)
   * 3. pslice * eu_num(dtype) * WIDTH(dtype) <= mem_sz_2 (weight/bias)
   **/
 
  const int mem_sz_0 =
      MIN(MIN(m->input_mem_sz, m->output_mem_sz), m->buffer_mem_sz);
  const int eu_num = tpu_eu_num(dtype);
  const int type_len = tpu_data_type_size(dtype);
  int max_kslice = DIV_UP(channel, NPU_NUM);
  if (ex_affine & (GN_NEED_MEAN | GN_NEED_RSTD)) {
    int mem_sz_1 = INT32_MAX;
    if (ex_affine & GN_NEED_MEAN)
      mem_sz_1 = MIN(mem_sz_1, m->mean_mem_sz);
    if (ex_affine & GN_NEED_RSTD)
      mem_sz_1 = MIN(mem_sz_1, m->rstd_mem_sz);
    max_kslice = MIN(max_kslice, mem_sz_1 / type_len / eu_num);
  }
  int max_pslice = DIV_UP(MIN(height, POOL_MAX_KSIZE), eu_num);
  if (ex_affine & (GN_HAVE_WEIGHT | GN_HAVE_BIAS)) {
    int mem_sz_2 = INT32_MAX;
    if (ex_affine & GN_HAVE_WEIGHT)
      mem_sz_2 = MIN(mem_sz_2, m->weight_mem_sz);
    if (ex_affine & GN_HAVE_BIAS)
      mem_sz_2 = MIN(mem_sz_2, m->bias_mem_sz);
    max_pslice = MIN(max_pslice, mem_sz_2 / type_len / eu_num);
  }
  int pslice = max_pslice;
  int kslice = MIN(max_kslice, mem_sz_0 / type_len / pslice / eu_num);
  TPUKERNEL_ASSERT(pslice > 0);
  if (kslice <= 0) {
    kslice = 1;
    pslice = MIN(max_pslice, mem_sz_0 / type_len / kslice / eu_num);
    TPUKERNEL_ASSERT(pslice > 0);
  }
  *p_cslice = MIN(kslice * NPU_NUM, channel);
  const int pieces = DIV_UP(height, eu_num);
  int psecs = DIV_UP(pieces, pslice);
  pslice = DIV_UP(pieces, psecs);
  *p_hslice = pslice * eu_num;
  return 0;
}
 
typedef struct {
  gn_mode_t mode;
  int ex_affine;
  float eps;
  data_type_t dtype;
  dim4 *p_glb_istr;
  int cidx;
  int real_cslice;
  dim4 *p_loc_istr;
  int rn;
  int rc;
  int rh;
} ch_split_fw_param_t;
 
static inline void rd_init(const local_addr_t *local_addrs,
                           const void *fw_param) {
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int eu_num = tpu_eu_num(p->dtype);
  const dim4 mshape = {1, p->real_cslice, 1, eu_num};
  const scalar_t C = {0};
  tpu_bdc_set_C(l->mean_addr, C, &mshape, NULL, p->dtype);
  tpu_bdc_set_C(l->rstd_addr, C, &mshape, NULL, p->dtype);
}
 
static inline void rd_load(const system_addr_t *system_addrs,
                           const local_addr_t *local_addrs,
                           const uint32_t *idxs,
                           const swpl_split_info_t *slice_info,
                           const void *fw_param) {
  const glb_scheme_t *g = (glb_scheme_t *)system_addrs;
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int hidx = idxs[0];
  const int hslice = slice_info->slices[0];
  const int height = slice_info->fulls[0];
  const int real_hslice = MIN(height - hidx, hslice);
  const int type_len = tpu_data_type_size(p->dtype);
  const dim4 ishape = {1, p->real_cslice, real_hslice, 1};
  tpu_gdma_cpy_S2L(l->input_addr,
                   g->input_addr +
                       (p->cidx * p->p_glb_istr->c + hidx) * type_len,
                   &ishape, p->p_loc_istr, p->p_glb_istr, p->dtype);
}
 
static inline void rd_compute_mean_halfway(const local_addr_t *local_addrs,
                                           const uint32_t *idxs,
                                           const swpl_split_info_t *slice_info,
                                           const void *fw_param,
                                           void *bw_param) {
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int eu_num = tpu_eu_num(p->dtype);
  const int hidx = idxs[0];
  const int hslice = slice_info->slices[0];
  const int height = slice_info->fulls[0];
  const int real_hslice = MIN(height - hidx, hslice);
  const int pslice = DIV_UP(hslice, eu_num);
  const dim4 ishape = {1, p->real_cslice, pslice, eu_num};
  const dim4 kshape = {1, p->real_cslice, 1, eu_num};
  const int type_len = tpu_data_type_size(p->dtype);
  // if h is not enough, pad with zeros
  if (real_hslice < hslice) {
    const scalar_t C = {0};
    const dim4 pad_shape = {1, p->real_cslice, 1, hslice - real_hslice};
    tpu_bdc_set_C(l->input_addr + real_hslice * type_len, C, &pad_shape,
                  p->p_loc_istr, p->dtype);
  }
  const dim2 kernel = {pslice, 1};
  // sum_h x_h
  fp_sum2d(l->output_addr, l->input_addr, &ishape, &kernel, p->dtype);
  inplace_fp_add(l->mean_addr, l->output_addr, &kshape, NULL, NULL, p->dtype);
}
 
static inline void rd_compute_rstd_halfway(const local_addr_t *local_addrs,
                                           const uint32_t *idxs,
                                           const swpl_split_info_t *slice_info,
                                           const void *fw_param,
                                           void *bw_param) {
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int eu_num = tpu_eu_num(p->dtype);
  const int hidx = idxs[0];
  const int hslice = slice_info->slices[0];
  const int height = slice_info->fulls[0];
  const int real_hslice = MIN(height - hidx, hslice);
  const int pslice = DIV_UP(hslice, eu_num);
  const dim4 ishape = {1, p->real_cslice, pslice, eu_num};
  const dim4 kshape = {1, p->real_cslice, 1, eu_num};
  const dim4 hbstr = {0, eu_num, 0, 0};
  const int type_len = tpu_data_type_size(p->dtype);
  if (p->mode != RMS_NORM_MODE) {
    inplace_fp_sub(l->input_addr, l->mean_addr, &ishape, NULL, &hbstr,
                   p->dtype);
  }
  // if h is not enough, pad with zeros
  if (real_hslice < hslice) {
    const scalar_t C = {0};
    const dim4 pad_shape = {1, p->real_cslice, 1, hslice - real_hslice};
    tpu_bdc_set_C(l->input_addr + real_hslice * type_len, C, &pad_shape,
                  p->p_loc_istr, p->dtype);
  }
  tpu_bdc_fp_square(l->input_addr, l->input_addr, &ishape, NULL, NULL,
                    p->dtype);
  // sum_w (x_w - mean)^2
  const dim2 kernel = {pslice, 1};
  fp_sum2d(l->output_addr, l->input_addr, &ishape, &kernel, p->dtype);
  inplace_fp_add(l->rstd_addr, l->output_addr, &kshape, NULL, NULL, p->dtype);
}
 
static inline void rd_mean_end(const local_addr_t *local_addrs,
                               const swpl_split_info_t *slice_info,
                               const void *fw_param) {
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int eu_num = tpu_eu_num(p->dtype);
  const dim4 kshape = {1, p->real_cslice, 1, eu_num};
  const dim4 mshape = {1, p->real_cslice, 1, 1};
  const int height = slice_info->fulls[0];
  const dim2 kernel = {1, eu_num};
  const scalar_t C = {.f32 = 1.0f / height};
  // sum_i x_i
  fp_sum2d(l->output_addr, l->mean_addr, &kshape, &kernel, p->dtype);
  // mean := (sum_i x_i) / n
  tpu_bdc_fp_mul_C(l->mean_addr, l->output_addr,
                   tpu_cast(C, p->dtype, DT_FP32, ROUND_MODE), &mshape, NULL,
                   NULL, p->dtype);
}
 
static inline void rd_rstd_end(const local_addr_t *local_addrs,
                               const swpl_split_info_t *slice_info,
                               const void *fw_param) {
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int eu_num = tpu_eu_num(p->dtype);
  const dim4 kshape = {1, p->real_cslice, 1, eu_num};
  const dim4 mshape = {1, p->real_cslice, 1, 1};
  const int height = slice_info->fulls[0];
  const dim2 kernel = {1, eu_num};
  scalar_t C;
  // sum_i (x_i-mean)^2
  C.f32 = 1.0f / height;
  fp_sum2d(l->output_addr, l->rstd_addr, &kshape, &kernel, p->dtype);
  // var := (sum_i (x_i-mean)^2) / n
  tpu_bdc_fp_mul_C(l->rstd_addr, l->output_addr,
                   tpu_cast(C, p->dtype, DT_FP32, ROUND_MODE), &mshape, NULL,
                   NULL, p->dtype);
  // var + eps
  C.f32 = p->eps;
  inplace_fp_add_C(l->rstd_addr, tpu_cast(C, p->dtype, DT_FP32, ROUND_MODE),
                   &mshape, NULL, p->dtype);
  // rstd := 1 / sqrt(var + eps)
  inplace_fp_rsqrt(l->rstd_addr, l->buffer_addr, &mshape, p->dtype);
}
 
static inline void gn_load__c_split(const system_addr_t *system_addrs,
                                    const local_addr_t *local_addrs,
                                    const uint32_t *idxs,
                                    const swpl_split_info_t *slice_info,
                                    const void *fw_param) {
  const glb_scheme_t *g = (glb_scheme_t *)system_addrs;
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int hidx = idxs[0];
  const int hslice = slice_info->slices[0];
  const int height = slice_info->fulls[0];
  const int real_hslice = MIN(height - hidx, hslice);
  const int type_len = tpu_data_type_size(p->dtype);
  const dim4 ishape = {1, p->real_cslice, real_hslice, 1};
  tpu_gdma_cpy_S2L(l->input_addr,
                   g->input_addr +
                       (p->cidx * p->p_glb_istr->c + hidx) * type_len,
                   &ishape, NULL, p->p_glb_istr, p->dtype);
  if (p->mode == LAYER_NORM_MODE || p->mode == RMS_NORM_MODE) {
    const dim4 wshape = {1, 1, real_hslice, 1};
    if (p->ex_affine & GN_HAVE_WEIGHT) {
      tpu_gdma_cpy_S2L(l->weight_addr, g->weight_addr + hidx * type_len,
                       &wshape, NULL, NULL, p->dtype);
    }
    if (p->ex_affine & GN_HAVE_BIAS) {
      tpu_gdma_cpy_S2L(l->bias_addr, g->bias_addr + hidx * type_len, &wshape,
                       NULL, NULL, p->dtype);
    }
  }
}
 
static inline void gn_load__ch_split(const system_addr_t *system_addrs,
                                     const local_addr_t *local_addrs,
                                     const uint32_t *idxs,
                                     const swpl_split_info_t *slice_info,
                                     const void *fw_param) {
  const glb_scheme_t *g = (glb_scheme_t *)system_addrs;
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int hidx = idxs[0];
  const int hslice = slice_info->slices[0];
  const int height = slice_info->fulls[0];
  const int real_hslice = MIN(height - hidx, hslice);
  const int type_len = tpu_data_type_size(p->dtype);
  const dim4 ishape = {1, p->real_cslice, real_hslice, 1};
  tpu_gdma_cpy_S2L(l->input_addr,
                   g->input_addr +
                       (p->cidx * p->p_glb_istr->c + hidx) * type_len,
                   &ishape, NULL, p->p_glb_istr, p->dtype);
}
 
static inline void gn_compute__c_split(const local_addr_t *local_addrs,
                                       const uint32_t *idxs,
                                       const swpl_split_info_t *slice_info,
                                       const void *fw_param, void *bw_param) {
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int hidx = idxs[0];
  const int hslice = slice_info->slices[0];
  const int height = slice_info->fulls[0];
  const int real_hslice = MIN(height - hidx, hslice);
  standardize_2d_local(l->input_addr, l->output_addr, l->mean_addr,
                       l->rstd_addr, p->real_cslice, real_hslice, p->dtype,
                       p->mode);
  if (p->mode == LAYER_NORM_MODE || p->mode == RMS_NORM_MODE) {
    layer_norm_scale_bias_local(l->weight_addr, l->bias_addr, l->output_addr,
                                p->real_cslice, real_hslice, p->ex_affine,
                                p->dtype);
  }
}
 
static inline void gn_store__c_split(const system_addr_t *system_addrs,
                                     const local_addr_t *local_addrs,
                                     const uint32_t *idxs,
                                     const swpl_split_info_t *slice_info,
                                     const void *fw_param) {
  const glb_scheme_t *g = (glb_scheme_t *)system_addrs;
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int hidx = idxs[0];
  const int hslice = slice_info->slices[0];
  const int height = slice_info->fulls[0];
  const int real_hslice = MIN(height - hidx, hslice);
  const int type_len = tpu_data_type_size(p->dtype);
  const dim4 ishape = {1, p->real_cslice, real_hslice, 1};
  tpu_gdma_cpy_L2S(g->output_addr +
                       (p->cidx * p->p_glb_istr->c + hidx) * type_len,
                   l->output_addr, &ishape, p->p_glb_istr, NULL, p->dtype);
}
 
static inline void
gn_store_optional__ch_split(const system_addr_t *system_addrs,
                            const local_addr_t *local_addrs,
                            const void *fw_param) {
  const glb_scheme_t *g = (glb_scheme_t *)system_addrs;
  const loc_scheme_t *l = (loc_scheme_t *)local_addrs;
  const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
  const int type_len = tpu_data_type_size(p->dtype);
  const dim4 mshape = {1, p->real_cslice, 1, 1};
  if (p->ex_affine & GN_NEED_MEAN) {
    tpu_gdma_cpy_L2S(g->mean_addr + p->cidx * type_len, l->mean_addr, &mshape,
                     NULL, NULL, p->dtype);
  }
  if (p->ex_affine & GN_NEED_RSTD) {
    tpu_gdma_cpy_L2S(g->rstd_addr + p->cidx * type_len, l->rstd_addr, &mshape,
                     NULL, NULL, p->dtype);
  }
}
 
static
void swpl_do_nothing(const system_addr_t *system_addrs,
                     const local_addr_t *local_addrs, const uint32_t *idxs,
                     const swpl_split_info_t *slice_info,
                     const void *fw_param) {
  UNUSED(system_addrs);
  UNUSED(local_addrs);
  UNUSED(idxs);
  UNUSED(slice_info);
  UNUSED(fw_param);
}
 
static void group_norm__ch_split(const global_addr_t *global_addrs,
                                 const swpl_lmem_scheme_t *lmem_scheme,
                                 ch_split_fw_param_t *fw_param, int channel,
                                 int cslice, int height, int hslice,
                                 gn_mode_t mode) {
  swpl_split_info_t slice_info;
  slice_info.rank = 1;
  slice_info.fulls[0] = height;
  slice_info.slices[0] = hslice;
 
  loop_3stage_func_t loop_3stage_func = plain_loop_3stage;
#ifdef USE_PIPELINE
  loop_3stage_func = software_pipeline_3stage;
#endif
 
  for (int cidx = 0; cidx < channel; cidx += cslice) {
    const int real_cslice = MIN(channel - cidx, cslice);
 
    const dim4 loc_ishape = {1, real_cslice, hslice, 1};
    dim4 loc_istr;
    tpu_aligned_stride(&loc_istr, 0, &loc_ishape, fw_param->dtype);
 
    fw_param->cidx = cidx;
    fw_param->real_cslice = real_cslice;
    fw_param->p_loc_istr = &loc_istr;
 
    /**
     * @author: shunrong.qian
     * @note: Note that single-pass algorithm of variance, e.g.
     *  D(x) := E(x^2) - E(x)^2 is proved to be numerical
     *  unstable.
     *  Ref to e.g. "(technical report) T.F.Chan et al.
     *   Algorithms for computing the sample variance:
     *   analysis and recommendations".
     *  So we use double-pass version here.
     **/
 
    rd_init(lmem_scheme->addrs, fw_param);
 
    if (mode != RMS_NORM_MODE) {
      // calc sum_i x_i
      loop_3stage_func(rd_load, rd_compute_mean_halfway, swpl_do_nothing,
                       global_addrs, lmem_scheme, &slice_info, fw_param, NULL);
 
      // calc mean
      rd_mean_end(lmem_scheme->addrs, &slice_info, fw_param);
    }
 
    // calc sum_i (x_i - mean)^2
    // for RMSNorm (x_i)^2
    loop_3stage_func(rd_load, rd_compute_rstd_halfway, swpl_do_nothing,
                     global_addrs, lmem_scheme, &slice_info, fw_param, NULL);
 
    // calc rstd
    rd_rstd_end(lmem_scheme->addrs, &slice_info, fw_param);
 
    loop_3stage_func(gn_load__c_split, gn_compute__c_split, gn_store__c_split,
                     global_addrs, lmem_scheme, &slice_info, fw_param, NULL);
 
    if (mode == LAYER_NORM_MODE) {
      gn_store_optional__ch_split(global_addrs, lmem_scheme->addrs, fw_param);
    }
  }
 
  if (mode == GROUP_NORM_MODE) {
    const glb_scheme_t *g = (glb_scheme_t *)global_addrs;
    const ch_split_fw_param_t *p = (ch_split_fw_param_t *)fw_param;
    if (p->ex_affine & GN_HAVE_WEIGHT && p->ex_affine & GN_HAVE_BIAS) {
#ifdef DEBUG_GN
      TPUKERNEL_DBG("++++++++++++++++++++++++I'm "
                    "here++++++++++++++++++++++++++++ch split calling scale\n");
#endif
      nodechip_scale_forward(g->output_addr, g->weight_addr, g->bias_addr,
                             g->output_addr, p->rn, p->rc, p->rh, 1, 1, 1, true,
                             false, 0, false, p->dtype);
    } else {
      int io_shape[3] = {p->rn, p->rc, p->rh};
      int wb_shape[3] = {1, p->rc, 1};
      if (p->ex_affine & GN_HAVE_WEIGHT) {
        nodechip_bcbinary_fp(g->output_addr, g->weight_addr, g->output_addr,
                             io_shape, wb_shape, 3, 3, BINARY_MUL, p->dtype,
                             false, 0);
      }
      if (p->ex_affine & GN_HAVE_BIAS) {
        nodechip_bcbinary_fp(g->output_addr, g->bias_addr, g->output_addr,
                             io_shape, wb_shape, 3, 3, BINARY_ADD, p->dtype,
                             false, 0);
      }
    }
  }
}
 
static
void _group_norm(global_addr_t input_global_addr,
                 global_addr_t weight_global_addr,
                 global_addr_t bias_global_addr,
                 global_addr_t output_global_addr,
                 global_addr_t mean_global_addr, global_addr_t rstd_global_addr,
                 const int *shape, int dims, int axis, int group_num, float eps,
                 int affine, bool need_mean, bool need_rstd, data_type_t dtype,
                 gn_mode_t mode) {
  TPUKERNEL_ASSERT(0 < dims && axis < dims);
  TPUKERNEL_ASSERT(group_num > 0);
  TPUKERNEL_ASSERT(shape[axis] % group_num == 0);
  TPUKERNEL_ASSERT(eps > 0);
  TPUKERNEL_ASSERT(0 <= affine && affine < 4);
  TPUKERNEL_ASSERT(tpu_is_data_type_fp(dtype));
 
  const int chn_per_grp = shape[axis] / group_num;
 
  uint64_t outer_dim_ = 1;
  for (int i = 0; i < axis; ++i) {
    outer_dim_ *= shape[i];
    TPUKERNEL_ASSERT(outer_dim_ <= INT32_MAX);
  }
  outer_dim_ *= group_num;
  TPUKERNEL_ASSERT(outer_dim_ <= INT32_MAX);
  uint64_t inner_dim_ = 1;
  for (int i = axis + 1; i < dims; ++i) {
    inner_dim_ *= shape[i];
    TPUKERNEL_ASSERT(inner_dim_ <= INT32_MAX);
  }
  inner_dim_ *= chn_per_grp;
  TPUKERNEL_ASSERT(inner_dim_ <= INT32_MAX);
 
  const int channel = (int)outer_dim_;
  const int height = (int)inner_dim_;
 
  const dim4 glb_ishape = {1, channel, height, 1};
  dim4 glb_istr;
  tpu_continuous_stride(&glb_istr, &glb_ishape);
 
  int ex_affine = affine;
  if (need_mean)
    ex_affine |= GN_NEED_MEAN;
  if (need_rstd)
    ex_affine |= GN_NEED_RSTD;
 
  int ex_affine_ = ex_affine;
  if (mode == GROUP_NORM_MODE)
    ex_affine_ &= ~(GN_HAVE_WEIGHT | GN_HAVE_BIAS);
 
  const glb_scheme_t glb_scheme = {
      .input_addr = input_global_addr,
      .weight_addr = weight_global_addr,
      .bias_addr = bias_global_addr,
      .output_addr = output_global_addr,
      .mean_addr = mean_global_addr,
      .rstd_addr = rstd_global_addr,
  };
 
  loc_scheme_t loc_scheme;
  loc_mem_sz_t loc_mem_sz;
  swpl_lmem_scheme_t lmem_scheme;
  lmem_scheme.addrs = (local_addr_t *)&loc_scheme;
  lmem_scheme.mem_sizes = (int *)&loc_mem_sz;
  lmem_scheme.num = sizeof(loc_scheme_t) / sizeof(local_addr_t);
 
  // --- try c split ---
 
  lmem_alloc_c_split(channel, height, ex_affine, dtype, &loc_scheme,
                     &loc_mem_sz,
#ifdef USE_PIPELINE
                     true
#else
                     false
#endif
  );
 
  int cslice = -1;
  int ret = calc_slice_c_split(channel, height, dtype, ex_affine, &loc_mem_sz,
                               &cslice);
 
  if (ret == 0) {
    TPUKERNEL_DBG("[cslice: %d]\n", cslice);
 
    c_split_fw_param_t fw_param = {
        .mode = mode,
        .height = height,
        .ex_affine = ex_affine,
        .eps = eps,
        .dtype = dtype,
        .p_glb_istr = &glb_istr,
    };
    if (mode == GROUP_NORM_MODE) {
      uint64_t outer_dim2_ = 1;
      for (int i = 0; i < axis; ++i) {
        outer_dim2_ *= shape[i];
        TPUKERNEL_ASSERT(outer_dim_ <= INT32_MAX);
      }
      uint64_t inner_dim2_ = 1;
      for (int i = axis + 1; i < dims; ++i) {
        inner_dim2_ *= shape[i];
        TPUKERNEL_ASSERT(inner_dim2_ <= INT32_MAX);
      }
      fw_param.rc = shape[axis];
      fw_param.rn = (int)outer_dim2_;
      fw_param.rh = (int)inner_dim2_;
    }
    group_norm__c_split((global_addr_t *)&glb_scheme, &lmem_scheme, &fw_param,
                        channel, cslice, mode);
  } else { // --- try c-h split ---
 
    lmem_alloc__ch_split(channel, height, ex_affine_, dtype, &loc_scheme,
                         &loc_mem_sz,
#ifdef USE_PIPELINE
                         true
#else
                         false
#endif
    );
 
    int cslice = -1, hslice = -1;
    int ret = calc_slice__ch_split(channel, height, dtype, ex_affine_,
                                   &loc_mem_sz, &cslice, &hslice);
    TPUKERNEL_ASSERT_INFO(ret == 0, "tensor split failed!");
    // TPUKERNEL_DBG("[cslice: %d, hslice: %d]\n", cslice, hslice);
 
    ch_split_fw_param_t fw_param = {
        .mode = mode,
        .ex_affine = ex_affine,
        .eps = eps,
        .dtype = dtype,
        .p_glb_istr = &glb_istr,
        .mode = mode,
    };
    if (mode == GROUP_NORM_MODE) {
      uint64_t outer_dim2_ = 1;
      for (int i = 0; i < axis; ++i) {
        outer_dim2_ *= shape[i];
        TPUKERNEL_ASSERT(outer_dim_ <= INT32_MAX);
      }
      uint64_t inner_dim2_ = 1;
      for (int i = axis + 1; i < dims; ++i) {
        inner_dim2_ *= shape[i];
        TPUKERNEL_ASSERT(inner_dim2_ <= INT32_MAX);
      }
      fw_param.rc = shape[axis];
      fw_param.rn = (int)outer_dim2_;
      fw_param.rh = (int)inner_dim2_;
    }
    group_norm__ch_split((global_addr_t *)&glb_scheme, &lmem_scheme, &fw_param,
                         channel, cslice, height, hslice, mode);
  }
}
 
void nodechip_native_group_norm(
    global_addr_t input_global_addr, global_addr_t weight_global_addr,
    global_addr_t bias_global_addr, global_addr_t output_global_addr,
    global_addr_t mean_global_addr, global_addr_t rstd_global_addr,
    const int *shape, int dims, int axis, int group_num, float eps, int affine,
    data_type_t dtype) {
  _group_norm(input_global_addr, weight_global_addr, bias_global_addr,
              output_global_addr, mean_global_addr, rstd_global_addr, shape,
              dims, axis, group_num, eps, affine, true, true, dtype,
              GROUP_NORM_MODE);
}
#endif
 
int tpu_kernel_api_native_group_norm_multi_core(const void *args) {
#ifdef BACKEND_SG2260
  TPUKERNEL_ASSERT_INFO(false, "not implementated");
  return 0;
#else
  sg_api_native_group_norm_t *api = (sg_api_native_group_norm_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);
 
  // tpu_initialize();
  // nodechip_native_group_norm(api->input_global_addr, api->weight_global_addr,
  //                            api->bias_global_addr, api->output_global_addr,
  //                            api->mean_global_addr, api->rstd_global_addr,
  //                            api->shape, api->dim, api->axis, api->group_num,
  //                            api->eps, api->affine, (data_type_t)api->dtype);
  // tpu_poll();
  return 0;
#endif
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_native_group_norm_multi_core);