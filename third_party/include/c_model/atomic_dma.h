#ifndef ATOMIC_DMA_H
#define ATOMIC_DMA_H

#include "cmodel_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GDMA_FILTER_RES_NUM_REG ((160/32) * 4)

void global_dma(
    void        *p_system_mem,
    LOCAL_MEM   *p_local_mem,
    P_COMMAND   p_command);

void atomic_global_dma(
    int nodechip_idx,
    P_COMMAND p_command);

typedef struct {
  u64 start_offset;
  int format;
  size_type nsize;
  size_type csize;
  size_type hsize;
  size_type wsize;
  stride_type nstride;
  stride_type cstride;
  stride_type hstride;
  stride_type wstride;
  // cached value, need to be calculated before use
  int is_localmem;
  int type_len; // get_type_len(format)
  // for localmem: convert_localmem_addr(start_offset, &mem_idx, &mem_base);
  // for globalmem: mem_base = start_offset
  int mem_idx;
  u64 mem_base;
} neuron_info_t;

#define CMODEL_FUNC_TEST
#ifdef CMODEL_FUNC_TEST

void convert_localmem_start_offset(u64 addr, int mem_num, int mem_size, int *mem_idx, u64 *mem_base);

u32 get_gdma_filter_res_num();

#endif

#ifdef __cplusplus
}
#endif

#endif /* ATOMIC_DMA_H */








