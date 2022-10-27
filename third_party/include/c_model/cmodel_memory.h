#ifndef CMODEL_MEMORY_H_
#define CMODEL_MEMORY_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NODECHIP_NUM 1

#define CMODEL_LOG_QUIET    -8
#define CMODEL_LOG_PANIC     0
#define CMODEL_LOG_FATAL     8
#define CMODEL_LOG_ERROR    16
#define CMODEL_LOG_WARNING  24
#define CMODEL_LOG_INFO     32
#define CMODEL_LOG_VERBOSE  40
#define CMODEL_LOG_DEBUG    48
#define CMODEL_LOG_TRACE    56

#ifdef SG_TV_GEN
extern int g_nLOGOFF;
#endif

typedef enum {
  CMODEL_SUCCESS = 0,
  CMODEL_FAIL = -1,
} CMODEL_STATUS;

typedef enum {
  MEM_TYPE_GLOBAL  = 0,     // global memory
  MEM_TYPE_STATIC  = 1,     // static memory for static data table
  MEM_TYPE_L2      = 2,     // HAU L2 memory
  MEM_TYPE_LOCAL   = 3,     // TPU local memory
  MEM_TYPE_UNKNOWN = 4,
} MEM_TYPE_T;

typedef struct local_mem {
  char *raw_ptr;
  unsigned int **mem_arr;
  int count;    // count of local mem
  int size_per_mem;  // size of each local mem
  int align_num;
  int need_free;
} LOCAL_MEM;

typedef struct {
  char* raw_ptr;
  u64 start_addr;
  u64 size;
  int type;
  LOCAL_MEM *p_local;
} CONTINUOUS_MEM;

void make_local_mem_from_continuous(LOCAL_MEM * mem, int count, int size_per_mem,
        int align_num, CONTINUOUS_MEM *con_mem);
void alloc_local_mem(LOCAL_MEM * mem, int count, int size_per_mem, int align_num);
void free_local_mem(LOCAL_MEM * mem);
void clear_local_mem(int node_idx);
void fill_local_mem(int node_idx);

LOCAL_MEM * get_local_mem_from_continuous(CONTINUOUS_MEM * mem);
void alloc_continuous_mem(CONTINUOUS_MEM *mem, u64 start, u64 size, int type);
void free_continuous_mem(CONTINUOUS_MEM *mem);
CONTINUOUS_MEM *get_continuous_mem(int nodechip_idx, u64 addr);
CONTINUOUS_MEM *get_continuous_mem_by_type(int nodechip_idx, int type);
CONTINUOUS_MEM *get_continuous_mem_by_ptr(int nodechip_idx, void* ptr);
void* get_continuous_mem_ptr(int nodechip_idx, u64 addr);

int  cmodel_init_without_smem_fill(int node_idx, unsigned long long global_mem_size);
int  cmodel_init(int node_idx, unsigned long long global_mem_size);
void cmodel_deinit(int node_idx);

LOCAL_MEM *get_local_mem(int node_idx);
LOCAL_MEM *get_cur_local_mem(void);
void *get_local_memaddr(LOCAL_MEM * p_local_mem, int np_idx);
void host_dma_copy_s2d_cmodel(u64 dst, const void *src, u64 size, int idx);
void host_dma_copy_d2s_cmodel(void *dst, u64 src, u64 size, int idx);
void *get_local_memaddr_by_node(int node_idx, int np_idx);
void *get_static_memaddr_by_node(int node_idx);
void *get_l2_sram(int node_idx);
void *get_global_memaddr(int node_idx);
void *get_share_memaddr(int node_idx);
u32  *get_arrange_reg(int node_idx);
u64 cmodel_get_global_mem_size(int node_idx);

int get_cur_nodechip_idx(void);
void set_cur_nodechip_idx(int node_idx);
int get_nodechip_status(int node_idx);

#ifdef __cplusplus
}
#endif

#endif
