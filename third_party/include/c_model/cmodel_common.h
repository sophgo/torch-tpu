#ifndef CMODEL_COMMON_H_
#define CMODEL_COMMON_H_

#include "cmodel_memory.h"

#ifdef __cplusplus
  extern "C" {
#endif

typedef struct {
  u32 n;
  u32 c;
  u32 h;
  u32 w;
} shape_t;

typedef shape_t stride_t;

typedef enum {
  CM_SINT8   = 0,
  CM_UINT8  = 1,
  CM_SINT16  = 2,
  CM_UINT16  = 3,
  CM_SINT32  = 4,
  CM_UINT32 = 5,
  CM_FP16   = 6,
  CM_BFP16  = 7,
  CM_FP32   = 8,
  CM_DTYPE_END,
} CM_DTYPE;

unsigned get_chip_id();
void init_cmd_id_node(CMD_ID_NODE *p_id_node);
void *create_cmd_id_node();
void destroy_cmd_id_node(void *pid_node);
void reset_cmd_id(void *pid_node);
void set_cmd_id_prefix(void* pid_node, const char * name_prefix);
void set_cmd_id_cycle(void *pid_node, long long val);
long long get_cmd_id_cycle(void *pid_node);

CM_DTYPE get_dtype(PREC precision, bool is_sign);

typedef struct tensor_info {
  u32 n, c, h, w;
  union {
    u32 str_data[4];
    struct {
      u32 n_stride;
      u32 c_stride;
      u32 h_stride;
      u32 w_stride;
    };
  };
  u32 address;
  PREC precision;
  u32 neuron_matrix;      // 0: neuron, 1: matrix
  u32 matrix_col_margin;  // the margin is not 0, when column_num%w_param!=0
} TENSOR_INFO;

typedef enum {
    ALIGN_STORE       = 0,
    COMPACT_STORE     = 1,
    BIAS_STORE        = 2,
    STRIDE_STORE      = 3
} STORE_MODE;

/* short_str:
 * 0: align
 * 1: compact
 * 2: n/h/w stride = 0, c_stride = 1
 * 3: use given stride
 */
void tensor_info_generate(
    u32 N, u32 C, u32 H, u32 W, u32 laddr, int short_str, shape_t* _stride,
    PREC precision, TENSOR_INFO* tensor_info);

void matrix_tensor_info_generate(
    u32 row_num, u32 col_num, u32 w_param, u32 laddr,
    PREC precision, TENSOR_INFO* info);

void gather_data_from_lmem(LOCAL_MEM* lmem, TENSOR_INFO* info, void* dst_ptr, bool transpose);
void get_local_mem_tensor_data(
    void* data_ptr, int node_idx, u32 N, u32 C, u32 H, u32 W, u32 laddr, int short_str,
    shape_t* stride, PREC precision, bool transpose);

void scatter_data_to_lmem(
    LOCAL_MEM* lmem, TENSOR_INFO* info, void* src_ptr,
    u64 lane_mask, bool enable_lane_mask, bool transpose);
void set_local_mem_tensor_data(
    void* data_ptr, int node_idx, u32 N, u32 C, u32 H, u32 W, u32 laddr, int short_str,
    shape_t* stride, PREC precision, u64 lane_mask, bool enable_lane_mask, bool transpose);

int get_cmd_bytesize(P_COMMAND cmd, ENGINE_ID eng_id);
int get_long_cmd_bytesize(ENGINE_ID eng_id);
u32 get_cur_atomic_id(P_COMMAND cmd, ENGINE_ID eng_id);

P_COMMAND cmd_mapping(int node_idx, P_COMMAND cmd, ENGINE_ID engine_id);
void execute_command(int node_idx, P_COMMAND cmd, ENGINE_ID engine_id);
void cmodel_call_atomic(int node_idx, P_COMMAND cmd, ENGINE_ID eng_id);

u32 get_local_addr(
  u32 addr, int C, int H, int W,
  int nidx, int cidx, int hidx, int widx,
  PREC prec, int short_str, const stride_t* stride);

/* Info about data compare */
int array_cmp_fix8b(void *p_exp, void* p_got, int sign, int len, const char* info_label, int delta);
int array_cmp_fix16b(void* p_exp, void* p_got, int sign, int len, const char* info_label, int delta);
int array_cmp_int32(int* p_exp, int* p_got, int len, const char* info_label, int delta);
int array_cmp(float* p_exp, float* p_got, int len, const char* info_label, float delta);

void save_data_to_file(const char* filename, const void* ptr, size_t size, PREC prec, bool sign);

u64 get_reg_id_val(P_COMMAND cmd, const reg_id_t id);
void set_reg_id_val(P_COMMAND cmd, const reg_id_t id, u64 val);
#define GET_REG_ID_VAL(cmd, id) get_reg_id_val(cmd, (reg_id_t)id)
#define SET_REG_ID_VAL(cmd, id, val) set_reg_id_val(cmd, (reg_id_t)id, val)

/* for sys command */
void set_lterm_valid(bool valid);
bool get_lterm_valid();
void set_sterm_valid(bool valid);
bool get_sterm_valid();
void set_lane_mask(u64 lane_mask);
u64 get_lane_mask(P_COMMAND cmd);
void reset_sys_reg();

/* for TV GEN */
int rand_alloc_bank(int *used_banks, int *alloc_map);

void host_dma_copy_cmodel(void *dst, void *src, u64 size,  HOST_CDMA_DIR dir, int node_idx, int mem_type);

#ifdef __cplusplus
  }
#endif

#endif
