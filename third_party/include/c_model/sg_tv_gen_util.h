#ifndef SG_TV_GEN_UTIL_H_
#define SG_TV_GEN_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

#define INVALID_ENGINE_IDX 0x7FFFFFFF

typedef enum {
  TV_IDUMP = 0,
  TV_ODUMP = 1,
} TV_DUMP_DIR;

typedef enum {
  NODECHIP_REG = 0,
  HOST_REG = 1,
} REG_TYPE;

int  sg_tv_gen_init(int nodechip_idx);
void sg_tv_gen_deinit(int nodechip_idx);

void sg_wr_tv_dump_reg(unsigned int addr, unsigned int data, REG_TYPE reg_type);
void sg_wr_tv_dump_reg_pointer(volatile unsigned int *addr, unsigned int data, REG_TYPE reg_type);
void sg_read32_wait_ge_tv(int addr, int wait_data, int bits, int shift, REG_TYPE reg_type);
void sg_read32_wait_eq_tv(int addr, int cmp_data, int bits, int shift, REG_TYPE reg_type);

int  get_localmem_cur_engine_dump_idx(ENGINE_ID engine_id, int nodechip_idx, TV_DUMP_DIR dir);

LOCAL_MEM *get_local_mem_mask(int node_idx, ENGINE_ID eng_id);

void sg_tv_io_localmem(LOCAL_MEM * p_local_mem, ENGINE_ID engine_id, TV_DUMP_DIR dir,
    LOCAL_MEM * p_local_mem_mask);
int  sg_tv_io_localmem_idx(LOCAL_MEM * p_local_mem, ENGINE_ID engine_id, TV_DUMP_DIR dir, int nodechip_idx,
    int engine_dump_idx, LOCAL_MEM * p_local_mem_mask);

void dump_cmd(int node_idx, void* cmd, ENGINE_ID eng_id);
int sg_tv_io_mem(int nodechip_idx, int engine_id, int mem_type, TV_DUMP_DIR dump_dir);
void sg_dump_mem2file(void * dump_buf, unsigned long long len, FILE* dump_file);
void sg_tv_dump_host_data(void * p_data, TV_DUMP_DIR dir, int data_size);

void enable_fixed_i_tv_mode();
void disable_fixed_i_tv_mode();


FILE* sg_tv_dump_get_log_file(int nodechip_idx);
FILE* sg_tv_dump_get_hau_i_file(int nodechip_idx);
FILE* sg_tv_dump_get_hau_o_file(int nodechip_idx);

#ifdef TV_GEN_GDMA_LOG
FILE * sg_tv_dump_get_gdma_log_file(int nodechip_idx);
#endif

void sgdnn_log_occupy_space(FILE * log_file, int col_num);

void sgdnn_log_channel_remain(FILE * log_file, int input_column_num,
    int output_column_num, int remaining_num, int window_size);

void set_conv_log(int val);
void sg_wr_tv_dump_sync();

INLINE static void log_str(FILE * outf, const char * str, int using_file) {
    if ( using_file ) {
        fprintf(outf, "%s\n", str);
    } else {
        printf("%s\n", str);
    }
}

INLINE static void log_int_hex(FILE * outf, int val, int using_file) {
    if (using_file) {
        fprintf(outf, "%x\n", val);
    } else {
        printf("%x\n", val);
    }
}

INLINE static void log_int_hex64(FILE * outf, u64 val, int using_file) {
    if (using_file) {
        fprintf(outf, "%llx\n", val);
    } else {
        printf("%llx\n", val);
    }
}

INLINE static void log_int_deci(FILE * outf, int val, int using_file) {
    if (using_file) {
        fprintf(outf, "%d\n", val);
    } else {
        printf("%d\n", val);
    }
}

INLINE static void log_float(FILE * outf, float val, int using_file) {
    if ( using_file ) {
        fprintf(outf, "%f\n", val);
    } else {
        printf("%f\n", val);
    }
}

void enable_reg_tv(int idx);
void disable_reg_tv(int idx);
int  get_reg_tv_enabled(int idx);

typedef enum sg_tv_init_type {
    SG_TV_INIT_INCR = 0,
    SG_TV_INIT_RANDOM = 1,
    SG_TV_INIT_255 = 2,
    SG_TV_INIT_255_F = 3,
    SG_TV_INIT_0_OR_1_F = 4,
    SG_TV_INIT_ALL_0F = 5,
    SG_TV_INIT_ALL_1F = 6,
    SG_TV_INIT_ALL_EDGE_0F = 7,
    SG_TV_INIT_ALL_EDGE_1F = 8,
    SG_TV_INIT_ALL_FRONT_EDGE_0F = 9,
    SG_TV_INIT_ALL_REAR_EDGE_0F = 10,
    SG_TV_INIT_0_OR_1_F16 = 11,
    SG_TV_INIT_0_OR_1_BF16 = 12,
    SG_TV_INIT_0_OR_1_INT8 = 13,
    SG_TV_INIT_0_OR_1_INT16 = 14,
    SG_TV_INIT_0_OR_1_INT32 = 16,
    SG_TV_INIT_INDEX_INT8 = 17,
    SG_TV_INIT_INDEX_INT16 = 18,
    SG_TV_INIT_INDEX_INT32 = 19,
    SG_TV_INIT_FIXED = 20,
} SG_TV_INIT_TYPE;

void sg_tv_init_localmem(LOCAL_MEM * p_local_mem, SG_TV_INIT_TYPE init_type, u32 init_value);
void sg_tv_init_part_localmem(LOCAL_MEM * p_local_mem, SG_TV_INIT_TYPE init_type, int offset, int init_len,
                              int start_npu_idx, int end_npu_idx, int init_npu_num);
void sg_tv_init_mem(void* p_ddr, int ddr_size, SG_TV_INIT_TYPE init_type);
void sg_tv_init_mem_by_mask_index(void * ptr, int size, SG_TV_INIT_TYPE init_type,
                                  u32 max_index);
void sg_tv_init_lmem_by_mask_index(LOCAL_MEM * p_local_mem, int offset, int size,
    SG_TV_INIT_TYPE init_type, u32 max_index);

void sg_tv_dump_blane_mask(int node_idx, u64 lane_mask, TV_DUMP_DIR dump_dir);

#ifdef __cplusplus
}
#endif

#endif /* SG_TV_GEN_UTIL_H */
