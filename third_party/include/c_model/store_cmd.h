#ifndef STORE_CMD_H_
#define STORE_CMD_H_

#include <stdio.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void allow_store_cmd();
void forbid_store_cmd();
int get_store_cmd_enable();

void use_atomic_cmodel();
void forbid_atomic_cmodel();
int get_atomic_cmodel_enable();

void set_total_id_ptr(u32* m_gdma_total_id_ptr, u32* m_bdc_total_id_ptr,
                      void* cmdid_node, void* m_gdma_group_id_ptr,
                      void* m_bdc_group_id_ptr, int* m_cmdid_groupnum);

void set_cmd_file_ptr(FILE * m_gdma_cmd_file_ptr, FILE * m_bdc_cmd_file_ptr);
void set_cmd_buffer_ptr(void * m_gdma_buffer_ptr, void * m_bdc_buffer_ptr);

void cmdid_overflow_reset(CMD_ID_NODE* p_cmdid_node);
void store_cmd(void * cmd, int engine_id);
void store_cmd_inst(void * cmd, int engine_id);
void store_cmd_end();
void set_cmd_len_ptr(void* gdma_cmd_len_ptr, void* bdc_cmd_len_ptr);

// api about profiling
int get_enable_profile();
FILE* get_profile_file();
void enable_profile(bool enable, FILE* fp);

#ifdef __cplusplus
}
#endif

#endif

