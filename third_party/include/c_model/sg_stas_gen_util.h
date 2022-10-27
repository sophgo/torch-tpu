#ifndef SG_STAS_GEN_UTIL_H
#define SG_STAS_GEN_UTIL_H


#ifdef __cplusplus
extern "C" {
#endif

//void sg_stas_add_node(CMD_ID_NODE *pid_node, ENGINE_ID i_engine_id);
void sg_stas_dump(void* pid_node);
void sg_stas_show(int bd_cmd_id, int gdma_cmd_id, long long cycle_count);
void sg_flops_dump(long long flops, void* pid_node);
void sg_flops_show(long long flops, long long cycle_count);
void sg_mem_show(long long total_mem, long long allocated_mem, long long coef_and_data);
void sg_stas_reset();
int  sg_stas_gen_init(int node_idx);
void sg_stas_gen_deinit(int node_idx);
void sg_stas_info_insert(int api_id);
void sg_set_profile_dump(bool enable);
void sg_set_profile_path(const char* path);
void allow_atomic_cmodel_assert();
void forbid_atomic_cmodel_assert();
int get_atomic_cmodel_assert_enable();

#ifdef __cplusplus
}
#endif


#endif /* SG_STAS_GEN_UTIL_H */

