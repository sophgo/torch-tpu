#ifndef CMODEL_RUNTIME_H_
#define CMODEL_RUNTIME_H_

#ifdef __cplusplus
extern "C" {
#endif

void cmodel_nodechip_runtime_init(int node_idx);
void cmodel_nodechip_runtime_exit(int node_idx);


u32 * cmodel_get_share_memory_addr(u32 offset, int node_idx);
void cmodel_write_share_reg(u32 idx , u32 val, int node_idx);
u32  cmodel_read_share_reg(u32 idx, int node_idx);



#ifdef __cplusplus
}
#endif

#endif /* CMODEL_RUNTIME_H_ */
