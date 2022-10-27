#ifndef CMODEL_MULTITHREAD_H_
#define CMODEL_MULTITHREAD_H_
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

  void cmodel_multi_thread_cxt_init(int node_idx);
  void cmodel_multi_thread_cxt_deinit(int node_idx);
  void send_command_to_fifo(int nodechip_idx, P_COMMAND cmd,
          ENGINE_ID eng_id, bool * set_cmd);
  u32 get_sync_id(int nodechip_idx, ENGINE_ID eng_id);
  void cmodel_multi_engine(u64 * engine_address, bool * using_cmd, int nodechip_idx);
  bool check_all_engine_done(int nodechip_idx, CMD_ID_NODE* id_node);
  void reset_all_engine_cmd_id(int nodechip_idx);

#ifdef __cplusplus
}
#endif

#endif
