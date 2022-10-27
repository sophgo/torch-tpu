#ifndef  BM_STAS_VEC_H_
#define  BM_STAS_VEC_H_

#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include "common.h"

const int STAS_ENGINE_NUM = 2;
class sta_node {
public:
  long long m_cycle_end;
  CMD_ID_NODE id_node;
  sta_node(CMD_ID_NODE *pid_node);
  unsigned int operator[](int eng_id);
};

class sta_vec {
  public:
  ENGINE_ID    m_engine_id;
  std::vector < sta_node> m_node_vec;
  void add_sta_node(CMD_ID_NODE *pid_node);

  void show(FILE * output_file);
  void reset();
  void set_engine_id(ENGINE_ID    i_engine_id);
  void get_value(long long * get_data_array, int node_idx);
};

struct summary_info_t{
  std::vector<long long> gdma_summary_info;
  std::string gdma_summary_info_string;
#ifdef ENABLE_ENERGY
  double ddr_access_energy; // nJ
  double sram_rw_energy; // nJ
  double eu_dyn_energy; // nJ
  std::map<std::string, double> op_dyn_energy;
  std::map<std::string, double> op_sram_rw_energy;
#endif
};

class sta_matrix {
  public:
  sta_matrix();
  sta_vec m_matrix[STAS_ENGINE_NUM];
  void add_sta_node(CMD_ID_NODE *pid_node, ENGINE_ID i_engine_id);
  summary_info_t show(FILE * output_file);
  summary_info_t show_row(FILE * output_file);
  void show_column(FILE * output_file);
  summary_info_t show_column_detail(FILE * output_file);
  void visual_timeline(FILE * output_file, long long total_cycle);
  void hw_runtime_output(FILE * output_file);

  void reset();
};

#endif






