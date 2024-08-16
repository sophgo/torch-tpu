#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <assert.h>
#include <vector>
#include <utility>
#include <stdlib.h>
#include <string.h>
#ifdef BACKEND_SG2260
#include "tpuv7_rt.h"
#include "sgdnn_runtime.h"

#define PROFILE_INFO(file, fmt, ...) fprintf(file, fmt, __VA_ARGS__)

#pragma pack(1)
typedef struct {
    unsigned int inst_start_time;
    unsigned int inst_end_time;
    unsigned int inst_id;
    // lower 1 bit: thread_id; higher 31 bits: bank_conflict
    unsigned int thread_id_and_bank_conflict;
} tiu_pmu_item_t;
#define REAL_TIU_ITEM_SIZE 32
typedef struct {
    tiu_pmu_item_t valid_data;
    //unsigned char reserved[REAL_TIU_ITEM_SIZE-sizeof(tiu_pmu_item_t)];
} tiu_pmu_item_ext_t;

typedef struct {
    // H0
    unsigned int inst_start_time;
    unsigned int inst_end_time;
    unsigned int inst_id;
    uint32_t thread_id: 1;
    uint32_t ar_latency_cnt: 19;
    uint32_t rip_valid_latency: 12;
    // H1
    unsigned int gif_wr_rd_stall_cntr;
    unsigned int axi_d0_w_cntr;
    unsigned int axi_d0_ar_cntr;
    unsigned int axi_d0_aw_cntr;
    // H2
    unsigned int axi_d0_wr_stall_cntr;
    unsigned int axi_d0_rd_stall_cntr;
    unsigned int gif_mem_w_cntr;
    unsigned int gif_mem_ar_cntr;
    // H3
    unsigned int axi_d0_wr_vaild_cntr;
    unsigned int axi_d0_rd_vaild_cntr;
    unsigned int gif_wr_valid_cntr;
    unsigned int gif_rd_valid_cntr;
} gdma_pmu_item_t;

// #define REAL_GDMA_ITEM_SIZE 256
typedef struct {
    gdma_pmu_item_t valid_data;
    // unsigned char reserved[REAL_GDMA_ITEM_SIZE-sizeof(gdma_pmu_item_t)];
} gdma_pmu_item_ext_t;

typedef struct {
    // H0
    unsigned int inst_start_time;
    unsigned int inst_end_time;
    unsigned int inst_id;
    uint32_t thread_id: 1;
    uint32_t ar_latency_cnt: 19;
    uint32_t rip_valid_latency: 12;
    // H1
    unsigned int gif_wr_rd_stall_cntr;
    unsigned int axi_d0_w_cntr;
    unsigned int axi_d0_ar_cntr;
    unsigned int axi_d0_aw_cntr;
    // H2
    unsigned int axi_d0_wr_stall_cntr;
    unsigned int axi_d0_rd_stall_cntr;
    unsigned int gif_mem_w_cntr;
    unsigned int gif_mem_ar_cntr;
    // H3
    unsigned int axi_d0_wr_vaild_cntr;
    unsigned int axi_d0_rd_vaild_cntr;
    unsigned int gif_wr_valid_cntr;
    unsigned int gif_rd_valid_cntr;
} sdma_pmu_item_t;

typedef struct {
    sdma_pmu_item_t valid_data;
    // unsigned char reserved[REAL_GDMA_ITEM_SIZE-sizeof(sdma_pmu_item_t)];
} sdma_pmu_item_ext_t;
#pragma pack()

static int all_zero(const unsigned char* data, int len){
    for(int i=0; i<len; i++){
        if(data[i]!=0) return 0;
    }
    return 1;
}

template<typename T>
static std::vector<decltype(T::valid_data)> parse_pmu_data(const uint32_t* raw_data, uint64_t len, FILE *file) {
  using RealType = decltype(T::valid_data);
  auto tpu_data = (T*)raw_data;
  std::vector<RealType> result;
  size_t max_len = len/sizeof(T);
  size_t valid_len = 0;
  while (valid_len < max_len) {
     if(all_zero((const unsigned char*)&tpu_data[valid_len], sizeof(T))) break;
     result.push_back(tpu_data[valid_len].valid_data);
     valid_len++;
  }
  PROFILE_INFO(file, "tiu record_num=%d, max_record_num=%d\n", (int)valid_len, (int)max_len);
  return result;
}

typedef struct {
    uint32_t type;
    uint32_t begin_usec;
    uint32_t end_usec;
    uint32_t id;
    uint32_t height;
    std::string info;
} show_param_t;

static void show_gdma_pmu_item(int core_id, const std::vector<gdma_pmu_item_t>& items, FILE *file){
  float freq_MHz = 1000;
  float time_offset = 0;
  float period = 1/freq_MHz;
  PROFILE_INFO(file, "Note: gdma record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
  if(items.empty()) return;
  int real_print = items.size();
  for(int i=0; i<real_print; i++){
    auto& p = items[i];
    auto total_time =  (p.inst_end_time - p.inst_start_time) * period;
    auto total_bytes = p.axi_d0_wr_vaild_cntr + p.gif_wr_valid_cntr;
    auto speed = total_time>0?total_bytes/total_time: -1000.0;
    PROFILE_INFO(file,
      "[%d]---> gdma record #%d, inst_id=%d, thread_id=%d, ar_latency_cnt=%d, "
      "rip_valid_latency=%d, cycle=%d\n"
      "start=%.3fus, end=%.3fus, interval=%.3fus, gif_wr_rd_stall_cntr= %d\n"
      "axi_d0_w_cntr=%d, axi_d0_ar_cntr=%d, axi_d0_aw_cntr=%d, "
      "axi_d0_wr_stall_cntr=%d, axi_d0_rd_stall_cntr=%d, gif_mem_w_cntr=%d\n"
      "gif_mem_ar_cntr=%d, axi_d0_wr_vaild_cntr=%d, "
      "axi_d0_rd_vaild_cntr=%d, gif_wr_valid_cntr=%d, gif_rd_valid_cntr=%d\n",
      core_id, i, p.inst_id, p.thread_id, p.ar_latency_cnt, p.rip_valid_latency,
      p.inst_end_time - p.inst_start_time,
      p.inst_start_time * period, p.inst_end_time * period, total_time,
      p.gif_wr_rd_stall_cntr, p.axi_d0_w_cntr, p.axi_d0_ar_cntr,
      p.axi_d0_aw_cntr, p.axi_d0_wr_stall_cntr, p.axi_d0_rd_stall_cntr,
      p.gif_mem_w_cntr, p.gif_mem_ar_cntr, p.axi_d0_wr_vaild_cntr,
      p.axi_d0_rd_vaild_cntr, p.gif_wr_valid_cntr, p.gif_rd_valid_cntr);
    std::string info;
    info+="speed="+std::to_string(speed/1000.0)+"GB/s<br>";
#define ADD_INFO(d) info += std::string(#d"=")+std::to_string(p.d)+"<br>"
    ADD_INFO(gif_wr_rd_stall_cntr);
    ADD_INFO(axi_d0_w_cntr);
    ADD_INFO(axi_d0_ar_cntr);
    ADD_INFO(axi_d0_aw_cntr);
    ADD_INFO(axi_d0_wr_stall_cntr);
    ADD_INFO(axi_d0_rd_stall_cntr);
    ADD_INFO(gif_mem_w_cntr);
    ADD_INFO(gif_mem_ar_cntr);
    ADD_INFO(axi_d0_wr_vaild_cntr);
    ADD_INFO(axi_d0_rd_vaild_cntr);
    ADD_INFO(gif_wr_valid_cntr);
    ADD_INFO(gif_rd_valid_cntr);
#undef ADD_INFO
  }
}

static void show_sdma_pmu_item(int core_id, const std::vector<sdma_pmu_item_t>& items, FILE *file){
  float freq_MHz = 1000;
  float time_offset = 0;
  float period = 1/freq_MHz;
  PROFILE_INFO(file, "Note: sdma record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
  if(items.empty()) return;
  int real_print = items.size();
  for(int i=0; i<real_print; i++){
    auto& p = items[i];
    auto total_time =  (p.inst_end_time - p.inst_start_time) * period;
    auto total_bytes = p.axi_d0_wr_vaild_cntr + p.gif_wr_valid_cntr;
    auto speed = total_time>0?total_bytes/total_time: -1000.0;
    PROFILE_INFO(file,
      "[%d]---> sdma record #%d, inst_id=%d, thread_id=%d, ar_latency_cnt=%d, "
      "rip_valid_latency=%d, cycle=%d\n"
      "start=%.3fus, end=%.3fus, interval=%.3fus, gif_wr_rd_stall_cntr= %d\n"
      "axi_d0_w_cntr=%d, axi_d0_ar_cntr=%d, axi_d0_aw_cntr=%d, "
      "axi_d0_wr_stall_cntr=%d, axi_d0_rd_stall_cntr=%d, gif_mem_w_cntr=%d\n"
      "gif_mem_ar_cntr=%d, axi_d0_wr_vaild_cntr=%d, "
      "axi_d0_rd_vaild_cntr=%d, gif_wr_valid_cntr=%d, gif_rd_valid_cntr=%d\n",
      core_id, i, p.inst_id, p.thread_id, p.ar_latency_cnt, p.rip_valid_latency,
      p.inst_end_time - p.inst_start_time,
      p.inst_start_time * period, p.inst_end_time * period, total_time,
      p.gif_wr_rd_stall_cntr, p.axi_d0_w_cntr, p.axi_d0_ar_cntr,
      p.axi_d0_aw_cntr, p.axi_d0_wr_stall_cntr, p.axi_d0_rd_stall_cntr,
      p.gif_mem_w_cntr, p.gif_mem_ar_cntr, p.axi_d0_wr_vaild_cntr,
      p.axi_d0_rd_vaild_cntr, p.gif_wr_valid_cntr, p.gif_rd_valid_cntr);
    std::string info;
    info+="speed="+std::to_string(speed/1000.0)+"GB/s<br>";
#define ADD_INFO(d) info += std::string(#d"=")+std::to_string(p.d)+"<br>"
    ADD_INFO(gif_wr_rd_stall_cntr);
    ADD_INFO(axi_d0_w_cntr);
    ADD_INFO(axi_d0_ar_cntr);
    ADD_INFO(axi_d0_aw_cntr);
    ADD_INFO(axi_d0_wr_stall_cntr);
    ADD_INFO(axi_d0_rd_stall_cntr);
    ADD_INFO(gif_mem_w_cntr);
    ADD_INFO(gif_mem_ar_cntr);
    ADD_INFO(axi_d0_wr_vaild_cntr);
    ADD_INFO(axi_d0_rd_vaild_cntr);
    ADD_INFO(gif_wr_valid_cntr);
    ADD_INFO(gif_rd_valid_cntr);
#undef ADD_INFO
  }
}

static void show_tiu_pmu_item(int core_id, const std::vector<tiu_pmu_item_t>& items, FILE *file){
  float freq_MHz = 1000;
  float time_offset = 0;
  int real_print = items.size();
  float period = 1/freq_MHz;
  PROFILE_INFO(file, "Note: tiu record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
  if(items.empty()) return;
  for(int i=0; i<real_print; i++){
    auto& p = items[i];
    PROFILE_INFO(file, "[%d]---> tiu record #%d, inst_id=%d, thread_id=%d, cycle=%d, "
               "start=%.3fus, end=%.3fus, interval=%.3fus, bank_conflict=%d\n",
               core_id, i, (int)(p.inst_id),
               (int)(p.thread_id_and_bank_conflict & 0x1),
               (int)(p.inst_end_time - p.inst_start_time),
               p.inst_start_time * period, p.inst_end_time * period,
               (p.inst_end_time - p.inst_start_time) * period,
               (int)(p.thread_id_and_bank_conflict >> 1));
  }
}

static void getPMU(tpu_resource_t stream) {
    tpuRtStreamSynchronize(stream);
    static FILE *__profile_fp = fopen("profile_data.log", "w");;
    enum {
        GDMA = 0,
        TIU = 1,
        SDMA = 2,
        PMUEngineNum,
    };
    static const uint64_t PMUStartAddr = 1ULL * 1024 * 1024;
    static const uint64_t PMUCoreOffset = 81ULL * 1024 * 1024;
    uint64_t PMUSize[PMUEngineNum] = {20ULL * 1024 * 1024, 40ULL * 1024 * 1024, 20ULL * 1024 * 1024};
    uint64_t EngineStartAddr[PMUEngineNum] = {PMUStartAddr, (PMUStartAddr + PMUSize[GDMA]), (PMUStartAddr + PMUSize[GDMA] + PMUSize[TIU])};
    const int coreNum = 8;
    std::vector<std::vector<void *>> pmuData(coreNum, std::vector<void *>(PMUEngineNum, nullptr));
    for (int coreIdx = 0; coreIdx < coreNum ; coreIdx ++ ) {
        for (int engineIdx = 0; engineIdx < PMUEngineNum; engineIdx++) {
            void *ret;
            tpuRtMallocHost(&ret, PMUSize[engineIdx]);
            memset(ret, 0, PMUSize[engineIdx]);
            tpuRtMemcpyD2S(ret, (void *)((uint64_t*)(EngineStartAddr[engineIdx] + coreIdx * PMUCoreOffset)), PMUSize[engineIdx]);
            pmuData[coreIdx][engineIdx] = ret;
        }
    }
    for (int coreIdx = 0; coreIdx < coreNum ; coreIdx ++ ) {
        auto tiuData = parse_pmu_data<tiu_pmu_item_ext_t>((uint32_t*)pmuData[coreIdx][TIU], PMUSize[TIU], __profile_fp);
        auto gdmaData = parse_pmu_data<gdma_pmu_item_ext_t>((uint32_t*)pmuData[coreIdx][GDMA], PMUSize[GDMA], __profile_fp);
        auto sdmaData = parse_pmu_data<sdma_pmu_item_ext_t>((uint32_t*)pmuData[coreIdx][SDMA], PMUSize[SDMA], __profile_fp);
        show_tiu_pmu_item(coreIdx, tiuData, __profile_fp);
        show_gdma_pmu_item(coreIdx, gdmaData, __profile_fp);
        show_sdma_pmu_item(coreIdx,sdmaData, __profile_fp);
        tpuRtFreeHost(pmuData[coreIdx][TIU]);
        tpuRtFreeHost(pmuData[coreIdx][GDMA]);
        tpuRtFreeHost(pmuData[coreIdx][SDMA]);
    }
    fclose(__profile_fp);
}
#endif