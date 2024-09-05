#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <cstdarg>
#ifdef BACKEND_SG2260
#include "tpuv7_rt.h"
#include "sgdnn_runtime.h"

struct SG2260PMUPolicy
{
  // specify the physical address of each engine
  // we can directly access the physical address, runtime already reserved the address space for us
  static const uint64_t MegaBytes = 1ULL * 1024 * 1024;
  static const uint64_t PMUStartAddr = 1ULL * MegaBytes;
  static const uint64_t PMUCoreOffset = 136ULL * MegaBytes;
  static const uint64_t CDMANum = 11;
  static const int CommonPMUEngineNum = 3;
  enum {
    GDMA = 0,
    TIU = 1,
    SDMA = 2,
    CDMA = 3, // CDMA need handle separately
    EngineNum = 4,
  };
  enum {
    GDMA_PMU_SIZE = 20ULL * MegaBytes,
    TIU_PMU_SIZE = 40ULL * MegaBytes,
    SDMA_PMU_SIZE = 20ULL * MegaBytes,
    CDMA_PMU_SIZE = 5ULL * MegaBytes,
  };
  // static const constexpr variable cannot be used in initializer list
  /*
  static const constexpr uint64_t PMUSize[PMUEngineNum] = {GDMA_PMU_SIZE, TIU_PMU_SIZE, SDMA_PMU_SIZE};
  static const constexpr uint64_t EngineStartAddr[PMUEngineNum] = {PMUStartAddr, \
      PMUStartAddr + PMUSize[GDMA], \
      PMUStartAddr + PMUSize[GDMA] + PMUSize[TIU]};
  */

  #pragma pack(1)
  typedef struct {
    unsigned int inst_start_time;
    unsigned int inst_end_time;
    unsigned int inst_id;
    // lower 1 bit: thread_id; higher 31 bits: bank_conflict
    unsigned int thread_id_and_bank_conflict;
  } tiu_pmu_item_t;

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
    // H0
    unsigned int inst_start_time;
    unsigned int inst_end_time;
    unsigned int inst_id: 24; // bit 0-23 inst_id; bit 24 thread id
    unsigned int thread_id: 1;
    unsigned int reserved0: 7;
    unsigned int _reserved0;
    // H1
    unsigned int m0_data_aw_cntr;
    unsigned int m0_data_w_cntr;
    unsigned int m0_data_ar_cntr;
    unsigned int reserved1;
    // H2
    unsigned int m0_data_wr_valid_cntr;
    unsigned int m0_data_wr_stall_cntr;
    unsigned int m0_data_rd_valid_cntr;
    unsigned int m0_data_rd_stall_cntr;
    // H3
    unsigned int ati_data_valid_cntr;
    unsigned int ati_data_stall_cntr;
    unsigned int ari_data_valid_cntr;
    unsigned int ari_data_stall_cntr;
    // H4
    unsigned int ati_txfifo_stall_cntr;
    unsigned int replay_number;
    unsigned int m0_data_b_st;
    unsigned int m0_data_b_end;
    // H5
    unsigned int m0_data_ar_st;
    unsigned int m0_data_ar_end;
    unsigned int m0_data_aw_st;
    unsigned int m0_data_aw_end;
    // H6
    unsigned int m0_data_rd_st;
    unsigned int m0_data_rd_end;
    unsigned int m0_data_wr_st;
    unsigned int m0_data_wr_end;
    // H7
    unsigned int ati_data_st;
    unsigned int ati_data_end;
    unsigned int ari_data_st;
    unsigned int ari_data_end;
  } cdma_pmu_item_t;
  #pragma pack()

  typedef struct {
    uint32_t type;
    uint32_t begin_usec;
    uint32_t end_usec;
    uint32_t id;
    uint32_t height;
    std::string info;
  } show_param_t;

  typedef struct {
    tiu_pmu_item_t tiu;
    gdma_pmu_item_t gdma;
    sdma_pmu_item_t sdma;
    cdma_pmu_item_t cdma;
  } pmu_item_t;

  static int allZero(const unsigned char* data, int len){
    for(int i=0; i<len; i++){
      if(data[i]!=0) return 0;
    }
    return 1;
  }

  static void profileInfo(FILE* fp, const char* fmt, ...){
    if(fp == nullptr) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);
  }

  static std::vector<tiu_pmu_item_t> parseTIUData(const uint32_t* rawData, uint64_t len, FILE* fp) {
    auto tiu_data = (tiu_pmu_item_t*)rawData;
    std::vector<tiu_pmu_item_t> result;
    size_t max_len = len/sizeof(tiu_pmu_item_t);
    size_t valid_len = 0;
    while (valid_len < max_len) {
      if(allZero((const unsigned char*)&tiu_data[valid_len], sizeof(tiu_pmu_item_t))) break;
      result.push_back(tiu_data[valid_len]);
      valid_len++;
    }
    profileInfo(fp, "tiu record_num=%d, max_record_num=%d\n", (int)valid_len, (int)max_len);
    return result;
  }

  static std::vector<gdma_pmu_item_t> parseGDMAData(const uint32_t* rawData, uint64_t len, FILE* fp) {
    auto gdma_data = (gdma_pmu_item_t*)rawData;
    std::vector<gdma_pmu_item_t> result;
    size_t max_len = len/sizeof(gdma_pmu_item_t);
    size_t valid_len = 0;
    while (valid_len < max_len) {
      if(allZero((const unsigned char*)&gdma_data[valid_len], sizeof(gdma_pmu_item_t))) break;
      result.push_back(gdma_data[valid_len]);
      valid_len++;
    }
    profileInfo(fp, "gdma record_num=%d, max_record_num=%d\n", (int)valid_len, (int)max_len);
    return result;
  }

  static std::vector<sdma_pmu_item_t> parseSDMAData(const uint32_t* rawData, uint64_t len, FILE* fp) {
    auto sdma_data = (sdma_pmu_item_t*)rawData;
    std::vector<sdma_pmu_item_t> result;
    size_t max_len = len/sizeof(sdma_pmu_item_t);
    size_t valid_len = 0;
    while (valid_len < max_len) {
      if(allZero((const unsigned char*)&sdma_data[valid_len], sizeof(sdma_pmu_item_t))) break;
      result.push_back(sdma_data[valid_len]);
      valid_len++;
    }
    profileInfo(fp, "sdma record_num=%d, max_record_num=%d\n", (int)valid_len, (int)max_len);
    return result;
  }

  static std::vector<cdma_pmu_item_t> parseCDMAData(const uint32_t* rawData, uint64_t len, FILE* fp) {
    auto cdma_data = (cdma_pmu_item_t*)rawData;
    std::vector<cdma_pmu_item_t> result;
    size_t max_len = len/sizeof(cdma_pmu_item_t);
    size_t valid_len = 0;
    while (valid_len < max_len) {
      if(allZero((const unsigned char*)&cdma_data[valid_len], sizeof(cdma_pmu_item_t))) break;
      result.push_back(cdma_data[valid_len]);
      valid_len++;
    }
    profileInfo(fp, "cdma record_num=%d, max_record_num=%d\n", (int)valid_len, (int)max_len);
    return result;
  }

  static std::vector<pmu_item_t> parsePMUData(const uint32_t* rawData, uint64_t len, uint64_t engineIdx, FILE* fp) {
    std::vector<pmu_item_t> result;
    switch (engineIdx)
    {
    case GDMA: {
      std::vector<gdma_pmu_item_t> gdma_items = parseGDMAData(rawData, len, fp);
      for (auto& item : gdma_items) {
        pmu_item_t pmu_item;
        pmu_item.gdma = item;
        result.push_back(pmu_item);
      }
      break;
    }
    case TIU: {
      std::vector<tiu_pmu_item_t> tiu_items = parseTIUData(rawData, len, fp);
      for (auto& item : tiu_items) {
        pmu_item_t pmu_item;
        pmu_item.tiu = item;
        result.push_back(pmu_item);
      }
      break;
    }
    case SDMA: {
      std::vector<sdma_pmu_item_t> sdma_items = parseSDMAData(rawData, len, fp);
      for (auto& item : sdma_items) {
        pmu_item_t pmu_item;
        pmu_item.sdma = item;
        result.push_back(pmu_item);
      }
      break;
    }
    case CDMA: {
      std::vector<cdma_pmu_item_t> cdma_items = parseCDMAData(rawData, len, fp);
      for (auto& item : cdma_items) {
        pmu_item_t pmu_item;
        pmu_item.cdma = item;
        result.push_back(pmu_item);
      }
      break;
    }
    default:
      break;
    }
    return result;
  }

  static void saveGDMAData(int coreIdx, const std::vector<pmu_item_t>& items, FILE* fp, int allInfo){
    float freq_MHz = 1000;
    float time_offset = 0;
    float period = 1/freq_MHz;
    profileInfo(fp, "Note: gdma record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
    if(items.empty()) return;
    int real_print = items.size();
    for(int i=0; i<real_print; i++){
      auto& p = items[i].gdma;
      auto total_time =  (p.inst_end_time - p.inst_start_time) * period;
      if(allInfo){
        profileInfo(
          fp,
          "[%d]---> gdma record #%d, inst_id=%d, thread_id=%d, ar_latency_cnt=%d, "
          "rip_valid_latency=%d, cycle=%d\n"
          "start=%.3fus, end=%.3fus, interval=%.3fus, gif_wr_rd_stall_cntr= %d\n"
          "axi_d0_w_cntr=%d, axi_d0_ar_cntr=%d, axi_d0_aw_cntr=%d, "
          "axi_d0_wr_stall_cntr=%d, axi_d0_rd_stall_cntr=%d, gif_mem_w_cntr=%d\n"
          "gif_mem_ar_cntr=%d, axi_d0_wr_vaild_cntr=%d, "
          "axi_d0_rd_vaild_cntr=%d, gif_wr_valid_cntr=%d, gif_rd_valid_cntr=%d\n",
          coreIdx, i, p.inst_id, p.thread_id, p.ar_latency_cnt, p.rip_valid_latency,
          p.inst_end_time - p.inst_start_time,
          p.inst_start_time * period, p.inst_end_time * period, total_time,
          p.gif_wr_rd_stall_cntr, p.axi_d0_w_cntr, p.axi_d0_ar_cntr,
          p.axi_d0_aw_cntr, p.axi_d0_wr_stall_cntr, p.axi_d0_rd_stall_cntr,
          p.gif_mem_w_cntr, p.gif_mem_ar_cntr, p.axi_d0_wr_vaild_cntr,
          p.axi_d0_rd_vaild_cntr, p.gif_wr_valid_cntr, p.gif_rd_valid_cntr);
      } else {
        profileInfo(
          fp,
          "[%d]---> gdma record #%d, inst_id=%d, thread_id=%d, cycle=%d, "
          "start=%.3fus, end=%.3fus, interval=%.3fus\n",
          coreIdx, i, p.inst_id, p.thread_id,
          p.inst_end_time - p.inst_start_time,
          p.inst_start_time * period, p.inst_end_time * period, total_time);
      }
    }
  }

  static void saveSDMAData(int coreIdx, const std::vector<pmu_item_t>& items, FILE* fp, int allInfo){
    float freq_MHz = 1000;
    float time_offset = 0;
    float period = 1/freq_MHz;
    profileInfo(fp, "Note: sdma record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
    if(items.empty()) return;
    int real_print = items.size();
    for(int i=0; i<real_print; i++){
      auto& p = items[i].sdma;
      auto total_time =  (p.inst_end_time - p.inst_start_time) * period;
      if(allInfo){
        profileInfo(
          fp,
          "[%d]---> sdma record #%d, inst_id=%d, thread_id=%d, ar_latency_cnt=%d, "
          "rip_valid_latency=%d, cycle=%d\n"
          "start=%.3fus, end=%.3fus, interval=%.3fus, gif_wr_rd_stall_cntr= %d\n"
          "axi_d0_w_cntr=%d, axi_d0_ar_cntr=%d, axi_d0_aw_cntr=%d, "
          "axi_d0_wr_stall_cntr=%d, axi_d0_rd_stall_cntr=%d, gif_mem_w_cntr=%d\n"
          "gif_mem_ar_cntr=%d, axi_d0_wr_vaild_cntr=%d, "
          "axi_d0_rd_vaild_cntr=%d, gif_wr_valid_cntr=%d, gif_rd_valid_cntr=%d\n",
          coreIdx, i, p.inst_id, p.thread_id, p.ar_latency_cnt, p.rip_valid_latency,
          p.inst_end_time - p.inst_start_time,
          p.inst_start_time * period, p.inst_end_time * period, total_time,
          p.gif_wr_rd_stall_cntr, p.axi_d0_w_cntr, p.axi_d0_ar_cntr,
          p.axi_d0_aw_cntr, p.axi_d0_wr_stall_cntr, p.axi_d0_rd_stall_cntr,
          p.gif_mem_w_cntr, p.gif_mem_ar_cntr, p.axi_d0_wr_vaild_cntr,
          p.axi_d0_rd_vaild_cntr, p.gif_wr_valid_cntr, p.gif_rd_valid_cntr);
      } else {
        profileInfo(
          fp,
          "[%d]---> sdma record #%d, inst_id=%d, thread_id=%d, cycle=%d, "
          "start=%.3fus, end=%.3fus, interval=%.3fus\n",
          coreIdx, i, p.inst_id, p.thread_id,
          p.inst_end_time - p.inst_start_time,
          p.inst_start_time * period, p.inst_end_time * period, total_time);
      }
    }
  }

  static void saveTIUData(int coreIdx, const std::vector<pmu_item_t>& items, FILE* fp){
    float freq_MHz = 1000;
    float time_offset = 0;
    int real_print = items.size();
    float period = 1/freq_MHz;
    profileInfo(fp, "Note: tiu record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
    if(items.empty()) return;
    for(int i=0; i<real_print; i++){
      auto& p = items[i].tiu;
      profileInfo(fp, "[%d]---> tiu record #%d, inst_id=%d, thread_id=%d, cycle=%d, "
        "start=%.3fus, end=%.3fus, interval=%.3fus, bank_conflict=%d\n",
        coreIdx, i, (int)(p.inst_id),
        (int)(p.thread_id_and_bank_conflict & 0x1),
        (int)(p.inst_end_time - p.inst_start_time),
        p.inst_start_time * period, p.inst_end_time * period,
        (p.inst_end_time - p.inst_start_time) * period,
        (int)(p.thread_id_and_bank_conflict >> 1));
    }
  }

  static void saveCDMAData(int portIdx, const std::vector<pmu_item_t>& items, FILE* fp){
    float freq_MHz = 1000;
    float time_offset = 0;
    int real_print = items.size();
    float period = 1/freq_MHz;
    profileInfo(fp, "Note: cdma record time_offset=%fus, freq=%gMHz, period=%.3fus\n", time_offset, freq_MHz, period);
    if(items.empty()) return;
    for(int i=0; i<real_print; i++){
      auto& p = items[i].cdma;
      auto total_time =  (p.inst_end_time - p.inst_start_time) * period;
      profileInfo(fp, "[%d]---> cdma record #%d, inst_id=%d, thread_id=%d, cycle=%d, "
          "start=%.3fus, end=%.3fus, interval=%.3fus\n",
          portIdx, i, p.inst_id, p.thread_id, p.inst_end_time - p.inst_start_time,
          p.inst_start_time * period, p.inst_end_time * period, total_time);
    }
  }

  static void savePMUData(int coreIdx, const std::vector<pmu_item_t>& items, uint64_t engineIdx, FILE* fp){
    char *dumpAllInfo = getenv("PMU_ALL_INFO");
    int allInfo = dumpAllInfo ? atoi(dumpAllInfo) : 1;
    switch (engineIdx)
    {
    case GDMA:
        saveGDMAData(coreIdx, items, fp, allInfo);
        break;
    case TIU:
        saveTIUData(coreIdx, items, fp);
        break;
    case SDMA:
        saveSDMAData(coreIdx, items, fp, allInfo);
        break;
    case CDMA:
        saveCDMAData(coreIdx, items, fp);
        break;
    default:
        break;
    }
  }
};

static std::string getFileName() {
  char* rank = getenv("OMPI_COMM_WORLD_RANK");
  if (!rank) {
    rank = getenv("LOCAL_RANK");
  }
  if(rank) return std::string("profile_data_") + std::string(rank) + std::string(".log");
  return std::string("profile_data.log");
}

static void getPMU(tpu_resource_t stream) {
    tpuRtStreamSynchronize(stream);
    static FILE *fp = fopen(getFileName().c_str(), "w");
    enum {
        GDMA = 0,
        TIU = 1,
        SDMA = 2,
        PMUEngineNum,
    };
    uint64_t PMUSize[SG2260PMUPolicy::EngineNum] = {
      SG2260PMUPolicy::GDMA_PMU_SIZE,
      SG2260PMUPolicy::TIU_PMU_SIZE,
      SG2260PMUPolicy::SDMA_PMU_SIZE,
      SG2260PMUPolicy::CDMA_PMU_SIZE
    };

    uint64_t EngineStartAddr[SG2260PMUPolicy::EngineNum] = {
      SG2260PMUPolicy::PMUStartAddr,
      SG2260PMUPolicy::PMUStartAddr + PMUSize[SG2260PMUPolicy::GDMA],
      SG2260PMUPolicy::PMUStartAddr + PMUSize[SG2260PMUPolicy::GDMA] + PMUSize[SG2260PMUPolicy::TIU],
      SG2260PMUPolicy::PMUStartAddr + PMUSize[SG2260PMUPolicy::GDMA] + PMUSize[SG2260PMUPolicy::TIU] + PMUSize[SG2260PMUPolicy::SDMA]
    };
    const int coreNum = 8;
    const int totalIterations = coreNum * SG2260PMUPolicy::CommonPMUEngineNum;
    for (int i = 0; i < totalIterations; i++) {
      int coreIdx = i / SG2260PMUPolicy::CommonPMUEngineNum;
      int engineIdx = i % SG2260PMUPolicy::CommonPMUEngineNum;

      uint64_t pmuSize = PMUSize[engineIdx];
      uint64_t pmuPhysAddr = SG2260PMUPolicy::PMUCoreOffset * coreIdx + EngineStartAddr[engineIdx];

      void *bufferAddr = nullptr;
      tpuRtMallocHost(&bufferAddr, PMUSize[engineIdx]);
      tpuRtMemcpyD2S(bufferAddr, (void *)((uint64_t*)(pmuPhysAddr)), PMUSize[engineIdx]);
      tpuRtStreamSynchronize(stream);

      auto pmuData = SG2260PMUPolicy::parsePMUData((uint32_t*)bufferAddr, pmuSize, engineIdx, fp);
      SG2260PMUPolicy::savePMUData(coreIdx, pmuData, engineIdx, fp);
      tpuRtFreeHost(bufferAddr);
    }
    for(uint64_t i = 0; i < SG2260PMUPolicy::CDMANum; i++) {
      uint64_t pmuSize = PMUSize[SG2260PMUPolicy::CDMA];
      uint64_t pmuPhysAddr = EngineStartAddr[SG2260PMUPolicy::CDMA] + i * PMUSize[SG2260PMUPolicy::CDMA];

      void *bufferAddr = nullptr;
      tpuRtMallocHost(&bufferAddr, PMUSize[SG2260PMUPolicy::CDMA]);
      tpuRtMemcpyD2S(bufferAddr, (void *)((uint64_t*)(pmuPhysAddr)), PMUSize[SG2260PMUPolicy::CDMA]);
      tpuRtStreamSynchronize(stream);
      auto pmuData = SG2260PMUPolicy::parsePMUData((uint32_t*)bufferAddr, pmuSize, SG2260PMUPolicy::CDMA, fp);
      SG2260PMUPolicy::savePMUData(i, pmuData, SG2260PMUPolicy::CDMA, fp);
      tpuRtFreeHost(bufferAddr);
    }
    fclose(fp);
}
#endif