#pragma once
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <string>
// file io
#include <fstream>
#include <tpuv7_rt.h>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include "sgdnn_runtime.h"


/**
 * @brief Wrappr class of the launch functions
 */
class Cached_DevMem_Mgr {

typedef struct FreeMsg{
  void*              dev_ptr;
  struct tpuRtEvent* event;
  tpuRtStream_t      stream;
} FreeMsg;

public:
  Cached_DevMem_Mgr();
  ~Cached_DevMem_Mgr();

  tpuRtStatus_t cache_malloc(void** p_dev_ptr, int64_t size);
  tpuRtStatus_t cache_free(void* dev_ptr, tpuRtStream_t stream);
  void free_cache_daemon();

private:
  bool malloc_cached_mem(void** p_to_malloc, void* cached_mem);
  bool mem_in_use(void* dev_ptr);
  void set_mem_in_use(void* dev_ptr);
  void set_mem_no_use(void* dev_ptr);
  std::map<int64_t, std::vector<void*>> _cache_dev_mem;
  std::mutex _cache_mem_mtx;
  std::set<void*> _used_mem;

  std::thread _free_msg_t;
  std::mutex _free_msg_mtx;
  std::queue<FreeMsg> _free_msg_queue;
  std::condition_variable _free_msg_cv;
  bool _stop;
};


class TPUKernelLauncher {
public:
  TPUKernelLauncher();

  tpuRtStatus_t register_kernel_module(tpuRtStream_t stream);

  tpuRtStatus_t unload_kernel_module(tpuRtStream_t stream);

  tpuRtStatus_t launch_async(const char* func_name, const void* api, size_t api_size, tpuRtStream_t stream, int group_num, int block_num);

  tpuRtStatus_t launch_sync(const char* func_name, const void* api, size_t api_size, tpuRtStream_t stream, int group_num, int block_num);

  tpuRtStatus_t cache_malloc(void** p_dev_ptr, int64_t size);
  tpuRtStatus_t cache_free(void* dev_ptr, tpuRtStream_t stream);

  tpuRtKernelModule_t get_kernel_module(tpuRtStream_t stream);

private:
  bool stream_registered(tpuRtStream_t stream);
  size_t _wrap_param(const void* api, size_t api_size, const char* func_name);

  std::map<tpuRtStream_t, tpuRtKernelModule_t> _stream_kernel_modules;
  char* _library_file;
  int _core_num;
  Cached_DevMem_Mgr _cache_mem_Mgr;
};

void sgdnn_dump_data_into_file(const void* data, size_t size, const char* file_name);
void sgdnn_dump_tensor_into_file(tpu_resource_t tpu_resource, tpu_device_mem_t tensor, size_t size, const char* file_name);