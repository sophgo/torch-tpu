#include <string>
#include <numeric>
#include <iostream>
#include <cassert>
#include <cstring>
#include "message.h"
#include "sg_api_struct.h"
#include "tpukernel_launcher.hpp"

#define MAX_MSG_QUEUE_LENGTH 2048
#define MAX_CORE_NUM 8

#ifdef DUMP_INS
#include <dlfcn.h>
struct CmdDump
{
    void *handle;
    typedef void (*Func)();
    Func inc_file_dump_group_num = nullptr;
    CmdDump()
    {
        char *fw_path = getenv("TPUKERNEL_FIRMWARE_PATH");
        if (!fw_path)
            return;
        handle = dlopen(fw_path, RTLD_LAZY);
        inc_file_dump_group_num = (Func)(dlsym(handle, "inc_file_dump_group_num"));
    }

    void operator()()
    {
        if (inc_file_dump_group_num)
            inc_file_dump_group_num();
    }
} cmd_dump;
#endif

TPUKernelLauncher::TPUKernelLauncher()
{
  _library_file = getenv("TPUKERNEL_FIRMWARE_PATH");
  const char* value_str = getenv("TPUTRAIN_CORE_NUM");
  if (value_str)
    _core_num = atoi(value_str);
  else
    _core_num = MAX_CORE_NUM;

  printf("[TPUKERNEL_FIRMWARE_PATH] : %s, [max_core_num] : %d \n",
          _library_file, _core_num);
}

bool TPUKernelLauncher::stream_registered(tpuRtStream_t stream)
{
  bool registered = false;
  if (_stream_kernel_modules.find(stream) != _stream_kernel_modules.end()){
    registered = true;
  }
  return registered;
}

#include "torch_tpu_kernel_data.h"

tpuRtStatus_t TPUKernelLauncher::register_kernel_module(tpuRtStream_t stream){
  if (!stream_registered(stream)){
    tpuRtKernelModule_t sg_module;
    if (_library_file)
    {
      sg_module = tpuRtKernelLoadModuleFile(_library_file, stream);
      if (!sg_module)
      {
        printf("Failed to load kernel module \"%s\"\n", _library_file);
      }
    } else {
      sg_module = tpuRtKernelLoadModule(torch_tpu_kernel_data, torch_tpu_kernel_data_length, stream);
    }
    _stream_kernel_modules[stream] = sg_module;
  }
  return tpuRtSuccess;
}

tpuRtStatus_t TPUKernelLauncher::unload_kernel_module(tpuRtStream_t stream) {
  for ( auto it : _stream_kernel_modules){
    tpuRtKernelUnloadModule(it.second, it.first);
  }
  return tpuRtSuccess;
}

tpuRtStatus_t TPUKernelLauncher::launch_async(
    const char* func_name, const void* api, size_t api_size, tpuRtStream_t stream, int group_num, int block_num) {
  tpuRtKernelModule_t kernel_module = _stream_kernel_modules[stream];
  tpuRtStatus_t status = tpuRtKernelLaunchAsync(kernel_module, func_name, (void *)api, api_size, group_num, block_num, stream);
#ifdef DUMP_INS
  tpuRtStreamSynchronize(stream);
  cmd_dump();
#endif
  return status;
}

tpuRtStatus_t TPUKernelLauncher::launch_sync(
    const char* func_name, const void* api, size_t api_size, tpuRtStream_t stream, int group_num, int block_num) {
  tpuRtKernelModule_t kernel_module = _stream_kernel_modules[stream];
  tpuRtStatus_t status = tpuRtKernelLaunch(kernel_module, func_name, (void*)api, api_size, group_num, block_num, stream );
#ifdef DUMP_INS
  cmd_dump();
#endif
  return status;
}

tpuRtStatus_t TPUKernelLauncher::cache_malloc(
  void** p_dev_ptr, int64_t size ) {
  return _cache_mem_Mgr.cache_malloc(p_dev_ptr, size);
}

tpuRtStatus_t TPUKernelLauncher::cache_free(void* dev_ptr, tpuRtStream_t stream){
  return _cache_mem_Mgr.cache_free(dev_ptr, stream);
}

tpuRtKernelModule_t TPUKernelLauncher::get_kernel_module(tpuRtStream_t stream)
{
    this->register_kernel_module(stream);
    return _stream_kernel_modules[stream];
}

/*********************************************************************
*****************   Cached Device Memory Manager ********************
**********************************************************************/

Cached_DevMem_Mgr::Cached_DevMem_Mgr()
{
  _stop = false;
  _free_msg_t = std::thread(&Cached_DevMem_Mgr::free_cache_daemon, this);
}

Cached_DevMem_Mgr::~Cached_DevMem_Mgr()
{
  _stop = true;
  std::unique_lock<std::mutex> lock(_free_msg_mtx);
  _free_msg_cv.notify_all();
  lock.unlock();
  _free_msg_t.join();
}


tpuRtStatus_t Cached_DevMem_Mgr::cache_malloc(
  void** p_dev_ptr, int64_t size ) {
    if (_cache_dev_mem.find(size) == _cache_dev_mem.end()){
      tpuRtStatus_t status = tpuRtMalloc(p_dev_ptr, size, NO_USE);
      if (status != tpuRtSuccess) { throw; }
      _cache_dev_mem[size] = {*p_dev_ptr};
      return tpuRtSuccess;
    }
    else{
      auto dev_mems = _cache_dev_mem[size];
      for (auto& dev_mem : dev_mems){
        if (malloc_cached_mem(p_dev_ptr, dev_mem)){ return tpuRtSuccess; }
      }
      tpuRtStatus_t status = tpuRtMalloc(p_dev_ptr, size, NO_USE);
      if (status != tpuRtSuccess) { throw; }
      _cache_dev_mem[size].push_back(*p_dev_ptr);
      {
        std::lock_guard<std::mutex> lock(_cache_mem_mtx);
        set_mem_in_use(*p_dev_ptr);
      }
    }
    return tpuRtSuccess;
}

tpuRtStatus_t Cached_DevMem_Mgr::cache_free(void* dev_ptr, tpuRtStream_t stream)
{
  struct tpuRtEvent *cache_free_event;
  tpuRtEventCreate(&cache_free_event);
  tpuRtEventRecord(cache_free_event, stream);
  FreeMsg msg = {.dev_ptr = dev_ptr, .event = cache_free_event, .stream = stream};
  std::lock_guard<std::mutex> lock(_free_msg_mtx);
  while(_free_msg_queue.size() >= MAX_MSG_QUEUE_LENGTH) {_free_msg_cv.notify_one();} // block
  _free_msg_queue.push(msg);
  _free_msg_cv.notify_one();
  return tpuRtSuccess;
}

bool Cached_DevMem_Mgr::malloc_cached_mem(void** p_to_malloc, void* cached_mem){
  std::lock_guard<std::mutex> lock(_cache_mem_mtx);
  if (mem_in_use(cached_mem)) {
    return false;
  } else {
    *p_to_malloc = cached_mem;
    set_mem_in_use(cached_mem);
    return true;
  }
}

bool Cached_DevMem_Mgr::mem_in_use(void* dev_ptr){
  if(_used_mem.find(dev_ptr) != _used_mem.end()) {
    return true;
  } else {
    return false;
  }
}

void Cached_DevMem_Mgr::set_mem_in_use(void* dev_ptr){
  _used_mem.emplace(dev_ptr);
}

void Cached_DevMem_Mgr::set_mem_no_use(void* dev_ptr){
  _used_mem.erase(dev_ptr);
}

void Cached_DevMem_Mgr::free_cache_daemon(){
  while( !_stop )
  {
    std::unique_lock<std::mutex> lock(_free_msg_mtx);
    _free_msg_cv.wait(lock);
    if (_free_msg_queue.empty()) { continue; }
    FreeMsg msg = _free_msg_queue.front();
    _free_msg_queue.pop();
    lock.unlock();

    tpuRtStreamWaitEvent(msg.stream, msg.event);

    std::lock_guard<std::mutex> lock_cached_mem(_cache_mem_mtx);
    set_mem_no_use(msg.dev_ptr);
  }
}

void sgdnn_dump_data_into_file(const void* data, size_t size, const char* file_name)
{
  // std::ofstream file(file_name, std::ios::binary);
  // file.write((const char*)data, size);
  // file.close();
  FILE *fp = fopen(file_name, "wb");
  if (fp) {
    fwrite(data, size, 1, fp);
    fclose(fp);
  }
}

void sgdnn_dump_tensor_into_file(tpu_resource_t tpu_resource, tpu_device_mem_t tensor, size_t size, const char* file_name)
{
  printf(">>>> tensor addr: %lld\n", (unsigned long long)tensor);
  printf(">>>> Dump tensor into file: %s, size: %ld\n", file_name, size);
  void* data = malloc(size);
  SAFE_CALL(sgdnnMemcpyD2S(tpu_resource, data, tensor, size));
  sgdnn_dump_data_into_file(data, size, file_name);
  printf(">>>>> after write\n");
  free(data);
}