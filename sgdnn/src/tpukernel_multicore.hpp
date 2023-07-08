#pragma once

#include <vector>
#include "device_mem_allocator.h"

/**
 * @brief get number of cores on current chip
 *
 * @param handle: device_handle
 * @return int: core number
 */
int tpukernel_get_core_num(bm_handle_t handle);

/**
 * @brief config core list through environment variable
 * When env_id=0, and TPUKERNEL_USING_CORES_0=0,1 is set, the returned list is {0,1}
 * If no env is set, then return all cores
 *
 * @param handle
 * @param env_id: Generate env name "TPUKERNEL_USING_CORES_<env_id>"
 * @return std::vector<int>: list of core ids
 */
std::vector<int> tpukernel_get_cores_by_env(bm_handle_t handle, int env_id=0);

/**
 * @brief get all cores on current chip
 *
 * @param handle
 * @return std::vector<int>: list of core ids
 */
std::vector<int> tpukernel_get_all_cores(bm_handle_t handle);

/**
 * @brief async launch tpu kernel api on assigned cores
 *
 * @param handle
 * @param func_name: registered tpu kernel func name from device
 * @param api_param: the tpu kernel func parameter data
 * @param api_size: the tpu kernel func parameter size
 * @param core_list: list of cores to run the tpu kernel
 * @return bm_status_t: return BM_SUCCESS if succeed
 */
bm_status_t tpukernel_launch_async_on_cores(
    bm_handle_t handle, const char *func_name,
    const void *api_param, size_t api_size,
    const std::vector<int> &core_list);

/**
 * @brief sync a list of cores to assure completion of tasks on the cores
 *
 * @param handle
 * @param core_list: list of cores to sync
 * @return bm_status_t: return BM_SUCCESS if succeed
 */
bm_status_t tpukernel_sync_cores(bm_handle_t handle, const std::vector<int>& core_list);


/**
 * @brief Wrappr class of the launch functions
 */
class TPUKernelLauncher {
public:
  TPUKernelLauncher(bm_handle_t handle) : _handle(handle),_allocator(handle) {
  }

  bm_status_t launch_async(const char* func_name, const void* api, size_t api_size) {
    if(core_list.empty()) env_cores();
    return tpukernel_launch_async_on_cores(_handle, func_name, api, api_size, core_list);
  }

  bm_status_t launch_sync(const char* func_name, const void* api, size_t api_size) {
    auto status = launch_async(func_name, api, api_size);
    if(status != BM_SUCCESS) return status;
    return sync();
  }

  template<typename APIType>
  bm_status_t launch_async(const char* func_name, const APIType& api) {
    // if no cores are set, use cores configed by env
    return launch_async(func_name, &api, sizeof(api));
  }

  template<typename APIType>
  bm_status_t launch_sync(const char* func_name, const APIType& api) {
    auto status = launch_async(func_name, api);
    if(status != BM_SUCCESS) return status;
    return sync();
  }

  bm_status_t sync() {
    auto status = tpukernel_sync_cores(_handle, core_list);
    if(status != BM_SUCCESS) return status;
    _allocator.flush_output();
    for(auto& mem: buffers){
      _allocator.dealloc(mem);
    }
    return status;
  }

  TPUKernelLauncher &env_cores(int env_id=0) {
    core_list = tpukernel_get_cores_by_env(_handle, env_id);
    return *this;
  }

  TPUKernelLauncher &all_cores() {
    core_list = tpukernel_get_all_cores(_handle);
    return *this;
  }

  TPUKernelLauncher &cores(const std::vector<int> &cores) {
    core_list = cores;
    return *this;
  }

  // the input is on host, map_input will alloc a device memory and do s2d automatically
  // or return the original device_addr directly
  unsigned long long map_input(bm_device_mem_t input, size_t num_elem, SgdnnDataType_t dtype){
    return _allocator.map_input_to_device_addr(input, num_elem, dtype);
  }

  // the output is on host, map_output will alloc a device memory and do d2s when sync is called
  unsigned long long map_output(bm_device_mem_t output, size_t num_elem, SgdnnDataType_t dtype, bool is_inplace=false){
    bool is_post_copy = true;
    return _allocator.map_output_to_device_addr(output, num_elem, dtype, is_post_copy, is_inplace);
  }

  // managed buffer, the buffer will be freed when sync is called
  unsigned long long use_buffer(size_t byte_size){
    auto mem = _allocator.alloc_on_device(byte_size);
    buffers.push_back(mem);
    return bm_mem_get_device_addr(mem);
  }

private:
  bm_handle_t _handle;
  DeviceMemAllocator _allocator;
  std::vector<int> core_list;
  std::vector<bm_device_mem_t> buffers;
};
