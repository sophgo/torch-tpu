#ifndef __bm1684x__
#include <string>
#include <numeric>
#include <iostream>
#include <cassert>
#include "tpukernel_multicore.hpp"
#include "sg_api_struct.h"
#include "message.h"
// #include "bmlib_internal.h"
#include "bmlib_runtime.h"

extern "C" {
bm_status_t bm_send_api_to_core(
  bm_handle_t  handle,
  sg_api_id_t  api_id,
  const unsigned char *api,
  u32          size,
  int          core_id);

bm_status_t bm_sync_api(bm_handle_t handle);
bm_status_t bm_sync_api_from_core(bm_handle_t handle, int core_id);

/**
 * @name    bm_get_tpu_scalar_num
 * @brief   To get the core number of TPU scalar
 * @ingroup bmlib_runtime
 *
 * @param [in] handle    The device handle
 * @param [out] core_num The core number of TPU scalar
 * @retval  BM_SUCCESS  Succeeds.
 *          Other code  Fails.
 */
bm_status_t bm_get_tpu_scalar_num(bm_handle_t handle, unsigned int *core_num);
}

int tpukernel_get_core_num(bm_handle_t handle) {
  unsigned int num = 0;
  auto status = bm_get_tpu_scalar_num(handle, &num);
  assert(status == BM_SUCCESS);
  return num;
}

static std::vector<int> str2list(const char *s) {
  const char *ptr = s;
  std::vector<int> results;
  while (ptr[0] != '\0') {
    auto val = strtol(ptr, (char **)&ptr, 10);
    results.push_back(val);
    if (ptr[0] == ',')
      ptr++;
  }
  return results;
}

std::vector<int> tpukernel_get_cores_by_env(bm_handle_t handle, int env_id) {
  std::string env_str = std::string("TPUKERNEL_USING_CORES_") + std::to_string(env_id);
  auto raw_str = getenv(env_str.c_str());
  if(!raw_str){
    return tpukernel_get_all_cores(handle);
  }
  std::cout << "Using ENV " << env_str << "=" << raw_str << std::endl;
  return str2list(raw_str);
}

std::vector<int> tpukernel_get_all_cores(bm_handle_t handle) {
  auto num = tpukernel_get_core_num(handle);
  std::vector<int> core_list(num);
  std::iota(core_list.begin(), core_list.end(), 0);
  return core_list;
}

bm_status_t tpukernel_launch_async_on_cores(
    bm_handle_t handle, const char *func_name,
    const void *api, size_t api_size,
    const std::vector<int> &core_list)
{
  auto func_name_len = strlen(func_name)+1;
  auto aligned_size = ((sizeof(sg_api_core_info_t) + api_size + func_name_len)+3)&(~0x3);
  std::vector<unsigned char> param_data(aligned_size);
  auto core_info = (sg_api_core_info_t*)param_data.data();
  core_info->core_num = core_list.size();
  core_info->name_len = func_name_len;
  core_info->api_size = api_size;
  unsigned int group_msg_id = 0;
  for(auto core_id: core_list){
    group_msg_id |= 1<<core_id;
  }
  // group_msg_id
  core_info->core_msg_id = group_msg_id;
  // put func_name after api_param for memory aligned requirement
  memcpy(core_info->api_data, api, api_size);
  memcpy(core_info->api_data+api_size, func_name, func_name_len);

  for(size_t i = 0; i< core_list.size(); i++){
    core_info->core_idx = i;
    auto core_id = core_list[i];
    bm_status_t status = bm_send_api_to_core(
        handle,
        SG_API_TPUKERNEL_MULTICORE,
        param_data.data(), param_data.size(),
        core_id);
    if(status != BM_SUCCESS) {
      std::cerr<<"core_idx="<<i<<" core_id="<<core_id<<" status ="<<status<<std::endl;
      return status;
    }
  }
  return BM_SUCCESS;
}

bm_status_t tpukernel_sync_cores(bm_handle_t handle, const std::vector<int>& core_list) {
  if(core_list.empty()){
    return bm_sync_api(handle);
  }
  for(auto core_id: core_list){
    auto status = bm_sync_api_from_core(handle, core_id);
    if(status != BM_SUCCESS){
      return status;
    }
  }
  return BM_SUCCESS;
}

#endif