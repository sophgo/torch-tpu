#include "sg_api_struct.h"
#include "sgdnn_api.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>

#if defined BACKEND_1684X
#include "torch_tpu_kernel_data.h"
#elif defined BACKEND_SG2260
#include "tpukernel_launcher.hpp"
#endif

#ifdef USING_PERF_MODE
#include <iostream>
#endif


tpu_status_t sgdnnInitialize( tpu_resource_t resource );

#if defined BACKEND_1684X
static std::map<tpu_resource_t, tpu_kernel_module_t> tpu_kernel_module;
tpu_kernel_module_t get_kernel_module(bm_handle_t handle)
{
  sgdnnInitialize(handle);
  return tpu_kernel_module[handle];
}
#elif defined BACKEND_SG2260
static TPUKernelLauncher* pkernel_launcher = nullptr;
tpuRtKernelModule_t get_kernel_module(tpuRtStream_t stream)
{
  return pkernel_launcher->get_kernel_module(stream);
}
#endif

tpu_status_t sgdnnInitialize( tpu_resource_t resource )
{
#if defined BACKEND_1684X
  if ( tpu_kernel_module.find ( resource  ) != tpu_kernel_module.end() )
  {
    return SG_SUCCESS;
  }
  const char* p = torch_tpu_kernel_data;
  const size_t length = torch_tpu_kernel_data_length;
  tpu_kernel_module_t tpu_module = tpu_kernel_load_module ( resource , ( const char * ) p, length );
  tpu_kernel_module.insert ( std::pair<tpu_resource_t, tpu_kernel_module_t> ( resource , tpu_module ) );
#elif defined BACKEND_SG2260
  if (pkernel_launcher == nullptr)
  {
    pkernel_launcher = new TPUKernelLauncher();
  }
  SGDNN_CHECK( pkernel_launcher->register_kernel_module(resource) == SG_SUCCESS);
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}

tpu_status_t sgdnnDeinitialize ( tpu_resource_t resource  )
{
#if defined BACKEND_1684X
  if ( tpu_kernel_module.find ( resource  ) == tpu_kernel_module.end() )
  {
    return SG_SUCCESS;
  }
  SGDNN_CHECK ( tpu_kernel_module.erase ( resource  ) );
#elif defined BACKEND_SG2260
  if ( pkernel_launcher )
  {
    SGDNN_CHECK ( pkernel_launcher->unload_kernel_module(resource) == SG_SUCCESS );
    delete pkernel_launcher;
    pkernel_launcher = nullptr;
  }
#else
  SGDNN_CHECK ( false );
#endif
  return SG_SUCCESS;
}