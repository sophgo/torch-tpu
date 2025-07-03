#ifndef SGDNN_API_H
#define SGDNN_API_H

#include "sg_api_struct.h"
#include "sgdnn_runtime.h"
#include <map>
#include <vector>

#if defined(__cplusplus)
extern "C" {
#endif
  
tpu_status_t sgdnnInitialize( tpu_resource_t resource );

tpu_status_t sgdnnDeinitialize( tpu_resource_t resource );

#if defined(__cplusplus)
}
#endif

#endif /* SGDNN_API_H */
