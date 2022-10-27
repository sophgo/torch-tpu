#ifndef BMDNN_HELPERS_H
#define BMDNN_HELPERS_H

//#include "../include/sgdnn_ext_api.h"
#include "sgdnn_api.h"
#include "bmlib_internal.h"
#include "bmlib_utils.h"
#include "bmlib_runtime.h"
#include <assert.h>
//#ifdef __linux__
//#include "common.h"
//#else
//#include "..\..\..\common\bm1684\include_win\common_win.h"
//#endif

#define DEVICE_MEM_ASSIGN_OR_COPY(handle, raw_mem, need_copy, len, new_mem)\
  do{\
      if (bm_mem_get_type(raw_mem) == BM_MEM_TYPE_SYSTEM){\
        BM_CHECK_RET(bm_mem_convert_system_to_device_coeff_byte(\
            handle, &new_mem, raw_mem, need_copy, \
            (len))); \
      }else{\
          new_mem = raw_mem; \
      }\
    }while(0)

#define DEVICE_MEM_NEW_BUFFER(handle, buffer_mem, len)\
  BM_CHECK_RET(bm_malloc_device_byte(handle, &buffer_mem, len));
#define DEVICE_MEM_DEL_BUFFER(handle, buffer_mem)\
  bm_free_device(handle, buffer_mem)

#define DEVICE_MEM_NEW_INPUT(handle, src, len, dst)\
    DEVICE_MEM_ASSIGN_OR_COPY(handle, src, true, len, dst)

#define DEVICE_MEM_NEW_OUTPUT(handle, src, len, dst)\
    DEVICE_MEM_ASSIGN_OR_COPY(handle, src, false, len, dst)

#define DEVICE_MEM_DEL_INPUT(handle, raw_mem, new_mem)\
    do{\
      if (bm_mem_get_type(raw_mem) == BM_MEM_TYPE_SYSTEM){\
        bm_free_device(handle, new_mem); \
      }\
    } while(0)

#define DEVICE_MEM_DEL_OUTPUT(handle, raw_mem, new_mem)\
    do{\
      if (bm_mem_get_type(raw_mem) == BM_MEM_TYPE_SYSTEM){\
        BM_CHECK_RET(bm_sync_api(handle)); \
        BM_CHECK_RET(bm_memcpy_d2s(handle, bm_mem_get_system_addr(raw_mem), new_mem)); \
        bm_free_device(handle, new_mem); \
      }\
    } while(0)

#endif // BMDNN_HELPERS_H

