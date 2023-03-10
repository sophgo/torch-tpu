#ifndef BMDNN_HELPERS_H
#define BMDNN_HELPERS_H

#include "bmlib_runtime.h"
#include <assert.h>

#ifndef USING_CMODEL
#define TPU_CHECK_RET(call)                                                    \
  do {                                                                        \
    bm_status_t ret = (bm_status_t)call;                                                   \
    if (ret != BM_SUCCESS) {                                                  \
      bmlib_log("BM_CHECK",16,"TPU_CHECK_RET fail %s: %s: %d\n", __FILE__, __func__, __LINE__); \
    }                                                                         \
  } while (0)
#else
#define TPU_CHECK_RET(call)                     \
  do {                                         \
    bm_status_t ret = call;                    \
    if (ret != BM_SUCCESS) {                   \
      bmlib_log("BM_CHECK",16,"TPU_CHECK_RET failed %d\n", ret);\
      assert(0);                               \
      exit(-ret);                              \
    }                                          \
  } while (0)
#endif

#define DEVICE_MEM_ASSIGN_OR_COPY(handle, raw_mem, need_copy, len, new_mem)\
  do{\
      if (bm_mem_get_type(raw_mem) == BM_MEM_TYPE_SYSTEM){\
        TPU_CHECK_RET(bm_mem_convert_system_to_device_coeff_byte(\
            handle, &new_mem, raw_mem, need_copy, \
            (len))); \
      }else{\
          new_mem = raw_mem; \
      }\
    }while(0)

#define DEVICE_MEM_NEW_BUFFER(handle, buffer_mem, len)\
  TPU_CHECK_RET(bm_malloc_device_byte(handle, &buffer_mem, len));
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
        TPU_CHECK_RET(bm_memcpy_d2s(handle, bm_mem_get_system_addr(raw_mem), new_mem)); \
        bm_free_device(handle, new_mem); \
      }\
    } while(0)

#endif // BMDNN_HELPERS_H

