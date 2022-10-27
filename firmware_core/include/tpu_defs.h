#ifndef _OK_DEFS_
#define _OK_DEFS_
typedef unsigned int local_addr_t;
typedef unsigned int static_addr_t;
typedef unsigned long long l2_sram_addr_t;
typedef unsigned long long system_addr_t;
typedef unsigned long long global_addr_t;
typedef unsigned long long addr_t;
typedef enum {
    DT_INT8   = (0 << 1) | 1,
    DT_UINT8  = (0 << 1) | 0,
    DT_INT16  = (3 << 1) | 1,
    DT_UINT16 = (3 << 1) | 0,
    DT_FP16   = (1 << 1) | 1,
    DT_BFP16  = (5 << 1) | 1,
    DT_INT32  = (4 << 1) | 1,
    DT_UINT32 = (4 << 1) | 0,
    DT_FP32   = (2 << 1) | 1
} data_type_t;
typedef enum {
    RM_HALF_TO_EVEN        = 0,
    RM_HALF_AWAY_FROM_ZERO = 1,
    RM_TOWARDS_ZERO        = 2,
    RM_DOWN                = 3,   /* FLOOR */
    RM_UP                  = 4,   /* CEIL */
    RM_HALF_UP             = 5,
    RM_HALF_DOWN           = 6
} rounding_mode_t;

typedef enum {
  REDUCE_MEAN = 0,
  REDUCE_SUM  = 1,
  REDUCE_MAX  = 2,
  REDUCE_MIN  = 3,
  REDUCE_PROD = 4
} reduce_method_t;

#endif /* _OK_DEFS_ */
