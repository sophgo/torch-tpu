#ifndef _COMMON_DEF_H_
#define _COMMON_DEF_H_

#include "../../firmware_core/include/tpu_defs.h"

// The data type number is the same with bmcompiler
typedef enum {
  SG_DTYPE_FP32 = 0,
  SG_DTYPE_FP16 = 1,
  SG_DTYPE_INT8 = 2,
  SG_DTYPE_UINT8 = 3,
  SG_DTYPE_INT16 = 4,
  SG_DTYPE_UINT16 = 5,
  SG_DTYPE_INT32 = 6,
  SG_DTYPE_UINT32 = 7,
  SG_DTYPE_BFP16 = 8,
  SG_DTYPE_UNKNOWN = -1,
} sg_data_type_t;

typedef enum {
  SG_ROUND_INF = 0,     // 1.5 -> 2   -1.5 -> -2
  SG_ROUND_UP = 1,      // 1.5 -> 2   -1.5 -> -1
  SG_ROUND_DOWN = 2,    // 1.5 -> 1   -1.5 -> -2
  SG_ROUND_EVEN = 3,    // 1.5 -> 2    2.5 -> 2
  SG_ROUND_ODD = 4,     // 1.5 -> 1    0.5 -> 1
  SG_ROUND_ZERO = 5,    // 1.5 -> 1   -1.5 -> -1
  SG_TRIM_ZERO = 6,     // 1.6 -> 1   -1.6 -> -1
  SG_TRIM_INF = 7,      // 1.4 -> 2   -1.4 -> -2
  SG_TRIM_UP = 8,       // 1.4 -> 2   -1.6 -> -1
  SG_TRIM_DOWN = 9,     // 1.6 -> 1   -1.4 -> -2
} sg_round_mode_t;

typedef enum {
  SG_REDUCE_MEAN = 0,
  SG_REDUCE_SUM  = 1,
  SG_REDUCE_MAX  = 2,
  SG_REDUCE_MIN  = 3,
  SG_REDUCE_PROD = 4
} sg_reduce_method_t;

typedef enum {
  BINARY_ADD          = 0,
  BINARY_SUB          = 1,
  BINARY_MUL          = 2,
  BINARY_DIV          = 3,
  BINARY_MAX          = 4,
  BINARY_MIN          = 10000,
  BINARY_GT           = 10001,
  BINARY_GE           = 10002,
  BINARY_LT           = 10003,
  BINARY_LE           = 10004,
  BINARY_EQ           = 10005,
  BINARY_NE           = 10006,
  BINARY_SQUARED_DIFF = 10007,
  BINARY_FLOOR_MOD    = 10008,
  BINARY_FLOOR_DIV    = 10009
} sg_binary_type_t;

typedef enum {
  ACTIVE_TANH      = 0,
  ACTIVE_SIGMOID   = 1,
  ACTIVE_RELU      = 2,
  ACTIVE_EXP       = 3,
  ACTIVE_ELU       = 4,
  ACTIVE_SQRT      = 5,
  ACTIVE_SQUARE    = 6,
  ACTIVE_RSQRT     = 7,
  ACTIVE_ABSVAL    = 8,
  ACTIVE_LN        = 9,
  ACTIVE_ROUND     = 10,
  ACTIVE_CEIL      = 11,
  ACTIVE_FLOOR     = 12,
  ACTIVE_SIN       = 13,
  ACTIVE_COS       = 14,
  ACTIVE_IS_FINITE = 15,
  ACTIVE_MISH      = 16,
  ACTIVE_SWISH     = 17,
  ACTIVE_HSWISH    = 18,
  ACTIVE_SILU      = 19,
  ACTIVE_ARCSIN    = 20,
  ACTIVE_ARCCOS    = 21,
  ACTIVE_ARCSINH   = 22,
  ACTIVE_ARCCOSH   = 23,
  ACTIVE_ARCTANH   = 24,
  ACTIVE_SINH      = 25,
  ACTIVE_COSH      = 26,
  ACTIVE_TAN       = 27,
  ACTIVE_SIGN      = 28,
  ACTIVE_GELU      = 29,
  ACTIVE_ERF       = 30,
  ACTIVE_HSIGMOID  = 31,
  ACTIVE_LOG_SIGMOID = 32,
  ACTIVE_SOFT_PLUS = 33,
  ACTIVE_SOFT_SIGN = 34,
} sg_active_type_t;

// Channel shift macro(left,right,circle left,circle right)
typedef enum {
  SHIFT_L  = 0,
  SHIFT_R  = 1,
  SHIFT_CL = 2,
  SHIFT_CR = 3
} sg_shift_type_t;

#define IS_INT8(t) (((t) == SG_DTYPE_UINT8) || ((t) == SG_DTYPE_INT8))
#define IS_INT16(t) (((t) == SG_DTYPE_UINT16) || ((t) == SG_DTYPE_INT16))
#define IS_INT32(t) (((t) == SG_DTYPE_UINT32) || ((t) == SG_DTYPE_INT32))
#define IS_FP16(t) ((t) == SG_DTYPE_FP16 || (t) == SG_DTYPE_BFP16)
#define IS_FP(t) (((t) == SG_DTYPE_FP32) || ((t) == SG_DTYPE_FP16) || ((t) == SG_DTYPE_BFP16))
#define IS_SIGN(t) (((t) == SG_DTYPE_INT8) || ((t) == SG_DTYPE_INT16) || ((t) == SG_DTYPE_INT32))

static inline unsigned sg_dtype_len(sg_data_type_t t)
{
    switch (t)
    {
        case SG_DTYPE_FP32:
        case SG_DTYPE_UINT32:
        case SG_DTYPE_INT32:
            return 4;
        case SG_DTYPE_UINT8:
        case SG_DTYPE_INT8:
            return 1;
        case SG_DTYPE_FP16:
        case SG_DTYPE_BFP16:
        case SG_DTYPE_UINT16:
        case SG_DTYPE_INT16:
            return 2;
        default:
            return 0;
    }
}

// Keep in sync with bmcompiler_net_param.h
typedef enum {
  /* 3D group if this group has CONV3D/DECONV3D/POOL3D
   * for 1684 float32, data in local memory storage as {d * n, c, h, w}
   * for 1684 int8, data in local memory storage as {n, d * c, h, w}
   * for 1684X, data in local memory storage as {d * n, c, h, w}
   * data in global memory always storage as {n, c, d, h, w}
   * group_type < 8, because 1684 dynamic compile reserved `3bit` for group_type
   */
  FW_GROUP_NORMAL = 0,
  GROUP_3D = 1,
} group_type_t;

#define UNUSED(x) (void)(x)

typedef enum {
  CAFFE_SUPPORT = 0,
  TENSORFLOW_SUPPORT = 1,
  CAFFE_NEAREST = 2,
  TENSORFLOW_NEAREST = 3,
  PYTORCH_SUPPORT = 4,
  PYTORCH_NEAREST = 5,
  OPENCV_BILINEAR = 6,
  ONNX_NEAREST = 7,
} PLATFORM_SUPPORT;

typedef enum {
    GridSampleNearest = 0,
    GridSampleBilinear = 1,
} GridSampleInterpMode;

typedef enum {
    GridSampleZeros = 0,
    GridSampleBorder = 1,
    GridSampleReflection = 2,
} GridSamplePaddingMode;

#define MIN_PROPOSAL_NUM (1)
#define MAX_PROPOSAL_NUM (40000)//(65536)
#define MAX_SOFT_SUPPORT_PROPOSAL_NUM (22500)
#define ALL_MASK_IN_L2_MAX_SIZE (1400)
#define ALL_MASK_IN_L2_SOFT_NMS_MAX_SIZE (350)
typedef enum sgdnn_nms_alg_ {
    HARD_NMS = 0,
    SOFT_NMS,
    ADAPTIVE_NMS,
    SSD_NMS,
    MAX_NMS_TYPE
} sgdnn_nms_alg_e;
typedef enum {
    LINEAR_WEIGHTING = 0,
    GAUSSIAN_WEIGHTING,
    MAX_WEIGHTING_TYPE
} sgdnn_weighting_method_e;
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
}__attribute__((packed)) face_rect_t;

typedef struct nms_proposal {
    int          size;
    face_rect_t  face_rect[MAX_PROPOSAL_NUM];
    int          capacity;
    face_rect_t *begin;
    face_rect_t *end;
} __attribute__((packed)) nms_proposal_t;
#endif // _COMMON_DEF_H_

#define SWAP_VAL(a, b) \
  a ^= b;              \
  b ^= a;              \
  a ^= b
