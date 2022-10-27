#ifndef __GDMA_REG_VALUE_H__
#define __GDMA_REG_VALUE_H__

////////////////////// descriptor value //////////////////////////
#define GDMA_VALUE_TYPE_TENSOR     0
#define GDMA_VALUE_TYPE_MATRIX     1
#define GDMA_VALUE_TYPE_MASKED_SEL 2
#define GDMA_VALUE_TYPE_GENERAL    3
#define GDMA_VALUE_TYPE_CW_TRANS   4
#define GDMA_VALUE_TYPE_NONZERO    5
#define GDMA_VALUE_TYPE_SYS        6
#define GDMA_VALUE_TYPE_GATHER     7
#define GDMA_VALUE_TYPE_SCATTER    8
#define GDMA_VALUE_TYPE_NUM        9

#define GDMA_VALUE_DIR_S2L  0
#define GDMA_VALUE_DIR_L2S  1
#define GDMA_VALUE_DIR_S2S  2
#define GDMA_VALUE_DIR_L2L  3
#define GDMA_VAULE_DIR_NUM  4

#define GDMA_VALUE_FUNC_NONE       0
#define GDMA_VALUE_FUNC_TRANS      1  // NC Transpose or Matrix Transpose
#define GDMA_VALUE_FUNC_COLLECT    2  // CW Transpose from lmem to gmem
#define GDMA_VALUE_FUNC_BROADCAST  3
#define GDMA_VALUE_FUNC_DISTRIBUTE 4  // CW Transpose from gmem to lmem
#define GDMA_VALUE_FUNC_4BANK_COPY 5
#define GDMA_VALUE_FUNC_4BANK_BDC  6

#define GDMA_VALUE_FORMAT_INT8     0
#define GDMA_VALUE_FORMAT_FLOAT16  1
#define GDMA_VALUE_FORMAT_FLOAT32  2
#define GDMA_VALUE_FORMAT_INT16    3
#define GDMA_VALUE_FORMAT_INT32    4
#define GDMA_VALUE_FORMAT_BFLOAT16 5
#define GDMA_VALUE_FORMAT_NUM      6

#define SRC_IS_LOCAL(direction) \
    ((direction) == GDMA_VALUE_DIR_L2L || (direction) == GDMA_VALUE_DIR_L2S)
#define DST_IS_LOCAL(direction) \
    ((direction) == GDMA_VALUE_DIR_S2L || (direction) == GDMA_VALUE_DIR_L2L)

#define FORMAT_IS_FLOAT(format) \
    ((format) == GDMA_VALUE_FORMAT_FLOAT32 || (format) == GDMA_VALUE_FORMAT_FLOAT16 || (format) == GDMA_VALUE_FORMAT_BFLOAT16)

static inline int get_type_len(int t) {
  switch (t) {
    case GDMA_VALUE_FORMAT_INT8:
      return 1;
    case GDMA_VALUE_FORMAT_FLOAT16:
    case GDMA_VALUE_FORMAT_BFLOAT16:
    case GDMA_VALUE_FORMAT_INT16:
      return 2;
    case GDMA_VALUE_FORMAT_FLOAT32:
    case GDMA_VALUE_FORMAT_INT32:
      return 4;
  }
  return 0;
}

#endif  // __GDMA_REG_VALUE_H__
