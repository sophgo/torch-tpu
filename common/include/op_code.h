#ifndef OP_CODE_H_
#define OP_CODE_H_

/*
 * The value of enum is according to chip registers
 * So NOT change the value at will !
 */

typedef enum {
    CONV = 0,
    PD   = 1,
    MM   = 2,
    AR   = 3,
    RQDQ = 4,
    TRANS_BC = 5,
    SG   = 6,
    LAR  = 7,
    SFU  = 9,
    LIN  = 10,
    CMP  = 13,
    VC   = 14,
    SYS  = 15
} TSK_TYPE;

typedef enum {
    PAD_CONSTANT    = 0,
    PAD_REFLECTION  = 1,
    PAD_REPLICATION = 2,
    PAD_CIRCULAR    = 3
} PAD_MODE;

typedef enum {
    LIN_MAC = 1,
    LIN_ADD_SQR = 20,
    LIN_SUB_SQR = 21
} LIN_OP;

typedef enum {
    SFU_TAYLOR_4X = 12,
    SFU_TAYLOR    = 13,
    SFU_NORM      = 15,
    SFU_RSQ       = 17
} SFU_OP;

typedef enum {
    CMP_GT_AND_SG = 22,
    CMP_SG = 23,
    CMP_SE = 24,
    CMP_LT_AND_SL = 25,
    CMP_SL = 26
} CMP_OP;

typedef enum {
    MM_NORMAL = 1,
    MM_WRQ = 2,
    MM_WRQ_RELU = 3,
    MM_NN = 4,
    MM_NT = 5,
    MM_TT = 6,
} MM_OP;

typedef enum {
    AR_MUL = 0,
    AR_NOT = 1,
    AR_ADD = 2,
    AR_SUB = 3,
    AR_MAX = 4,
    AR_MIN = 5,
    AR_LOGIC_SHIFT = 6,
    AR_AND = 7,
    AR_OR = 8,
    AR_XOR = 9,
    AR_SG = 10,
    AR_SE = 11,
    AR_DIV = 12,
    AR_SL = 13,
    AR_DATA_CONVERT = 14,
    AR_ADD_SATU = 15,
    AR_SUB_SATU = 16,
    AR_CLAMP = 17,
    AR_MAC = 18,
    AR_COPY = 19,
    AR_MUL_SATU = 20,
    AR_ARITH_SHIFT = 21,
    AR_ROTATE_SHIFT = 22,
    AR_MULDHR = 23,
    AR_EU_IDX_GEN = 24,
    AR_NPU_IDX_GEN = 25,
    AR_ABS = 26,
    AR_FSUBABS = 27,
    AR_COPY_MB = 28,
    AR_GET_FIRST_ONE = 29,
    AR_GET_FIRST_ZERO = 30
} AR_OP;

typedef enum {
    PD_DEPTHWISE = 0,
    PD_AVG = 1,
    PD_DEPTHWISE_RELU = 2,
    PD_MAX = 4,
    PD_ROI_DEPTHWISE = 5,
    PD_ROI_AVG = 6,
    PD_ROI_MAX = 7
} PD_OP;

typedef enum {
    LANE_COPY = 2,
    LANE_BROAD = 3,
    STATIC_BROAD = 4,
    STATIC_DISTRIBUTE = 5,
} BC_OP;

typedef enum {
    TRAN_C_W_TRANSPOSE = 0,
    TRAN_W_C_TRANSPOSE = 1,
} TRAN_OP;

typedef enum {
    PL_gather_d1coor = 0,
    PL_gather_d2coor = 1,
    PL_gather_rec = 2,
    PL_scatter_d1coor = 3,
    PL_scatter_d2coor = 4,
    PE_S_gather_d1coor = 5,
    PE_S_scatter_d1coor = 6,
    PE_M_gather_d1coor = 7,
    PE_S_mask_select = 8,
    PE_S_nonzero = 9,
    PE_S_scatter_pp_d1coor = 10,
    PE_S_gather_hzd = 13,
    PE_S_scatter_hzd = 14,
    PE_S_mask_selhzd = 15,
    PE_S_nonzero_hzd = 16,
    PE_S_gather_line = 17,
    PE_S_scatter_line = 18,
    PE_S_mask_seline = 19,
} SG_OP;

typedef enum {
    RQ_0 = 0,
    RQ_1 = 1,
    DQ_0 = 3,
    DQ_1 = 4,
} RQDQ_OP;

typedef enum {
    INSTR_BARRIER = 0, // no use
    SPB = 1,
    SWR = 2,
    SWR_FROM_LMEM = 3,
    SWR_COL_FROM_LMEM = 4,
    SYNC_ID = 5,
    DATA_BARRIER = 6, // no use
    SYS_END = 31
} SYS_TYPE;

#define BM_REDUCE_MEAN 0
#define BM_REDUCE_SUM  1
#define BM_REDUCE_MAX  2
#define BM_REDUCE_MIN  3
#define BM_REDUCE_PROD 4
#define BM_REDUCE_ALL  5
#define BM_REDUCE_ANY  6
#define BM_REDUCE_L2   7

#define BM_BINARY_ADD 0
#define BM_BINARY_SUB 1
#define BM_BINARY_MUL 2
#define BM_BINARY_DIV 3
#define BM_BINARY_MAX 4
#define BM_BINARY_MIN 10000

#define BM_BINARY_GT 10001
#define BM_BINARY_GE 10002
#define BM_BINARY_LT 10003
#define BM_BINARY_LE 10004
#define BM_BINARY_EQ 10005
#define BM_BINARY_NE 10006
#define BM_BINARY_SQUARED_DIFF 10007
#define BM_BINARY_FLOOR_MOD 10008
#define BM_BINARY_FLOOR_DIV 10009

#define TENSOR_N_DIM 0
#define TENSOR_C_DIM 1
#define TENSOR_H_DIM 2
#define TENSOR_W_DIM 3

#endif /* OP_CODE_H_ */
