#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SG_API_ID_RESERVED                         = 0,
    /*
     * API ID that is used in bmlib
     */
    SG_API_ID_MEM_SET                          = 1,
    SG_API_ID_MEM_CPY                          = 2,

    /*
     * Layer APIs
     */
    SG_API_ID_POOLING_PARALLEL                 = 3,
    SG_API_ID_ARITHMETIC                       = 4,
    SG_API_ID_SORT                             = 5,
    SG_API_ID_SCALE                            = 6,
    SG_API_ID_FC                               = 7,
    SG_API_ID_BATCH_MATMUL                     = 8,
    SG_API_ID_POOLING_FIX8B                    = 9,
    SG_API_ID_BNSCALE_FIX8B                    = 10,
    SG_API_ID_ELTWISE                          = 11,
    SG_API_ID_ELTWISE_FIX8B                    = 12,
    SG_API_ID_PRELU                            = 13,
    SG_API_ID_PRELU_FIX8B                      = 14,
    SG_API_ID_RELU                             = 15,
    SG_API_ID_DEPTHWISE_NORMAL                 = 16,
    SG_API_ID_CONV_QUANT_SYM                   = 17,
    SG_API_ID_CONV_FLOAT                       = 18,
    SG_API_ID_DEPTHWISE_FIX8B_NORMAL           = 19,
    SG_API_ID_SOFTMAX                          = 20,
    SG_API_ID_FC_FIX8B                         = 21,
    SG_API_ID_SOFTMAX_TFLITE_FIX8B             = 22,
    SG_API_ID_BATCH_MATMUL_FIX8B               = 23,
    SG_API_ID_REDUCE                           = 24,
    SG_API_ID_BCBINARY_FIX8B                   = 25,
    SG_API_ID_BN_FWD_INF_PARALLEL              = 26,
    SG_API_ID_CONST_BINARY_FIX8B               = 27,
    SG_API_ID_REDUCE_FIX8B                     = 28,
    SG_API_ID_INDEX_SELECT                     = 29,
    SG_API_ID_ACTIVE                           = 30,
    SG_API_ID_TRANSPOSE                        = 31,
    SG_API_ID_CONCAT                           = 32,
    SG_API_ID_UPSAMPLE                         = 33,
    SG_API_ID_DEPTH2SPACE                      = 34,
    SG_API_ID_REQUANT_INT                      = 35,
    SG_API_ID_REQUANT_FLOAT                    = 36,
    SG_API_ID_TOPK                             = 37,
    SG_API_ID_INTERP_PARALLEL                  = 38,
    SG_API_ID_GEMM                             = 39,
    SG_API_ID_DEQUANT_INT                      = 40,
    SG_API_ID_DEQUANT_FLOAT                    = 41,
    SG_API_ID_ROUND_FP                         = 42,
    SG_API_ID_ARG                              = 43,
    SG_API_ID_GATHER_ND_TF                     = 44,
    SG_API_ID_YOLO                             = 45,
    SG_API_ID_SORT_PER_DIM                     = 46,
    SG_API_ID_CONSTANT_FILL                    = 47,
    SG_API_ID_ROI_POOLING                      = 48,
    SG_API_ID_NMS                              = 49,
    SG_API_ID_SSD_DETECT_OUT                   = 50,
    SG_API_ID_ARITHMETIC_SHIFT                 = 51,
    SG_API_ID_ROI_ALIGN                        = 52,
    SG_API_ID_WHERE                            = 53,
    SG_API_ID_PSROI_POOLING                    = 54,
    SG_API_ID_COLUMN_HASH                      = 55,
    SG_API_ID_SHIFT                            = 56,
    SG_API_ID_TILE                             = 57,
    SG_API_ID_SPLIT                            = 58,
    SG_API_ID_YOLOV3_DETECT_OUT                = 59,
    SG_API_ID_MASKED_SELECT                    = 60,
    SG_API_ID_UPSAMPLE_MASK                    = 61,
    SG_API_ID_BINARY_SHIFT                     = 62,
    SG_API_ID_PRIOR_BOX                        = 63,
    SG_API_ID_QUANT_DIV                        = 64,
    SG_API_ID_STRIDESLICE                      = 65,
    SG_API_ID_MASKED_FILL                      = 66,
    SG_API_ID_BATCH2SPACE                      = 67,
    SG_API_ID_GRID_SAMPLE                      = 68,
    SG_API_ID_SHUFFLE_CHANNEL                  = 69,
    SG_API_ID_ADAPTIVE_POOL                    = 70,
    SG_API_ID_BCBINARY_FLOAT                   = 71,
    SG_API_ID_EMBEDDING_BAG                    = 72,


    /*
     * API ID that is used in bmlib
     */
    SG_API_ID_MEMCPY_BYTE                      = 136,
    SG_API_ID_MEMCPY_WSTRIDE                   = 137,

    // for PROFILE, same as BM1684
    SG_API_ID_SET_PROFILE_ENABLE               = 986,
    SG_API_ID_GET_PROFILE_DATA                 = 987,

    //  ID for bringup of asic
    SG_API_ID_BRINGUP                          = 0x0ffffffa,
    // ID for fullnet
    SG_API_ID_MULTI_FULLNET                    = 0x0ffffffb,
    SG_API_ID_DYNAMIC_FULLNET                  = 0x0ffffffc,
    // ID for Palladium Test
    SG_API_PLD_TEST                            = 0x0ffffffd,
    // ID for TPUKernel
    SG_API_TPUKERNEL                           = 0x0ffffffe,

    // ID for Test all instructions
    SG_API_TEST_ALL_STRUCTION_PERF                 = 0x0fffffff,
    // QUIT ID
} sg_api_id_t;

#define SG_API_QUIT 0xffffffff


#ifdef __cplusplus
}
#endif

#endif  /* _MESSAGE_H_ */
