#ifndef SGDNN_API_H
#define SGDNN_API_H

#include "bmlib_runtime.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct
{
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;
}
SgdnnConv2dParam_t;

typedef enum
{
  SGDNN_DTYPE_FP32 = 0,
  SGDNN_DTYPE_FP16 = 1,
  SGDNN_DTYPE_INT8 = 2,
  SGDNN_DTYPE_UINT8 = 3,
  SGDNN_DTYPE_INT16 = 4,
  SGDNN_DTYPE_UINT16 = 5,
  SGDNN_DTYPE_INT32 = 6,
  SGDNN_DTYPE_UINT32 = 7,
  SGDNN_DTYPE_BF16 = 8, // BFP vs BF
  SGDNN_DTYPE_INT4 = 9,
  SGDNN_DTYPE_UINT4 = 10,
  SGDNN_DTYPE_FP20 = 11,
  SGDNN_DTYPE_INT64 = 12,
  SGDNN_DTYPE_UNKNOWN = -1,
}
SgdnnDataType_t;

typedef struct
{
  unsigned long long addr;
  int dim;
  int shape[8];
  int stride[8];
  SgdnnDataType_t dtype;
}
SgdnnTensor_t;

static inline SgdnnTensor_t sgdnnUndefinedTensor()
{
  SgdnnTensor_t tensor = { .addr = 0 };
  return tensor;
}

bm_status_t sgdnnInitialize ( bm_handle_t handle );

bm_status_t sgdnnDeinitialize ( bm_handle_t handle );

bm_status_t sgdnnReorderConv2dWeight ( bm_handle_t handle,
                                       SgdnnTensor_t input,
                                       int mode,
                                       SgdnnTensor_t output );

/*
 * OUTPUT = CONV2D ( INPUT, WEIGHT, BIAS )
 * Note:
 * 1. The data types of INPUT, WEIGHT, BIAS and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT, WEIGHT and OUTPUT must be 4, BIAS must be 1
 * 3. The shape of INPUT is ( N, IC, IH, IW ), WEIGHT is ( OC, IC, KH, KW ), BIAS is ( OC ), OUTPUT is ( N, OC, OH, OW )
 * 4. INPUT, WEIGHT, BIAS and OUTPUT must be contiguous
 * 5. BIAS is optional
 */
bm_status_t sgdnnConv2d ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          SgdnnTensor_t weight,
                          SgdnnTensor_t bias,
                          SgdnnConv2dParam_t param,
                          SgdnnTensor_t output );

/*
 * [ GRAD_INPUT, GRAD_WEIGHT, GRAD_BIAS ] = CONV2D BACKWARD ( GRAD_OUTPUT, INPUT, WEIGHT )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT, GRAD_OUTPUT, WEIGHT, GRAD_WEIGHT, OUTPUT and GRAD_OUTPUT must be 4, GRAD_BIAS must be 1
 * 3. The shape of INPUT and GRAD_INPUT is ( N, IC, IH, IW ), WEIGHT and GRAD_WEIGHT is ( OC, IC, KH, KW ), GRAD_BIAS is ( OC ), GRAD_OUTPUT is ( N, OC, OH, OW )
 * 4. All the tensors must be contiguous
 * 5. GRAD_INPUT, GRAD_WEIGHT, GRAD_BIAS are optional
 */
bm_status_t sgdnnConv2dBackward ( bm_handle_t handle,
                                  SgdnnTensor_t grad_output,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t weight,
                                  SgdnnConv2dParam_t param,
                                  SgdnnTensor_t grad_input,
                                  SgdnnTensor_t grad_weight,
                                  SgdnnTensor_t grad_bias );

/*
 * [ OUTPUT, SAVED_MEAN, SAVED_INVSTD, RUNNING_MEAN, RUNNING_VAR ] = BATCHNORM2D ( INPUT, WEIGHT, BIAS, EPS, RUNNING_MEAN, RUNNING_VAR, MOMENTUM )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT and OUTPUT must be the same and be 3 or 4
 * 3. The dimensions of WEIGHT, BIAS, RUNNING_MEAN, RUNNING_VAR, SAVED_MEAN and SAVED_INVSTD must be 1
 * 4. The shape of INPUT is ( N, C, H, W ) or ( N, C, H ), OUTPUT is ( N, C, H, W ) or ( N, C, H ) and the other tensor, such as WEIGHT, is ( C )
 * 5. All the tensors must be contiguous
 * 6. WEIGHT, BIAS, RUNNING_MEAN, RUNNING_VAR, SAVED_MEAN and SAVED_INVSTD are optional
 */
bm_status_t sgdnnBatchnorm2d ( bm_handle_t handle,
                               SgdnnTensor_t input,
                               SgdnnTensor_t weight,
                               SgdnnTensor_t bias,
                               float eps,
                               SgdnnTensor_t running_mean,
                               SgdnnTensor_t running_var,
                               float momentum,
                               SgdnnTensor_t output,
                               SgdnnTensor_t saved_mean,
                               SgdnnTensor_t saved_invstd );

/*
 * [ GRAD_INPUT, GRAD_WEIGHT, GRAD_BIAS ] = BATCHNORM2D BACKWARD ( GRAD_OUTPUT, INPUT, WEIGHT, SAVED_MEAN, SAVED_INVSTD )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of GRAD_OUTPUT and INTPUT must be the same and be 3 or 4
 * 3. The dimensions of WEIGHT, SAVED_MEAN and SAVED_INVSTD must be 1
 * 4. The shape of INPUT is ( N, C, H, W ) or ( N, C, H ), GRAD_OUTPUT is ( N, C, H, W ) or ( N, C, H ) and the other tensor, such as WEIGHT, is ( C )
 * 5. All the tensors must be contiguous
 * 6. WEIGHT, GRAD_INPUT, GRAD_WEIGHT and GRAD_BIAS are optional
 */
bm_status_t sgdnnBatchnorm2dBackward ( bm_handle_t handle,
                                       SgdnnTensor_t grad_output,
                                       SgdnnTensor_t input,
                                       SgdnnTensor_t weight,
                                       SgdnnTensor_t saved_mean,
                                       SgdnnTensor_t saved_invstd,
                                       SgdnnTensor_t grad_input,
                                       SgdnnTensor_t grad_weight,
                                       SgdnnTensor_t grad_bias );

/*
 * [ OUTPUT, MEAN, RSTD ] = LAYERNORM ( INPUT, WEIGHT, BIAS, START_DIM, EPS )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 4. The shape of INPUT is ( D0, D1, ..., D(A-1), DA, D(A+1), ... ), where DA is the START_DIM, then WEIGHT and BIAS are ( D(A), D(A+1), ... ), MEAN and RSTD are ( D0, D1, ..., D(A-1), 1, 1, ... )
 * 5. All the tensors must be contiguous
 * 6. WEIGHT and BIAS are optional
 */
bm_status_t sgdnnLayernorm ( bm_handle_t handle,
                             SgdnnTensor_t input,
                             SgdnnTensor_t weight,
                             SgdnnTensor_t bias,
                             int start_dim,
                             float eps,
                             SgdnnTensor_t output,
                             SgdnnTensor_t mean,
                             SgdnnTensor_t rstd );

/*
 * [ GRAD_INPUT, GRAD_WEIGHT, GRAD_BIAS ] = LAYERNORM BACKWARD ( GRAD_OUTPUT, INPUT, WEIGHT, MEAN, RSTD, START_DIM )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, GRAD_OUTPUT and GRAD_INPUT must be the same
 * 4. The shape of INPUT is ( D0, D1, ..., D(A-1), DA, D(A+1), ... ), where DA is the START_DIM, then WEIGHT, GRAD_WEIGHT and GRAD_BIAS are ( D(A), D(A+1), ... ), MEAN and RSTD are ( D0, D1, ..., D(A-1), 1, 1, ... )
 * 5. All the tensors must be contiguous
 * 6. WEIGHT, GRAD_INPUT, GRAD_WEIGHT and GRAD_BIAS are optional
 */
bm_status_t sgdnnLayernormBackward ( bm_handle_t handle,
                                     SgdnnTensor_t grad_output,
                                     SgdnnTensor_t input,
                                     SgdnnTensor_t weight,
                                     SgdnnTensor_t mean,
                                     SgdnnTensor_t rstd,
                                     int start_dim,
                                     SgdnnTensor_t grad_input,
                                     SgdnnTensor_t grad_weight,
                                     SgdnnTensor_t grad_bias );

/*
 * OUTPUT = LEFT * RIGHT + BIAS, where " * " is matrix multiplication
 * Note:
 * 1. The data types of LEFT, RIGHT, BIAS and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of LEFT, RIGHT, BIAS and OUTPUT must be 2
 * 3. The shape of LEFT is ( M, K ), RIGHT is ( K, N ), BIAS is ( N ), OUTPUT is ( M, N )
 * 4. LEFT is allowed to be contiguous or transposed, meaning that the stride is ( K, 1 ) or ( 1, M )
 *    RIGHT is allowed to be contiguous or transposed, meaning that the stride is ( N, 1 ) or ( 1, K )
 *    OUTPUT is only allowed to be contiguous, meaning that the stride is ( N, 1 )
 *    BIAS must be contiguous
 * 5. BIAS is optional
 */
bm_status_t sgdnnMatmul ( bm_handle_t handle,
                          SgdnnTensor_t left,
                          SgdnnTensor_t right,
                          SgdnnTensor_t bias,
                          SgdnnTensor_t output );

/*
 * OUTPUT ( batch ) = LEFT ( batch ) * RIGHT ( batch ), where " * " is matrix multiplication
 * Note:
 * 1. The data types of LEFT, RIGHT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of LEFT, RIGHT and OUTPUT must be 3
 * 3. The shape of LEFT is ( B, M, K ), RIGHT is ( B, K, N ), OUTPUT is ( B, M, N )
 * 4. LEFT is allowed to be contiguous or transposed, meaning that the stride is ( M * K, K, 1 ) or ( M * K, 1, M )
 *    RIGHT is allowed to be contiguous or transposed, meaning that the stride is ( N * K, N, 1 ) or ( N * K, 1, K )
 *    OUTPUT is only allowed to be contiguous, meaning that the stride is ( M * N, N, 1 )
 */
bm_status_t sgdnnBatchMatmul ( bm_handle_t handle,
                               SgdnnTensor_t left,
                               SgdnnTensor_t right,
                               SgdnnTensor_t output );

/*
 * OUTPUT = INPUT + SCALAR * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnAdd ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       float scalar,
                       SgdnnTensor_t output );

/*
 * OUTPUT = INPUT - SCALAR * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnSub ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       float scalar,
                       SgdnnTensor_t output );

/*
 * OUTPUT = INPUT * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnMul ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output );

/*
 * OUTPUT = INPUT / OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnDiv ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output );

/*
 * OUTPUT = INPUT + SCALAR * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimentions of INPUT, OTHER and OUTPUT must be the same
 * 3. INPUT and OTHER are allowed broadcasting
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnAddBcast ( bm_handle_t handle,
                            SgdnnTensor_t input,
                            SgdnnTensor_t other,
                            float scalar,
                            SgdnnTensor_t output );

/*
 * OUTPUT = INPUT + SCALAR
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnAddC ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = SCALAR - INPUT
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnCSub ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = INPUT * SCALAR
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnMulC ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = SCALAR / INPUT
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnCDiv ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = INPUT + SCALAR * ( TENSOR1 * TENSOR2 )
 * Note:
 * 1. The data types of INPUT, TENSOR1, TENSOR2 and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, TENSOR1, TENSOR2 and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, TENSOR1, TENSOR2 and OUTPUT must be contiguous
 */
bm_status_t sgdnnAddCMul ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           SgdnnTensor_t tensor1,
                           SgdnnTensor_t tensor2,
                           float scalar,
                           SgdnnTensor_t output );

/*
 * OUTPUT = INPUT + SCALAR * ( TENSOR1 / TENSOR2 )
 * Note:
 * 1. The data types of INPUT, TENSOR1, TENSOR2 and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, TENSOR1, TENSOR2 and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, TENSOR1, TENSOR2 and OUTPUT must be contiguous
 */
bm_status_t sgdnnAddCDiv ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           SgdnnTensor_t tensor1,
                           SgdnnTensor_t tensor2,
                           float scalar,
                           SgdnnTensor_t output );

/*
 * OUTPUT = ReLU ( INPUT )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnReLU ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output );

/*
 * GRAD_INPUT = ReLU Backward ( GRAD_OUTPUT, INPUT )
 * Note:
 * 1. The data types of GRAD_OUTPUT, INPUT and GRAD_INPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of GRAD_OUTPUT, INPUT and GRAD_INPUT must be the same
 * 3. GRAD_OUTPUT, INPUT and GRAD_INPUT must be contiguous
 */
bm_status_t sgdnnReLUBackward ( bm_handle_t handle,
                                SgdnnTensor_t grad_output,
                                SgdnnTensor_t input,
                                SgdnnTensor_t grad_input );

/*
 * OUTPUT = GELU ( INPUT )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnGELU ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output );

/*
 * GRAD_INPUT = GELU BACKWARD ( GRAD_OUTPUT, INPUT )
 * Note:
 * 1. The data types of GRAD_OUTPUT, INPUT and GRAD_INPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of GRAD_OUTPUT, INPUT and GRAD_INPUT must be the same
 * 3. GRAD_OUTPUT, INPUT and GRAD_INPUT must be contiguous
 */
bm_status_t sgdnnGELUBackward ( bm_handle_t handle,
                                SgdnnTensor_t grad_output,
                                SgdnnTensor_t input,
                                SgdnnTensor_t grad_input );

/*
 * OUTPUT = COPY ( INPUT )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT are allowed to be NOT contiguous
 */
bm_status_t sgdnnStridedCopy ( bm_handle_t handle,
                               SgdnnTensor_t input,
                               SgdnnTensor_t output );

/*
 * OUTPUT = CONVERT ( INPUT )
 * Note:
 * 1. The shapes of INPUT and OUTPUT must be the same
 * 2. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnConvert ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           SgdnnTensor_t output );

/*
 * OUTPUT = SQRT ( INPUT )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnSqrt ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output );

/*
 * OUTPUT = SOFTMAX ( INPUT, DIM )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnSoftmax ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           int dim,
                           SgdnnTensor_t output );

/*
 * GRAD_INPUT = SOFTMAX BACKWARD ( GRAD_OUTPUT, OUTPUT, DIM )
 * Note:
 * 1. The data types of GRAD_OUTPUT, OUTPUT and GRAD_INPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of GRAD_OUTPUT, OUTPUT and GRAD_INPUT must be the same
 * 3. GRAD_OUTPUT, OUTPUT and GRAD_INPUT must be contiguous
 */
bm_status_t sgdnnSoftmaxBackward ( bm_handle_t handle,
                                   SgdnnTensor_t grad_output,
                                   SgdnnTensor_t output,
                                   int dim,
                                   SgdnnTensor_t grad_input );

/*
 * OUTPUT = NORM2 ( INPUT, KEEPDIM )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. If keepdim is TRUE, the dimensions of INPUT and OUTPUT must be the same, the shape of OUTPUT must be ( 1, 1, ... )
 * 3. If keepdim is FALSE, the dimension of OUTPUT must be zero
 * 4. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnNorm2 ( bm_handle_t handle,
                         SgdnnTensor_t input,
                         int keepdim,
                         SgdnnTensor_t output );

/*
 * OUTPUT = CROSS ENTROPY LOSS ( INPUT, TARGET, REDUCTION, LABEL_SMOOTHING )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The data type of TARGET must be INT32 or INT64, and will be used as INT32
 * 3. The dimension of INPUT must be 2, TARGET must be 1 and OUTPUT must be 0
 * 4. The shape of INPUT is ( B, C ), TARGET is ( B )
 * 5. INPUT, TARGET and OUTPUT must be contiguous
 * 6. REDUCTION must be 0 ( mean ) or 1 ( sum )
 */
bm_status_t sgdnnCrossEntropyLoss ( bm_handle_t handle,
                                    SgdnnTensor_t input,
                                    SgdnnTensor_t target,
                                    int reduction,
                                    float label_smoothing,
                                    SgdnnTensor_t output );

/*
 * GRAD_INPUT = CROSS ENTROPY LOSS BACKWARD ( INPUT, TARGET, GRAD_OUTPUT, REDUCTION, LABEL_SMOOTHING )
 * Note:
 * 1. The data types of INPUT, GRAD_OUTPUT and GRAD_INPUT must be the same and one of FP32, FP16 and BF16
 * 2. The data type of TARGET must be INT32 or INT64, and will be used as INT32
 * 3. The dimensions of INPUT and GRAD_INPUT must be 2, TARGET must be 1 and GRAD_OUTPUT must be 0
 * 4. The shape of INPUT is ( B, C ), GRAD_INPUT is ( B, C ), TARGET is ( B )
 * 5. INPUT, TARGET, GRAD_OUTPUT and GRAD_INPUT must be contiguous
 * 6. REDUCTION must be 0 ( mean ) or 1 ( sum )
 */
bm_status_t sgdnnCrossEntropyLossBackward (
bm_handle_t handle,
SgdnnTensor_t input,
SgdnnTensor_t target,
SgdnnTensor_t grad_output,
int reduction,
float label_smoothing,
SgdnnTensor_t grad_input );

/*
 * OUTPUT = FILL ( SCALAR ), the elements of OUTPUT are all set SCALAR
 * Note:
 * 1. The data types of SCALAR and OUTPUT must be the same
 * 2. SCALAR_PTR is a host pointer
 * 3. OUTPUT must be contiguous
 */
bm_status_t sgdnnFill ( bm_handle_t handle,
                        const void * scalar_ptr,
                        SgdnnTensor_t output );

/*
 * OUTPUT = REDUCE ( INPUT, START_DIM, END_DIM, KEEPDIM, MODE )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shape of INPUT is ( D0, D1, ..., D(S-1), DS, ..., D(E-1), DE, D(E+1), ... ), where DS is START_DIM and DE is END_DIM,
 *    if keepdim is TRUE, OUTPUT is ( D0, D1, ..., D(S-1), 1, ..., 1, DE, D(E+1), ... ), otherwise, ( D0, D1, ..., D(S-1), DE, D(E+1), ... )
 * 3. INPUT and OUTPUT must be contiguous
 * 4. MODE must be 0 ( mean ) or 1 ( sum )
 */
bm_status_t sgdnnReduce ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          int start_dim,
                          int end_dim,
                          int keepdim,
                          int mode,
                          SgdnnTensor_t output );

bm_status_t sgdnnConcat ( bm_handle_t handle,
                          const SgdnnTensor_t * inputs,
                          int input_num,
                          int dim,
                          SgdnnTensor_t output );

/*
 * OUTPUT = WHERE ( COND, SELF, OTHER ) = COND ? SELF : OTHER
 * Note:
 * 1. The data types of SELF, OTHER and OUTPUT must be the same
 * 2. The dimension of COND and OUTPUT must be the same
 * 3. COND, SELF, OTHER and OUTPUT must be contiguous
 * 4. COND, SELF and OTHER are allowed broadcastable, SELF and OTHER are allowed scalar
 */
bm_status_t sgdnnWhere ( bm_handle_t handle,
                         SgdnnTensor_t cond,
                         SgdnnTensor_t self,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output );

/*
 * OUTPUT = INDEX SELECT ( INPUT, INDICES, DIM )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same, INDICES must be INT32 or INT64, but used as INT32
 * 2. The shape of INPUT is ( D0, ..., D(D-1), DD, D(D+1), ... ), where DD is the DIM, INDICES is ( I0, ..., IX ),
 *    then OUTPUT is ( D0, ..., D(D-1), I0, ..., IX, D(D+1), ...  )
 * 3. INPUT, INDICES and OUTPUT must be contiguous
 */
bm_status_t sgdnnIndexSelect ( bm_handle_t handle,
                               SgdnnTensor_t input,
                               SgdnnTensor_t indices,
                               int dim,
                               SgdnnTensor_t output );

bm_status_t sgdnnEmbeddingBackward ( bm_handle_t handle,
                                     SgdnnTensor_t grad_output,
                                     SgdnnTensor_t indices,
                                     SgdnnTensor_t grad_input );

#if defined(__cplusplus)
}
#endif

#endif /* SGDNN_API_H */

