#ifndef SGDNN_API_H
#define SGDNN_API_H

#include "bmlib_runtime.h"
#include "sg_api_struct.h"
#include <map>
#include <vector>

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
  SGDNN_DTYPE_UNKNOWN = 0,
  SGDNN_DTYPE_INT8,
  SGDNN_DTYPE_UINT8,
  SGDNN_DTYPE_INT16,
  SGDNN_DTYPE_UINT16,
  SGDNN_DTYPE_FP16,
  SGDNN_DTYPE_BF16,
  SGDNN_DTYPE_INT32,
  SGDNN_DTYPE_UINT32,
  SGDNN_DTYPE_FP32,
  SGDNN_DTYPE_INT64
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
 * OUTPUT = POOLING ( INPUT, POOLING_DESC )
 * Note:
 *
 */
bm_status_t sgdnnPoolingForward ( bm_handle_t handle,
                               SgdnnTensor_t input,
                               SgdnnTensor_t output,
                               PoolingDescriptor_t pooling_desc);

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
 * OUTPUT = GATHER ( INPUT, AXIS, INDEX )
 * Note:
 * 1. Input and index must have the same number of dimensions
 * 2. index.size(d) <= input.size(d) for all dimensions d != dim
 * 3. Output have the same shape as index, and do not broadcast against each other
 */

bm_status_t sgdnnGather ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          SgdnnTensor_t index,
                          SgdnnTensor_t output,
                          int axis );
/*
 * OUTPUT = Active(INPUT)
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
* 2. INPUT must be contiguous
 */
bm_status_t sgdnnActive(bm_handle_t handle, SgdnnTensor_t input,
                        SgdnnTensor_t output, sg_active_type_t active_type);
/*
 * OUTPUT = log(INPUT)
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * 2. INPUT must be contiguous
 */
bm_status_t sgdnnLog ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output,
                       sg_log_type_t log_type);

/*
 * OUTPUT = squeeze(INPUT)
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * 2. INPUT must be contiguous
 */
bm_status_t sgdnnSqueeze ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output);

/*
 * OUTPUT = native_group_norm(INPUT)
 * Note:
 */
bm_status_t sgdnnNativeGroupNorm ( bm_handle_t handle, SgdnnTensor_t input,
                     SgdnnTensor_t gamma, SgdnnTensor_t beta,
                     int group, int affine, float eps, SgdnnTensor_t output,
                     SgdnnTensor_t mean, SgdnnTensor_t rstd);

bm_status_t sgdnnNativeGroupNormBackward(bm_handle_t handle, SgdnnTensor_t grad_output,
                     SgdnnTensor_t input, SgdnnTensor_t weight,
                     SgdnnTensor_t mean, SgdnnTensor_t rstd,
                     int group, SgdnnTensor_t out0,
                     SgdnnTensor_t out1, SgdnnTensor_t out2);

/*
 * OUTPUT = INPUT || OTHER
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * INPUT and OTHER must be contiguous
 */

bm_status_t sgdnnLogicalOr ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output );

/*
 *  OUTPUT  = SHIFT_LEFT ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 */
bm_status_t sgdnnShiftLeft ( bm_handle_t handle,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output );
/*
 *  OUTPUT  = SHIFT_LEFT ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 */
bm_status_t sgdnnShiftLeftBcast ( bm_handle_t handle,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output );
/*
 *  OUTPUT  = SHIFT_LEFT ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. one of input and other must be a scalar.
 *  3. input and output must be in the same shape.
 */
bm_status_t sgdnnShiftLeftC ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          char scalar,
                          SgdnnTensor_t output );

/*
 *  OUTPUT  = SHIFT_RIGHT_ARITHMETIC ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 */
bm_status_t sgdnnShiftRightArithmetic ( bm_handle_t handle,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output );
/*
 *  OUTPUT  = SHIFT_RIGHT_ARITHMETIC ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 */
bm_status_t sgdnnShiftRightArithmeticBcast ( bm_handle_t handle,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output );
/*
 *  OUTPUT  = SHIFT_RIGHT_ARITHMETIC ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. one of input and other must be a scalar.
 *  3. input and output must be in the same shape.
 */
bm_status_t sgdnnShiftRightArithmeticC ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          int scalar,
                          SgdnnTensor_t output );

/*
 * OUTPUT = logical_not(INPUT)
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * 2. INPUT must be contiguous
 */

bm_status_t sgdnnLogicalNot ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output );

/*
 * OUTPUT = math.flip(INPUT, DIMS)
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * 2. INPUT must be contiguous
 */

bm_status_t sgdnnFlip ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        int axis,
                        SgdnnTensor_t output );
/*
 * OUTPUT = INPUT && OTHER
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * INPUT and OTHER must be contiguous
 */

bm_status_t sgdnnLogicalAnd ( bm_handle_t handle,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output );
/*
 * OUTPUT = math.pow (INPUT, OTHER)
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnPow ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
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
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */

bm_status_t sgdnnBinary ( bm_handle_t handle,
                            SgdnnTensor_t input,
                            SgdnnTensor_t other,
                            float scalar,
                            SgdnnTensor_t output,
                            int binary_type );

/*
 * OUTPUT = INPUT + SCALAR * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimentions of INPUT, OTHER and OUTPUT must be the same
 * 3. INPUT and OTHER are allowed broadcasting
 * 4. INPUT, OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnBinaryBcast (  bm_handle_t handle,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t other,
                                  float scalar,
                                  SgdnnTensor_t output,
                                  int binary_type );

/*
 * OUTPUT = INPUT + SCALAR
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnBinaryC (  bm_handle_t handle,
                              SgdnnTensor_t input,
                              float scalar,
                              SgdnnTensor_t output,
                              int binary_type,
                              bool inversed );

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
 * 2. Broadcasting is allowed
 * 3. INPUT, TENSOR1, TENSOR2 and OUTPUT must be contiguous
 */

bm_status_t sgdnnAddCMulBcast ( bm_handle_t handle,
                                SgdnnTensor_t input,
                                SgdnnTensor_t tensor1,
                                SgdnnTensor_t tensor2,
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
 * OUTPUT = LeakyRelu (INPUT)
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * 2. INPUT must be contiguous
 */

bm_status_t sgdnnLeakyReLU ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output,
                       float negative_slope );

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

 /*
 * OUTPUT = CLAMP ( INPUT, MIN, MAX )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnClamp ( bm_handle_t handle,
                         SgdnnTensor_t input,
                         float min,
                         float max,
                         SgdnnTensor_t output );

/*
 * OUTPUT = SIGN ( INPUT )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnSign ( bm_handle_t handle,
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

/*
 * OUTPUT = REDUCE_PROD ( INPUT, START_DIM, END_DIM, KEEPDIM, MODE )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shape of INPUT is ( D0, D1, ..., D(S-1), DS, ..., D(E-1), DE, D(E+1), ... ), where DS is START_DIM and DE is END_DIM,
 *    if keepdim is TRUE, OUTPUT is ( D0, D1, ..., D(S-1), 1, ..., 1, DE, D(E+1), ... ), otherwise, ( D0, D1, ..., D(S-1), DE, D(E+1), ... )
 * 3. INPUT and OUTPUT must be contiguous
 * 4. MODE must be 0 ( prod ) or 1 ( sum )
 */
bm_status_t sgdnnReduceProd ( bm_handle_t handle,
                              SgdnnTensor_t input,
                              int axis,
                              int keepdim,
                              SgdnnTensor_t output );

/*
 * OUTPUT = CONCAT ( INPUTS, DIM )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same
 * 2. The shape of INPUT0 is ( D0, D1, ..., D(A-1), DA0, D(A+1), ... ), INPUT1 is ( D0, D1, ..., D(A-1), DA1, D(A+1), ... ),
 *    where DA0, DA1, ... are the "DIM" dimension of the corresponding tensors, the shape of OUTPUT is ( D0, D1, ..., D(A-1), ( DA0 + DA1 + ... ), D(A+1), ... )
 * 3. INPUTS and OUTPUT must be contiguous
 */
bm_status_t sgdnnConcat ( bm_handle_t handle,
                          const SgdnnTensor_t * inputs,
                          int input_num,
                          int dim,
                          SgdnnTensor_t output );
/*
 * OUTPUT = Upsampling ( INPUTS, align_corners, output_size)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same
 * 2. The shape of INPUT0 is ( D0, D1, ..., D(A-1), DA0, D(A+1), ... ), INPUT1 is ( D0, D1, ..., D(A-1), DA1, D(A+1), ... ),
 *    where DA0, DA1, ... are the "DIM" dimension of the corresponding tensors, the shape of OUTPUT is ( D0, D1, ..., D(A-1), ( DA0 + DA1 + ... ), D(A+1), ... )
 * 3. INPUTS and OUTPUT must be contiguous
 */
bm_status_t sgdnnUpsampling(bm_handle_t handle, SgdnnTensor_t input,
                            SgdnnTensor_t output,
                            bool align_corners,
                            sg_resize_mode_t upsampling_type);

bm_status_t sgdnnUpsampleNearest2dBackward(bm_handle_t handle, SgdnnTensor_t grad_output,
                            SgdnnTensor_t grad_input, int scale,
                            PoolingDescriptor_t pooling_desc);

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
 *    then OUTPUT is ( D0, ..., D(D-1), [X+1], D(D+1), ...  )
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
/*
 * OUPUT = INPUT[indices]
 *   indices's size <= INPUT.dim
 *   OUTPUT[0] = INPUT[indices[0]] and so on ...
*/
bm_status_t sgdnnMulIndexSelect ( bm_handle_t handle,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t output,
                                  std::vector<SgdnnTensor_t>& indices);


/*
  OUTPUT = MASKE_FILL ( INPUT, MASK, VALUE )
*/
bm_status_t sgdnnMaskedFill ( bm_handle_t handle,
                              SgdnnTensor_t input,
                              SgdnnTensor_t mask,
                              float value,
                              SgdnnTensor_t out );

/*
 * [ OUTPUT, MASK ] = DROPOUT ( INPUT, SEED, THRESHOLD )
 * MASK = RANDOM ( SEED ) > THRESHOLD ? 1 : 0
 * OUTPUT = INPUT * MASK
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same, MASK must be UINT8
 * 2. The shapes of INPUT, OUTPUT and MASK must be the same
 * 3. INPUT, OUTPUT and MASK must be contiguous
 * 4. The random values are uniformly distributed in 0 ~ 1
 * 5. MASK is optional
 */
bm_status_t sgdnnDropout ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           unsigned long long seed,
                           float threshold,
                           SgdnnTensor_t output,
                           SgdnnTensor_t mask );

/*
 * OUTPUT = MLP ( INPUT, W1, W2, B1, B2, OUT1, P )
 * Note:
 * 1. The data types of INPUT, W1, W2, B1, B2, OUT1, P must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT OUT1 P and OUTPUT must be 3, W1 W2 must be 2, B1 B2 must be 1
 * 3. The shape of INPUT is ( B, M, N ), W1 is ( N, D1 ), B1 is ( D1 ), W2 is ( D1, D2 ), B2 is ( D2 ), OUT1 P is ( B, M, D1 ), OUTPUT is ( B, M, D2 )
 * 4. INPUT, W1, W2, B1, B2, OUT1, P and OUTPUT must be contiguous
 * 5. W1 B1 represents the weight and bias of first layer, OUT1 represents the output of first layer, P represents output of activation function,
 * W2 B2 represents the weight and bias of second layer
 */
bm_status_t sgdnnMlp ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          SgdnnTensor_t w1,
                          SgdnnTensor_t w2,
                          SgdnnTensor_t b1,
                          SgdnnTensor_t b2,
                          SgdnnTensor_t out1,
                          SgdnnTensor_t p,
                          SgdnnTensor_t output );

/*
 * Calc process:
    *
 * d = hidden_size / attention_heads
 * The shape of WEIGHTS:
    * cos: (n, 1, d)
    * weight2: (n, 1, d)
    * weight3: optional(n, n)
 * The shape of INPUTS:
    * Q : ( batch, attention_heads, 1, d )
    * K : ( batch, k_v_heads, 1, d )
    * V : ( batch, k_v_heads, 1, d )
    * Kcache : ( batch, k_v_heads, embeddings-1, d )
    * Vcache : ( batch, k_v_heads, embeddings-1, d )
 * The shape of OUTPUTS:
    * Y : ( batch, attention_heads, 1, d )
 *
 */
bm_status_t sgdnnLlamaAttention ( bm_handle_t handle,
                          SgdnnTensor_t Q,
                          SgdnnTensor_t K,
                          SgdnnTensor_t V,
                          SgdnnTensor_t Kcache,
                          SgdnnTensor_t Vcache,
                          SgdnnTensor_t cos,
                          SgdnnTensor_t sin,
                          SgdnnTensor_t mask,
                          SgdnnTensor_t Y,
                          int embeddings,
                          int attention_mode,
                          float C);

/*
 * [ GRAD_INPUT, GRAD_W1, GRAD_W2, GRAD_B1, GRAD_B2 ] = MLP BACKWARD ( GRAD_OUTPUT, INPUT, W1, W2, OUT1, P )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT, GRAD_OUTPUT, OUT1, P and GRAD_INPUT must be 3; W1, W2, GRAD_W1 and GRAD_W2 must be 2; GRAD_B1, GRAD_B2 must be 1
 * 3. The shape of INPUT and GRAD_INPUT is ( B, M, N ); W1 and GRAD_W1 is ( N, D1 ); W2 and GRAD_W2 is ( D1, D2 ); OUT1, P is ( B, M, D1 ); GRAD_B1 is ( D1 ); GRAD_B2 is ( D2 ); GRAD_OUTPUT is ( B, M, D2 )
 * 4. All the tensors must be contiguous
 * 5. W1 B1 represents the weight and bias of first layer, OUT1 represents the output of first layer, P represents output of activation function,
 *    W2 B2 represents the weight and bias of second layer, GRAD_x means the gradient of tensor x
 */
bm_status_t sgdnnMlpBackward ( bm_handle_t handle,
                                  SgdnnTensor_t grad_output,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t w1,
                                  SgdnnTensor_t w2,
                                  SgdnnTensor_t out1,
                                  SgdnnTensor_t p,
                                  SgdnnTensor_t grad_input,
                                  SgdnnTensor_t grad_w1,
                                  SgdnnTensor_t grad_w2,
                                  SgdnnTensor_t grad_b1,
                                  SgdnnTensor_t grad_b2);

/*
 * OUTPUT = MLP ( INPUT, WEIGHT0, WEIGHT1, WEIGHT2, OUTPUT )
 * Note:
 * 1. The data types of INPUT, WEIGHT0, WEIGHT1, WEIGHT2, OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT, WEIGHT0, WEIGHT1, WEIGHT2, OUTPUT must be 2
 * 3. The shape of INPUT is ( B, M ), WEIGHT0 is ( M, N ), WEIGHT1 is ( M, N ), WEIGHT2 is ( N, M ), OUTPUT is ( B, M )
 * 4. INPUT, WEIGHT0, WEIGHT1, WEIGHT2, OUTPUT must be contiguous
 * 5. formula: matmul(mul(matmul(INPUT, WEIGHT0), mul(matmul(INPUT, WEIGHT1), sigmoid(matmul(INPUT, WEIGHT1))), WEIGHT2)
 */
bm_status_t sgdnnLLamaMlp ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          SgdnnTensor_t weight0,
                          SgdnnTensor_t weight1,
                          SgdnnTensor_t weight2,
                          SgdnnTensor_t output);

/*
 * OUTPUT = RMSNorm ( INPUT, WEIGHT, BIAS, OUTPUT )
 * Note:
 * 1. The data types of INPUT, WEIGHT, BIAS, OUTPUT  must be the same and one of FP32, FP16 and BF16
 * 2. All the tensors must be contiguous
 */
bm_status_t sgdnnRMSNorm (  bm_handle_t handle,
                          SgdnnTensor_t input,
                          SgdnnTensor_t weight,
                          SgdnnTensor_t bias,
                          SgdnnTensor_t output,
                          int           axis,
                          float         eps,
                          float         partial,
                          int           with_scale,
                          int           with_bias);


/*
 * OUT = ATTENTION ( INPUT, W_ATTN, W_PROJ, B_ATTN, B_PROJ, Q, K, V, SOFTMAX_OUT, SOFT_V )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT Q K V and OUT must be 3, W_ATTN W_PROJ must be 2, B_ATTN B_PROJ must be 1, SOFTMAX_OUT SOFT_V must be 4
 * 3. The shape of INPUT is ( B, M, N ), W_PROJ is ( N, D_attn ), B1 is ( D_attn ), W_PROJ is ( D_attn/3, N ), B2 is ( N ),
 *    Q K V is ( B, M, D_attn/3 ), SOFTMAX_OUT is ( B, H, M, M ), SOFT_V is ( B, H, M, D_attn/3 ), OUTPUT is ( B, M, N )
 * 4. All the tensors must be contiguous
 * 5. W_ATTN B_ATTN represents the weight and bias of attention layer (which generates Q K V), W_PROJ B_PROJ represents the weight and bias of projection layer
 * 6. SOFT_V = SOFTMAX_OUT * V
 */
bm_status_t sgdnnAttn ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          SgdnnTensor_t w_attn,
                          SgdnnTensor_t w_proj,
                          SgdnnTensor_t b_attn,
                          SgdnnTensor_t b_proj,
                          SgdnnTensor_t q,
                          SgdnnTensor_t k,
                          SgdnnTensor_t v,
                          SgdnnTensor_t softmax_out,
                          SgdnnTensor_t soft_v,
                          SgdnnTensor_t out );

/*
 * [ GRAD_INPUT, GRAD_W_ATTN, GRAD_W_PROJ, GRAD_B_ATTN, GRAD_B_PROJ ] = ATTENTION BACKWARD ( GRAD_OUTPUT, INPUT, W_ATTN, W_PROJ, Q, K, V, SOFTMAX_OUT, SOFT_V, BIAS )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT, GRAD_OUTPUT, Q, K, V and GRAD_INPUT must be 3; W_ATTN, W_PROJ, GRAD_W_ATTN and GRAD_W_PROJ must be 2; GRAD_B_ATTN, GRAD_B_PROJ must be 1,
 *    SOFTMAX_OUT, SOFT_V and BIAS must be 4
 * 3. The shape of INPUT and GRAD_INPUT is ( B, M, N ); W_ATTN and GRAD_W_ATTN is ( N, D_attn ); W_PROJ and GRAD_W_PROJ is ( D_attn/3, N );
 *    Q K V is ( B, M, D_attn/3 ); SOFTMAX_OUT is ( B, H, M, M ); SOFT_V is ( B, H, M, D_attn/3 );  BIAS is ( 1, 1, M, M );
 *    GRAD_B_ATTN is ( D_attn ); GRAD_B_PROJ is ( N ); GRAD_OUTPUT is ( B, M, N )
 * 4. All the tensors must be contiguous
 * 5. W_ATTN B_ATTN represents the weight and bias of attention layer (which generates Q K V), W_PROJ B_PROJ represents the weight and bias of projection layer,
 *    GRAD_x means the gradient of tensor x, BIAS represents triangular matrix (mask).
 */
bm_status_t sgdnnAttnBackward ( bm_handle_t handle,
                                  SgdnnTensor_t grad_output,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t w_attn,
                                  SgdnnTensor_t w_proj,
                                  SgdnnTensor_t q,
                                  SgdnnTensor_t k,
                                  SgdnnTensor_t v,
                                  SgdnnTensor_t softmax_out,
                                  SgdnnTensor_t soft_v,
                                  SgdnnTensor_t bias,
                                  SgdnnTensor_t grad_input,
                                  SgdnnTensor_t grad_w_attn,
                                  SgdnnTensor_t grad_w_proj,
                                  SgdnnTensor_t grad_b_attn,
                                  SgdnnTensor_t grad_b_proj);


/*
 *  OUTPUT  = ARANGE ( START, END, STEP )
 *  Note:
 *  1. START, END and STEP only support int dtype.
 */
bm_status_t sgdnnArange ( bm_handle_t handle,
                                int start,
                                int end,
                                int step,
                                SgdnnTensor_t out);

/*
 *  OUTPUT  = ELEMENT_BITWISE ( input, other, output )
 *  Note:
 *  1. input, other and output only support int32 uint32 int8 uint8.
 *  2. input, other and output must be the same shape.
 *  3. mode : 0 for xor, 1 for and, 2 for or
 */
bm_status_t sgdnnElementBitwise ( bm_handle_t handle,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t other,
                                  int mode,
                                  SgdnnTensor_t output );

/*
 *  OUTPUT  = ELEMENT_BITWISE ( input, other, output )
 *  Note:
 *  1. input, other and output only support int32 uint32 int8 uint8.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 *  4. mode : 0 for xor, 1 for and, 2 for or
 */
bm_status_t sgdnnElementBitwiseBcast ( bm_handle_t handle,
                                       SgdnnTensor_t input,
                                       SgdnnTensor_t other,
                                       int mode,
                                       SgdnnTensor_t output );
/*
 *  OUTPUT  = ELEMENT_BITWISE ( input, other, output )
 *  Note:
 *  1. input, other and output only support int32 uint32 int8 uint8.
 *  2. input and output must be the same shape.
 *  3. mode : 0 for xor, 1 for and, 2 for or
 */
bm_status_t sgdnnElementBitwiseC ( bm_handle_t handle,
                                   SgdnnTensor_t input,
                                   int scalar,
                                   int mode,
                                   SgdnnTensor_t output );

/*
 * OUTPUT = ROUND(INPUT)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnRound (bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output );

/*
 * OUTPUT = NEG(INPUT)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */

bm_status_t sgdnnNeg (bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output );

/*
 * OUTPUT = BITWISE_NOT(INPUT)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and INT32
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnBitwiseNot (bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output );


/*
 *  OUTPUT  = COMPARISION ( input, other, output )
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input, other and output must be the same shape.
 *  3. 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
 */
bm_status_t sgdnnComparision ( bm_handle_t handle,
                               SgdnnTensor_t input,
                               SgdnnTensor_t other,
                               int mode,
                               SgdnnTensor_t output );
/*
 *  OUTPUT  = COMPARISION ( input, other, output )
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 *  4. 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
 */
bm_status_t sgdnnComparisionBcast ( bm_handle_t handle,
                                    SgdnnTensor_t input,
                                    SgdnnTensor_t other,
                                    int mode,
                                    SgdnnTensor_t output );
/*
 *  OUTPUT  = COMPARISION ( input, other, output )
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. one of input and other must be a scalar.
 *  3. input and output must be in the same shape.
 *  4. 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
 */
bm_status_t sgdnnComparisionC ( bm_handle_t handle,
                                SgdnnTensor_t input,
                                float scalar,
                                int mode,
                                int scalar_pos,
                                SgdnnTensor_t output );

/*
 * OUTPUT = MINIMUMC(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16, BF16 and INT32
 * 2. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnMinimumC ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = MINIMUM(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. The shapes of INPUT ,OTHER and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnMinimum ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = MINIMUMBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
bm_status_t sgdnnMinimumBcast ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );
/*
 * OUTPUT = MAXIMUMC(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16, BF16 and INT32
 * 2. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnMaximumC ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = MAXIMUM(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. The shapes of INPUT ,OTHER and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnMaximum ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = MINIMUMBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
bm_status_t sgdnnMaximumBcast ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = ATAN2C(SCALAR,OTHER)
 * Note:
 * 1. The data types of OTHER must be FP32 and data type of OUTPUT must be one of FP32, FP16 and F16
 * 2. The shapes of OTHER and OUTPUT must be the same
 * 3. OTHER, OUTPUT must be contiguous
 */
bm_status_t sgdnnAtan2C ( bm_handle_t handle,
                        float scalar,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = ATAN2_C(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT must be FP32 and data type of OUTPUT must be one of FP32, FP16 and F16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT, OUTPUT must be contiguous
 */
bm_status_t sgdnnAtan2_C ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = ATAN2(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT and OTHER must be FP32 and data type of OUTPUT must be one of FP32, FP16 and F16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same
 * 3. INPUT, OTHER, OUTPUT must be contiguous
 */
bm_status_t sgdnnAtan2 ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = ATAN2BCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT and OTHER must be FP32 and data type of OUTPUT must be one of FP32, FP16 and F16
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
bm_status_t sgdnnAtan2Bcast ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT  = Badddbmm ( INPUT1, BATCH1, BATCH2, OUT, ALPHA, BETA )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of BATCH1, BATCH2 and OUT must be 3
 * 3. The shape of INPUT1 is ( B, M, D2 ), BATCH1 is ( B, M, D1 ), BATCH2 is ( B, D1, D2 ), OUT is ( B, M, D2 )
*/
bm_status_t sgdnnBaddbmm ( bm_handle_t handle,
                          SgdnnTensor_t input1,
                          SgdnnTensor_t batch1,
                          SgdnnTensor_t batch2,
                          SgdnnTensor_t out,
                          double alpha,
                          double beta);

/*
 *  OUTPUT  = MSE_LOSS ( SELF, TARGET, OUT, REDUCTION )
 * Note:
 * 1. The shape of SELF and TARGET must be the same.
 * 2. The shape of SELF or TARGET can be any shape.
 * 3. REDUCTION = 0, 'none' model, OUT = ( SELF - TARGET )^2, the shape of OUT the same as SELF.
 * 4. REDUCTION = 1, 'mean' model, OUT = REDUCE_MEAN( ( SELF - TARGET )^2 ), the shape of OUT is (1,).
 * 5. REDUCTION = 2, 'sum' model, OUT = REDUCE_SUM( ( SELF - TARGET )^2 ), the shape of OUT is (1,).
 */
bm_status_t sgdnnMseloss( bm_handle_t handle,
                                    SgdnnTensor_t self,
                                    SgdnnTensor_t target,
                                    SgdnnTensor_t out,
                                    int reduction );

/*
 * OUTPUT = LAYERNORM_MATMUL ( INPUT, W, B, GAMMA, BETA, EPS, MEAN, RSTD, OUTPUT )
 * Note:
 * 1. The data types of INPUT, W, B, GAMMA, BETA, EPS, MEAN, RSTD, OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT OUTPUT must be 2 or 3, W MEAN RSTD must be 2, GAMMA BETA B must be 1
 * 3. The shape of INPUT is ( B, M, N ) or ( M, N ), W1 is ( N, D ), B is ( D ), OUTPUT is ( B, M, D ) or ( M, N )
 * 4. INPUT, W, B, GAMMA, BETA, MEAN, RSTD and OUTPUT must be contiguous
 * 5. GAMMA BETA represents the elementwise affine in LayerNorm, MEAN RSTD represents mean and rstd of LayerNorm
 * 6. W B represents the weight and bias of first layer
 */
bm_status_t sgdnnLnMm ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          SgdnnTensor_t w,
                          SgdnnTensor_t b,
                          SgdnnTensor_t gamma,
                          SgdnnTensor_t beta,
                          float eps,
                          SgdnnTensor_t mean,
                          SgdnnTensor_t rstd,
                          SgdnnTensor_t output );

/*
 * OUTPUT = FMAXC(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16, BF16 and INT32
 * 2. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnFmaxC ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = FMAX(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. The shapes of INPUT ,OTHER and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnFmax ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = FMAXBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
bm_status_t sgdnnFmaxBcast ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = FMINC(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16, BF16 and INT32
 * 2. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnFminC ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = FMIN(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. The shapes of INPUT ,OTHER and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnFmin ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = FMINBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
bm_status_t sgdnnFminBcast ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = LAYERNORM_MATMUL ( INPUT0, INPUT1, W, B, GAMMA, BETA, EPS, OUT_ADDR, MEAN, RSTD, OUTPUT )
 * Note:
 * 1. The data types of INPUT0, INPUT1, W, B, GAMMA, BETA, EPS, OUT_ADDR, MEAN, RSTD, OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT0 INPUT1 OUT_ADDR OUTPUT must be 2 or 3, W MEAN RSTD must be 2, GAMMA BETA B must be 1
 * 3. The shape of INPUT0/INPUT1/OUT_ADDR is ( B, M, N ) or ( M, N ), W1 is ( N, D ), B is ( D ), OUTPUT is ( B, M, D ) or ( M, N )
 * 4. INPUT0, INPUT1, W, B, GAMMA, BETA, OUT_ADDR, MEAN, RSTD and OUTPUT must be contiguous
 * 5. GAMMA BETA represents the elementwise affine in LayerNorm, MEAN RSTD represents mean and rstd of LayerNorm
 * 6. W B represents the weight and bias of first layer
 * 7. OUT_ADDR represents the result of INPUT0 + INPUT1
 */
bm_status_t sgdnnAddLnMm ( bm_handle_t handle,
                          SgdnnTensor_t input0,
                          SgdnnTensor_t input1,
                          SgdnnTensor_t w,
                          SgdnnTensor_t b,
                          SgdnnTensor_t gamma,
                          SgdnnTensor_t beta,
                          float eps,
                          SgdnnTensor_t out_add,
                          SgdnnTensor_t mean,
                          SgdnnTensor_t rstd,
                          SgdnnTensor_t output );

/*
 * OUTPUT = SIGNBIT(INPUT)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
bm_status_t sgdnnSignbit(bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output );


/*
 * OUTPUT = POWC(INPUT, EXPONENT)
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 * 4. EXPONENT is a single float number.
*/
bm_status_t sgdnnPowC ( bm_handle_t handle,
                      SgdnnTensor_t self,
                      float scalar,
                      SgdnnTensor_t out );

/*
 * OUTPUT = Real(INPUT)
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * 2. INPUT must be contiguous
 */
bm_status_t sgdnnReal ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output);
bm_status_t sgdnnConj ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t output);
/*
 * OUTPUT = PERMUTE ( INPUT, DIM_ORDER )
 * Node:
 * 1. the dim of input and out must be the same
 * 2. the data type of input and out must be the same
 * 3. the number of dim_order must be the same with input's dim
 * 4. input and output must be contiguous
 */
bm_status_t sgdnnPermute ( bm_handle_t handle,
                           SgdnnTensor_t input,
                           int *dim_order,
                           SgdnnTensor_t output );

/*
 * value, index = TOPK ( input, k, dim, largest, sorted )
 * Node:
 * 1. the data type of input, value must be the same
 * 2. only support in FP32, INT32, UINT32
 * 3. input, value and index must be contiguous
 */
bm_status_t sgdnnTopk ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        int k,
                        int dim,
                        bool largest,
                        bool sorted,
                        SgdnnTensor_t value,
                        SgdnnTensor_t index );

/*
 * OUTPUT = NONZERO (INPUT)
 * Node:
 * 1.input must be contiguous
 * 2.only support in UINT8, INT8, INT32
 * 3.the dtype of output is INT32
*/
bm_status_t sgdnnNonzero ( bm_handle_t handle,
                      SgdnnTensor_t self,
                      SgdnnTensor_t out,
                      SgdnnTensor_t num);

/*
 * OUTPUT = REDUCE_MAX_OR_MIN ( INPUT, REDUCTION_DIM, REDUCTION_DIM_LENGTH, KEEPDIM, MODE )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. INPUT and OUTPUT must be contiguous
 * 3. MODE must be 0 ( max ) or 1 ( min )
 */
bm_status_t sgdnnReduceMaxOrMin ( bm_handle_t handle,
                              SgdnnTensor_t input,
                              int *reduction_dim,
                              int reduction_dim_length,
                              int keepdim,
                              int mode,
                              SgdnnTensor_t output );
/*
 * { VALUES,INDICES } = ARG ( INPUT, AXIS, MODE )
 * Note:
 * 1. The data types of INPUT and VALUES must be the same and one of FP32, FP16 and BF16
 * 2. INPUT, VALUES, INDICES must be contiguous
 * 3. The shape of VALUES and INDICES must be the same
 * 3. MODE must be 0 ( argmax ) or 1 ( argmin ) or 2 ( max.dim ) or 3 ( min.dim )
 */
bm_status_t sgdnnArg( bm_handle_t handle,
                              SgdnnTensor_t input,
                              int axis,
                              int mode,
                              SgdnnTensor_t values,
                              SgdnnTensor_t indices);


/*
 * out = REPEAT ( input, repeat_times, repeat_dim )
 * Node:
 * 1. the data type of input and output must be the same
 * 2. dim <= 4 && dim <= repeat_dim
 * 3. input and output must be contiguous
 */
bm_status_t sgdnnRepeat ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          int *repeat_times,
                          int repeat_dim,
                          SgdnnTensor_t output );

/*
 * out = HARDTANH ( input, output )
 * Node:
 * 1. the data type of input and output must be the same
 * 2. the shape of input and output must be the same
 * 3. input and output must be contiguous
 * 4. dim <= 4
 */
bm_status_t sgdnnHardtanh ( bm_handle_t handle,
                            SgdnnTensor_t input,
                            float min_value,
                            float max_value,
                            SgdnnTensor_t output );

/*
 *  OUTPUT  = HYPOT ( input, other )
 *  Note:
 *  1. input, other and output only support float32.
 *  2. input, other and output must be the same shape.
 *  3. input, other and output must be contiguous
 */
bm_status_t sgdnnHypot ( bm_handle_t handle,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output );

/*
 *  OUTPUT  = HYPOT ( input, other )
 *  Note:
 *  1. input, other and output only support float32.
 *  2. input and other must be the same dim.
 *  3. input, other and output must be contiguous
 */
bm_status_t sgdnnHypotBcast ( bm_handle_t handle,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output );
/*
 *  OUTPUT  = HYPOT ( input, other )
 *  Note:
 *  1. input, other and output only support float32.
 *  2. input and output must be the same shape.
 *  3. input and output must be contiguous
 */
bm_status_t sgdnnHypotC ( bm_handle_t handle,
                          SgdnnTensor_t input,
                          float scalar,
                          SgdnnTensor_t output );

/*
 * OUTPUT = NEXTAFTERC(SCALAR,OTHER)
 * Note:
 * 1. The data types of OTHER and OUTPUT must be the same and one of FP32 and BF16
 * 2. The shapes of OTHER and OUTPUT must be the same
 * 3. OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnNextafterC ( bm_handle_t handle,
                        float scalar,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = NEXTAFTER_C(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. OUTPUT must be contiguous
 */
bm_status_t sgdnnNextafter_C ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output );

/*
 * OUTPUT = NEXTAFTER(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnNextafter ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 * OUTPUT = NEXTAFTERBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32 and BF16
 * 2. INPUT ,OTHER and OUTPUT must be contiguous
 */
bm_status_t sgdnnNextafterBcast ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output );

/*
 *  OUTPUT  = VAR ( input, reduce_list, correction, keepdim )
 *  Note:
 *  1. input and output only support float32.
 *  2. input and output must be contiguous
 *  3. reduce_dims less than or equal to dim of input
 */
bm_status_t sgdnnReduceVar ( bm_handle_t handle,
                             SgdnnTensor_t input,
                             int *reduce_list,
                             int reduce_dim,
                             int correction,
                             bool keepdim,
                             SgdnnTensor_t output );

/*
 *  OUTPUT  = VAR ( input, correction, keepdim )
 *  Note:
 *  1. input and output only support float32.
 *  2. input and output must be contiguous
 *  3. reduce_dims less than or equal to dim of input
 */
bm_status_t sgdnnReduceVarAll ( bm_handle_t handle,
                                SgdnnTensor_t input,
                                int correction,
                                bool keepdim,
                                SgdnnTensor_t output );
/*
 *  OUTPUT = TRIANGULARIZE(SELF, IS_UPPER, DIAGONAL)
 *  NOTE:
 *  1. SELF and OUT must be contiguous
 *  2. The data types of SELF and OUT must be the same and one of FP32, FP16 and BF16
 *  3. The shapes of SELF and OUT must be the same
*/
bm_status_t sgdnnTriangularize ( bm_handle_t handle,
                      SgdnnTensor_t self,
                      int is_upper,
                      int diagonal,
                      SgdnnTensor_t out );

/*
 *  output  = CBRT ( input )
 *  Note:
 *  1. input and output must be the same shape.
 *  2. input and output must be contiguous
 */
bm_status_t sgdnnCbrt ( bm_handle_t handle,
                        SgdnnTensor_t input,
                        SgdnnTensor_t output );

/*
 *  OUTPUT  = PAD ( INPUT, PAD, VALUE )
 *  Note:
 *  1. input and output must be the same dim.
 *  2. input and output must be the same dtype.
 *  3. input and output must be contiguous
 *  4. pad size must less than or equal to twice the number of the input dimensions
 */
bm_status_t sgdnnPad ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       int *pad,
                       int pad_size,
                       float value,
                       int mode,
                       bool pad3d,
                       SgdnnTensor_t output );

/*
 *  OUTPUT  = SLICE_SCATTER ( INPUT, SRC ,INDICES,dim)
 *  Note:
 *  1. input and output must be the same dim.
 *  2. input and output must be the same dtype.
 *  3. input and output must be contiguous
 */
bm_status_t sgdnnSliceScatter ( bm_handle_t handle,
                       SgdnnTensor_t input,
                       SgdnnTensor_t src,
                       SgdnnTensor_t indices,
                       int dim,
                       SgdnnTensor_t output );

/*
*  Found_inf = inf in input ? 1 : 0;
*  Input     = inv_scale * Input;
*  Note:
*  1. the dim of found_inf = 1
*/
bm_status_t sgdnnInfCheckAndUnscale( bm_handle_t handle,
                                    std::vector<SgdnnTensor_t>& input,
                                    SgdnnTensor_t found_inf,
                                    float inv_scale );

#if defined(__cplusplus)
}
#endif

#endif /* SGDNN_API_H */

