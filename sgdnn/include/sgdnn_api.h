#ifndef SGDNN_API_H
#define SGDNN_API_H

#include "sg_api_struct.h"
#include "sgdnn_runtime.h"
#include <map>
#include <vector>

#if defined(__cplusplus)
extern "C" {
#endif

#define DIV_UP(a, b) ( ( ( a ) + ( b ) - 1 ) / ( b ) )

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
  SGDNN_DTYPE_INT64,
  SGDNN_DTYPE_FP8E4M3,
}
SgdnnDataType_t;

typedef enum
{
  SGDNN_NO_FORMATED = 0,
  SGDNN_CONV_W_INFER_FORMAT  = 1,
  SGDNN_CONV_W_TRAIN_FORMAT  = 2,
  SGDNN_CONV_DW_TRAIN_FORMAT = 3,
}
SgdnnFormatedType_t;

typedef struct
{
  unsigned long long addr;
  int dim;
  int shape[8];
  int stride[8];
  SgdnnDataType_t dtype;
  SgdnnFormatedType_t format_casted = SGDNN_NO_FORMATED; // default no cast
}
SgdnnTensor_t;

static inline SgdnnTensor_t sgdnnUndefinedTensor()
{
  SgdnnTensor_t tensor = { .addr = 0 };
  return tensor;
}

tpu_status_t sgdnnInitialize( tpu_resource_t resource );

tpu_status_t sgdnnDeinitialize( tpu_resource_t resource );

#if defined BACKEND_SG2260
//must implement non_blocking
tpu_status_t sgdnnTPUKernelLaunch (
            tpu_resource_t resource,
            const char * func_name,
            const void * api,
            size_t api_size,
            bool non_blocking,
            int group_num = 1,
            int block_num = 8);

static inline tpu_status_t sgdnnTPUKernelLaunchMultiCore (
                            tpu_resource_t resource,
                            const char * func_name,
                            const void * api,
                            size_t api_size,
                            bool non_blocking,
                            int group_num = 1,
                            int block_num = 8) {
  return sgdnnTPUKernelLaunch(resource, func_name, api, api_size, non_blocking, group_num, block_num);
}

#else
tpu_status_t sgdnnTPUKernelLaunch (
            tpu_resource_t resource,
            const char * func_name,
            const void * api,
            size_t api_size,
            bool non_blocking = false,
            int group_num = 1,
            int block_num = 1);
#endif

/**
 * HelerFunction for CONV 32 IC and 32OC
*/
void sgdnn32ICShape ( const int * shape, int * _32ic_shape );
void sgdnn32OCShape ( const int * shape, int * _32oc_shape );
void sgdnnContiguousStride ( const int * shape, int dim,  int * stride );


tpu_status_t sgdnnReorderConv2dWeight ( tpu_resource_t resource,
                                         SgdnnTensor_t input,
                                         int mode,
                                         SgdnnTensor_t output,
                                         bool non_blocking = true );

tpu_status_t sgdnnReorderConv2dGrad ( tpu_resource_t resource ,
                                       SgdnnTensor_t input,
                                       SgdnnTensor_t output,
                                       bool non_blocking = true );

tpu_status_t sgdnnRecoverConv2dWeight ( tpu_resource_t resource ,
                                       SgdnnTensor_t input,
                                       int mode,
                                       SgdnnTensor_t output,
                                       bool non_blocking = true );

tpu_status_t sgdnnRecoverConv2dGrad ( tpu_resource_t resource ,
                                      SgdnnTensor_t input,
                                      SgdnnTensor_t output,
                                      bool non_blocking = true );

tpu_status_t sgdnnConvertInt64toInt32 ( tpu_resource_t resource,
                                         SgdnnTensor_t input,
                                         SgdnnTensor_t output,
                                         bool non_blocking = true );

/**
 * ONLY USED FOR TEST KERNEL LAUNCH
*/
tpu_status_t sgdnnDummy ( tpu_resource_t  stream,
                           bool non_blocking );
/**
 * ONLY USED FOR TEST HOST CPU TIME
*/
tpu_status_t sgdnnDummy_WO_KERNEL_LAUNCH ( tpu_resource_t  stream,
                           bool non_blocking );

/**
 * Physical Memory Format Cast OP.
 * Because Some TPU's instrution( operation ) need particular data layout.
 * For example conv need 32IC for fp16, 64IC for in8.
 * - cast_type :
 *          0 - 32IC
*/
tpu_status_t sgdnnFormatCast( tpu_resource_t  resource,
                              SgdnnTensor_t input,
                              SgdnnTensor_t output,
                              int cast_type
                              );

/*
 * [ OUTPUT, MEAN, RSTD ] = LAYERNORM ( INPUT, WEIGHT, BIAS, START_DIM, EPS )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 4. The shape of INPUT is ( D0, D1, ..., D(A-1), DA, D(A+1), ... ), where DA is the START_DIM, then WEIGHT and BIAS are ( D(A), D(A+1), ... ), MEAN and RSTD are ( D0, D1, ..., D(A-1), 1, 1, ... )
 * 5. All the tensors must be contiguous
 * 6. WEIGHT and BIAS are optional
 */
tpu_status_t sgdnnLayernorm ( tpu_resource_t  stream,
                             SgdnnTensor_t input,
                             SgdnnTensor_t weight,
                             SgdnnTensor_t bias,
                             int start_dim,
                             float eps,
                             SgdnnTensor_t output,
                             SgdnnTensor_t mean,
                             SgdnnTensor_t rstd ,
                             bool non_blocking = true);

/*
 * [ GRAD_INPUT, GRAD_WEIGHT, GRAD_BIAS ] = LAYERNORM BACKWARD ( GRAD_OUTPUT, INPUT, WEIGHT, MEAN, RSTD, START_DIM )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, GRAD_OUTPUT and GRAD_INPUT must be the same
 * 4. The shape of INPUT is ( D0, D1, ..., D(A-1), DA, D(A+1), ... ), where DA is the START_DIM, then WEIGHT, GRAD_WEIGHT and GRAD_BIAS are ( D(A), D(A+1), ... ), MEAN and RSTD are ( D0, D1, ..., D(A-1), 1, 1, ... )
 * 5. All the tensors must be contiguous
 * 6. WEIGHT, GRAD_INPUT, GRAD_WEIGHT and GRAD_BIAS are optional
 */
tpu_status_t sgdnnLayernormBackward ( tpu_resource_t  stream,
                                     SgdnnTensor_t grad_output,
                                     SgdnnTensor_t input,
                                     SgdnnTensor_t weight,
                                     SgdnnTensor_t mean,
                                     SgdnnTensor_t rstd,
                                     int start_dim,
                                     SgdnnTensor_t grad_input,
                                     SgdnnTensor_t grad_weight,
                                     SgdnnTensor_t grad_bias ,
                                     int requires_grad_input,
                                     bool non_blocking = true);


/*
 * OUTPUT = native_group_norm(INPUT)
 * Note:
 */
tpu_status_t sgdnnNativeGroupNorm ( tpu_resource_t  stream, SgdnnTensor_t input,
                     SgdnnTensor_t gamma, SgdnnTensor_t beta,
                     int group, int affine, float eps, SgdnnTensor_t output,
                     SgdnnTensor_t mean, SgdnnTensor_t rstd, bool non_blocking = true);

tpu_status_t sgdnnNativeGroupNormBackward(tpu_resource_t  stream, SgdnnTensor_t grad_output,
                     SgdnnTensor_t input, SgdnnTensor_t weight,
                     SgdnnTensor_t mean, SgdnnTensor_t rstd,
                     int group, SgdnnTensor_t out0,
                     SgdnnTensor_t out1, SgdnnTensor_t out2, bool non_blocking = true);


/*
 *  OUTPUT  = SHIFT_LEFT ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 */
tpu_status_t sgdnnShiftLeft ( tpu_resource_t  stream,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output ,
                         bool non_blocking = true);
/*
 *  OUTPUT  = SHIFT_LEFT ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 */
tpu_status_t sgdnnShiftLeftBcast ( tpu_resource_t  stream,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output ,
                              bool non_blocking = true);
/*
 *  OUTPUT  = SHIFT_LEFT ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. one of input and other must be a scalar.
 *  3. input and output must be in the same shape.
 */
tpu_status_t sgdnnShiftLeftC ( tpu_resource_t  stream,
                          SgdnnTensor_t input,
                          char scalar,
                          SgdnnTensor_t output ,
                          bool non_blocking = true);

/*
 *  OUTPUT  = SHIFT_RIGHT_ARITHMETIC ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 */
tpu_status_t sgdnnShiftRightArithmetic ( tpu_resource_t  stream,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output ,
                         bool non_blocking = true);
/*
 *  OUTPUT  = SHIFT_RIGHT_ARITHMETIC ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 */
tpu_status_t sgdnnShiftRightArithmeticBcast ( tpu_resource_t  stream,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output ,
                              bool non_blocking = true);
/*
 *  OUTPUT  = SHIFT_RIGHT_ARITHMETIC ( INPUT, OTHER)
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. one of input and other must be a scalar.
 *  3. input and output must be in the same shape.
 */
tpu_status_t sgdnnShiftRightArithmeticC ( tpu_resource_t  stream,
                          SgdnnTensor_t input,
                          int scalar,
                          SgdnnTensor_t output ,
                          bool non_blocking = true);

/*
 * OUTPUT = math.flip(INPUT, DIMS)
 * Note:
 * 1. The data types of INPUT must be the same and one of FP32, FP16 and BF16
 * 2. INPUT must be contiguous
 */

tpu_status_t sgdnnFlip ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        int axis,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);
/*
 * OUTPUT = math.pow (INPUT, OTHER)
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be one of FP32, FP16, BF16 and INT32
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnPow ( tpu_resource_t  stream,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output ,
                       bool non_blocking = true);

/*
 * OUTPUT = math.pow (INPUT, OTHER)
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be one of FP32, FP16, BF16 and INT32
 * 2. The shapes of INPUT, OTHER must can be broadcast
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnPowBcast ( tpu_resource_t  stream,
                            SgdnnTensor_t input,
                            SgdnnTensor_t other,
                            SgdnnTensor_t output,
                            bool non_blocking = true );

/*
 * OUTPUT = INPUT - SCALAR * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnSub ( tpu_resource_t  stream,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       float scalar,
                       SgdnnTensor_t output ,
                       bool non_blocking = true);

/*
 * OUTPUT = INPUT * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnMul ( tpu_resource_t  stream,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output ,
                       bool non_blocking = true);

/*
 * OUTPUT = INPUT / OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnDiv ( tpu_resource_t  stream,
                       SgdnnTensor_t input,
                       SgdnnTensor_t other,
                       SgdnnTensor_t output ,
                       bool non_blocking = true);

/*
 * OUTPUT = INPUT + SCALAR * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same, broadcasting is not allowed
 * 3. INPUT, OTHER and OUTPUT must be contiguous
 */

tpu_status_t sgdnnBinary ( tpu_resource_t  stream,
                            SgdnnTensor_t input,
                            SgdnnTensor_t other,
                            float scalar,
                            SgdnnTensor_t output,
                            int binary_type ,
                            bool non_blocking = true);

/*
 * OUTPUT = INPUT + SCALAR * OTHER
 * Note:
 * 1. The data types of INPUT, OTHER and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The dimentions of INPUT, OTHER and OUTPUT must be the same
 * 3. INPUT and OTHER are allowed broadcasting
 * 4. INPUT, OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnBinaryBcast (  tpu_resource_t  stream,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t other,
                                  float scalar,
                                  SgdnnTensor_t output,
                                  int binary_type ,
                                  bool non_blocking = true);

/*
 * OUTPUT = INPUT + SCALAR
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
tpu_status_t sgdnnBinaryC (  tpu_resource_t  stream,
                              SgdnnTensor_t input,
                              float scalar,
                              SgdnnTensor_t output,
                              int binary_type,
                              bool inversed ,
                              bool non_blocking = true);

/*
 * OUTPUT = SCALAR - INPUT
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
tpu_status_t sgdnnCSub ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = INPUT * SCALAR
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 */
tpu_status_t sgdnnMulC ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = COPY ( INPUT )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT are allowed to be NOT contiguous
 */
tpu_status_t sgdnnStridedCopy ( tpu_resource_t resource,
                             SgdnnTensor_t input,
                             SgdnnTensor_t output,
                             bool non_blocking = true );

/*
 * OUTPUT = INDEX SELECT ( INPUT, INDICES, DIM )
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same, INDICES must be INT32 or INT64, but used as INT32
 * 2. The shape of INPUT is ( D0, ..., D(D-1), DD, D(D+1), ... ), where DD is the DIM, INDICES is ( I0, ..., IX ),
 *    then OUTPUT is ( D0, ..., D(D-1), [X+1], D(D+1), ...  )
 * 3. INPUT, INDICES and OUTPUT must be contiguous
 */
tpu_status_t sgdnnIndexSelect ( tpu_resource_t  stream,
                               SgdnnTensor_t input,
                               SgdnnTensor_t indices,
                               int dim,
                               SgdnnTensor_t output ,
                               bool non_blocking = true);

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
tpu_status_t sgdnnMlp ( tpu_resource_t  stream,
                          SgdnnTensor_t input,
                          SgdnnTensor_t w1,
                          SgdnnTensor_t w2,
                          SgdnnTensor_t b1,
                          SgdnnTensor_t b2,
                          SgdnnTensor_t out1,
                          SgdnnTensor_t p,
                          SgdnnTensor_t output ,
                          bool non_blocking = true);

/*
 * [ GRAD_INPUT, GRAD_W1, GRAD_W2, GRAD_B1, GRAD_B2 ] = MLP BACKWARD ( GRAD_OUTPUT, INPUT, W1, W2, OUT1, P )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT, GRAD_OUTPUT, OUT1, P and GRAD_INPUT must be 3; W1, W2, GRAD_W1 and GRAD_W2 must be 2; GRAD_B1, GRAD_B2 must be 1
 * 3. The shape of INPUT and GRAD_INPUT is ( B, M, N , bool non_blocking = true); W1 and GRAD_W1 is ( N, D1 , bool non_blocking = true); W2 and GRAD_W2 is ( D1, D2 , bool non_blocking = true); OUT1, P is ( B, M, D1 , bool non_blocking = true); GRAD_B1 is ( D1 , bool non_blocking = true); GRAD_B2 is ( D2 , bool non_blocking = true); GRAD_OUTPUT is ( B, M, D2 )
 * 4. All the tensors must be contiguous
 * 5. W1 B1 represents the weight and bias of first layer, OUT1 represents the output of first layer, P represents output of activation function,
 *    W2 B2 represents the weight and bias of second layer, GRAD_x means the gradient of tensor x
 */
tpu_status_t sgdnnMlpBackward ( tpu_resource_t  stream,
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
                                  SgdnnTensor_t grad_b2,
                                  bool non_blocking = true);

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
tpu_status_t sgdnnAttn ( tpu_resource_t  stream,
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
                          SgdnnTensor_t out ,
                          bool non_blocking = true);

/*
 * [ GRAD_INPUT, GRAD_W_ATTN, GRAD_W_PROJ, GRAD_B_ATTN, GRAD_B_PROJ ] = ATTENTION BACKWARD ( GRAD_OUTPUT, INPUT, W_ATTN, W_PROJ, Q, K, V, SOFTMAX_OUT, SOFT_V, BIAS )
 * Note:
 * 1. The data types of all the tensors must be the same and one of FP32, FP16 and BF16
 * 2. The dimensions of INPUT, GRAD_OUTPUT, Q, K, V and GRAD_INPUT must be 3; W_ATTN, W_PROJ, GRAD_W_ATTN and GRAD_W_PROJ must be 2; GRAD_B_ATTN, GRAD_B_PROJ must be 1,
 *    SOFTMAX_OUT, SOFT_V and BIAS must be 4
 * 3. The shape of INPUT and GRAD_INPUT is ( B, M, N , bool non_blocking = true); W_ATTN and GRAD_W_ATTN is ( N, D_attn , bool non_blocking = true); W_PROJ and GRAD_W_PROJ is ( D_attn/3, N , bool non_blocking = true);
 *    Q K V is ( B, M, D_attn/3 , bool non_blocking = true); SOFTMAX_OUT is ( B, H, M, M , bool non_blocking = true); SOFT_V is ( B, H, M, D_attn/3 , bool non_blocking = true);  BIAS is ( 1, 1, M, M , bool non_blocking = true);
 *    GRAD_B_ATTN is ( D_attn , bool non_blocking = true); GRAD_B_PROJ is ( N , bool non_blocking = true); GRAD_OUTPUT is ( B, M, N )
 * 4. All the tensors must be contiguous
 * 5. W_ATTN B_ATTN represents the weight and bias of attention layer (which generates Q K V), W_PROJ B_PROJ represents the weight and bias of projection layer,
 *    GRAD_x means the gradient of tensor x, BIAS represents triangular matrix (mask).
 */
tpu_status_t sgdnnAttnBackward ( tpu_resource_t  stream,
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
                                  SgdnnTensor_t grad_b_proj,
                                  bool non_blocking = true);

/*
 *  OUTPUT  = ELEMENT_BITWISE ( input, other, output )
 *  Note:
 *  1. input, other and output only support int32 uint32 int8 uint8.
 *  2. input, other and output must be the same shape.
 *  3. mode : 0 for xor, 1 for and, 2 for or
 */
tpu_status_t sgdnnElementBitwise ( tpu_resource_t  stream,
                                  SgdnnTensor_t input,
                                  SgdnnTensor_t other,
                                  int mode,
                                  SgdnnTensor_t output ,
                                  bool non_blocking = true);

/*
 *  OUTPUT  = ELEMENT_BITWISE ( input, other, output )
 *  Note:
 *  1. input, other and output only support int32 uint32 int8 uint8.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 *  4. mode : 0 for xor, 1 for and, 2 for or
 */
tpu_status_t sgdnnElementBitwiseBcast ( tpu_resource_t  stream,
                                       SgdnnTensor_t input,
                                       SgdnnTensor_t other,
                                       int mode,
                                       SgdnnTensor_t output ,
                                       bool non_blocking = true);
/*
 *  OUTPUT  = ELEMENT_BITWISE ( input, other, output )
 *  Note:
 *  1. input, other and output only support int32 uint32 int8 uint8.
 *  2. input and output must be the same shape.
 *  3. mode : 0 for xor, 1 for and, 2 for or
 */
tpu_status_t sgdnnElementBitwiseC ( tpu_resource_t  stream,
                                   SgdnnTensor_t input,
                                   int scalar,
                                   int mode,
                                   SgdnnTensor_t output ,
                                   bool non_blocking = true);

/*
 *  OUTPUT  = COMPARISION ( input, other, output )
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input, other and output must be the same shape.
 *  3. 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
 */
tpu_status_t sgdnnComparision ( tpu_resource_t  stream,
                               SgdnnTensor_t input,
                               SgdnnTensor_t other,
                               int mode,
                               SgdnnTensor_t output ,
                               bool non_blocking = true);
/*
 *  OUTPUT  = COMPARISION ( input, other, output )
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. input and output must be the same shape.
 *  3. input and other must be the same dim.
 *  4. 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
 */
tpu_status_t sgdnnComparisionBcast ( tpu_resource_t  stream,
                                    SgdnnTensor_t input,
                                    SgdnnTensor_t other,
                                    int mode,
                                    SgdnnTensor_t output ,
                                    bool non_blocking = true);
/*
 *  OUTPUT  = COMPARISION ( input, other, output )
 *  Note:
 *  1. input, other and output only support in dtype.
 *  2. one of input and other must be a scalar.
 *  3. input and output must be in the same shape.
 *  4. 0 equal, 1 not equal, 2 greater, 3 greater or equal, 4 less than, 5 less than or equal
 */
tpu_status_t sgdnnComparisionC ( tpu_resource_t  stream,
                                SgdnnTensor_t input,
                                float scalar,
                                int mode,
                                int scalar_pos,
                                SgdnnTensor_t output ,
                                bool non_blocking = true);

/*
 * OUTPUT = MINIMUMC(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16, BF16 and INT32
 * 2. INPUT and OUTPUT must be contiguous
 */
tpu_status_t sgdnnMinimumC ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = MINIMUM(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. The shapes of INPUT ,OTHER and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnMinimum ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = MINIMUMBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
tpu_status_t sgdnnMinimumBcast ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);
/*
 * OUTPUT = MAXIMUMC(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16, BF16 and INT32
 * 2. INPUT and OUTPUT must be contiguous
 */
tpu_status_t sgdnnMaximumC ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = MAXIMUM(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. The shapes of INPUT ,OTHER and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnMaximum ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = MINIMUMBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
tpu_status_t sgdnnMaximumBcast ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = ATAN2C(SCALAR,OTHER)
 * Note:
 * 1. The data types of OTHER must be FP32 and data type of OUTPUT must be one of FP32, FP16 and F16
 * 2. The shapes of OTHER and OUTPUT must be the same
 * 3. OTHER, OUTPUT must be contiguous
 */
tpu_status_t sgdnnAtan2C ( tpu_resource_t  stream,
                        float scalar,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = ATAN2_C(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT must be FP32 and data type of OUTPUT must be one of FP32, FP16 and F16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT, OUTPUT must be contiguous
 */
tpu_status_t sgdnnAtan2_C ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = ATAN2(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT and OTHER must be FP32 and data type of OUTPUT must be one of FP32, FP16 and F16
 * 2. The shapes of INPUT, OTHER and OUTPUT must be the same
 * 3. INPUT, OTHER, OUTPUT must be contiguous
 */
tpu_status_t sgdnnAtan2 ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = ATAN2BCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT and OTHER must be FP32 and data type of OUTPUT must be one of FP32, FP16 and F16
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
tpu_status_t sgdnnAtan2Bcast ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

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
tpu_status_t sgdnnLnMm ( tpu_resource_t  stream,
                          SgdnnTensor_t input,
                          SgdnnTensor_t w,
                          SgdnnTensor_t b,
                          SgdnnTensor_t gamma,
                          SgdnnTensor_t beta,
                          float eps,
                          SgdnnTensor_t mean,
                          SgdnnTensor_t rstd,
                          SgdnnTensor_t output ,
                          bool non_blocking = true);

/*
 * OUTPUT = FMAXC(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16, BF16 and INT32
 * 2. INPUT and OUTPUT must be contiguous
 */
tpu_status_t sgdnnFmaxC ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = FMAX(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. The shapes of INPUT ,OTHER and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnFmax ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = FMAXBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
tpu_status_t sgdnnFmaxBcast ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = FMINC(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16, BF16 and INT32
 * 2. INPUT and OUTPUT must be contiguous
 */
tpu_status_t sgdnnFminC ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = FMIN(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. The shapes of INPUT ,OTHER and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnFmin ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = FMINBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT ,OTHER and OUTPUT must be the same and one of FP32, FP16 ,BF16 and INT32
 * 2. INPUT, OTHER, OUTPUT must be contiguous
 */
tpu_status_t sgdnnFminBcast ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

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
tpu_status_t sgdnnAddLnMm ( tpu_resource_t  stream,
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
                          SgdnnTensor_t output ,
                          bool non_blocking = true);

/*
 * OUTPUT = POWC(INPUT, EXPONENT)
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32, FP16 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 * 4. EXPONENT is a single float number.
*/
tpu_status_t sgdnnPowC ( tpu_resource_t  stream,
                      SgdnnTensor_t self,
                      float scalar,
                      SgdnnTensor_t out ,
                      bool non_blocking = true);

/*
 * OUTPUT = POWC(EXPONENT, INPUT)
 * 1. The data types of INPUT and OUTPUT must be one of FP32, FP16, BF16 and INT32
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT and OUTPUT must be contiguous
 * 4. EXPONENT is a single float number.
*/
tpu_status_t sgdnnCPow ( tpu_resource_t  stream,
                      SgdnnTensor_t self,
                      float scalar,
                      SgdnnTensor_t out,
                      bool non_blocking = true);

/*
 *  OUTPUT  = HYPOT ( input, other )
 *  Note:
 *  1. input, other and output only support float32.
 *  2. input, other and output must be the same shape.
 *  3. input, other and output must be contiguous
 */
tpu_status_t sgdnnHypot ( tpu_resource_t  stream,
                         SgdnnTensor_t input,
                         SgdnnTensor_t other,
                         SgdnnTensor_t output ,
                         bool non_blocking = true);

/*
 *  OUTPUT  = HYPOT ( input, other )
 *  Note:
 *  1. input, other and output only support float32.
 *  2. input and other must be the same dim.
 *  3. input, other and output must be contiguous
 */
tpu_status_t sgdnnHypotBcast ( tpu_resource_t  stream,
                              SgdnnTensor_t input,
                              SgdnnTensor_t other,
                              SgdnnTensor_t output ,
                              bool non_blocking = true);
/*
 *  OUTPUT  = HYPOT ( input, other )
 *  Note:
 *  1. input, other and output only support float32.
 *  2. input and output must be the same shape.
 *  3. input and output must be contiguous
 */
tpu_status_t sgdnnHypotC ( tpu_resource_t  stream,
                          SgdnnTensor_t input,
                          float scalar,
                          SgdnnTensor_t output ,
                          bool non_blocking = true);

/*
 * OUTPUT = NEXTAFTERC(SCALAR,OTHER)
 * Note:
 * 1. The data types of OTHER and OUTPUT must be the same and one of FP32 and BF16
 * 2. The shapes of OTHER and OUTPUT must be the same
 * 3. OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnNextafterC ( tpu_resource_t  stream,
                        float scalar,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = NEXTAFTER_C(INPUT,SCALAR)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. OUTPUT must be contiguous
 */
tpu_status_t sgdnnNextafter_C ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        float scalar,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = NEXTAFTER(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32 and BF16
 * 2. The shapes of INPUT and OUTPUT must be the same
 * 3. INPUT ,OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnNextafter ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 * OUTPUT = NEXTAFTERBCAST(INPUT,OTHER)
 * Note:
 * 1. The data types of INPUT and OUTPUT must be the same and one of FP32 and BF16
 * 2. INPUT ,OTHER and OUTPUT must be contiguous
 */
tpu_status_t sgdnnNextafterBcast ( tpu_resource_t  stream,
                        SgdnnTensor_t input,
                        SgdnnTensor_t other,
                        SgdnnTensor_t output ,
                        bool non_blocking = true);

/*
 *  OUTPUT  = PAD ( INPUT, PAD, VALUE )
 *  Note:
 *  1. input and output must be the same dim.
 *  2. input and output must be the same dtype.
 *  3. input and output must be contiguous
 *  4. pad size must less than or equal to twice the number of the input dimensions
 */
tpu_status_t sgdnnPad ( tpu_resource_t  stream,
                       SgdnnTensor_t input,
                       int *pad,
                       int pad_size,
                       float value,
                       int mode,
                       bool pad3d,
                       SgdnnTensor_t output ,
                       bool non_blocking = true);

/*
*  Found_inf = inf in input ? 1 : 0;
*  Input     = inv_scale * Input;
*  Note:
*  1. the dim of found_inf = 1
*/
tpu_status_t sgdnnInfCheckAndUnscale( tpu_resource_t  stream,
                                    std::vector<SgdnnTensor_t>& input,
                                    SgdnnTensor_t found_inf,
                                    float inv_scale ,
                                    bool non_blocking = true);
#if defined(__cplusplus)
}
#endif

#endif /* SGDNN_API_H */
