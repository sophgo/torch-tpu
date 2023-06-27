#ifndef SGDNN_API_H
#define SGDNN_API_H

#include "bmlib_runtime.h"
#include "common.h"
#include "common_def.h"
#include "sg_api_struct.h"
#include <string>

#if defined(__cplusplus)
extern "C" {
#endif

void get_coeff_data ( const std::string& modelpath, u64 addr_offset, int coeff_size, float* coeff );

void tpu_module_init ( bm_handle_t handle );
void tpu_module_deinit ( bm_handle_t handle );

typedef enum {
  OP_ELTWISE_PRODUCT     = 0,
  OP_ELTWISE_COEF_ADD    = 1,
  OP_ELTWISE_MAX         = 2,
} EltwiseOpMode_t;

typedef enum {
  OP_BINARY_ADD          = 0,
  OP_BINARY_SUB          = 1,
  OP_BINARY_MUL          = 2,
  OP_BINARY_DIV          = 3,
  OP_BINARY_MAX          = 4,
  OP_BINARY_MIN          = 10000,
  OP_BINARY_GT           = 10001,
  OP_BINARY_GE           = 10002,
  OP_BINARY_LT           = 10003,
  OP_BINARY_LE           = 10004,
  OP_BINARY_EQ           = 10005,
  OP_BINARY_NE           = 10006,
  OP_BINARY_SQUARED_DIFF = 10007,
  OP_BINARY_FLOOR_MOD    = 10008,
  OP_BINARY_FLOOR_DIV    = 10009
} BinaryOpMode_t;

typedef enum {
  Pooling_MAX = 0,
  Pooling_AVERAGE = 1,
} PoolingMode_t;

typedef enum {
  BatchNorm_Spatial = 0,
  BatchNorm_Per_Layer = 1,
  BatchNorm_Per_Activation = 2,
} BatchNormMode_t;

typedef enum {
  Activation_Sigmoid        = 0,
  Activation_Relu           = 1,
  Activation_Tanh           = 2,
  Activation_Elu            = 3,
  Activation_Gelu           = 4,
  Activation_Swish          = 5,
} ActivationMode_t;

typedef enum {
  Not_Propagate_Nan = 0,
  Propagate_Nan = 1,
} NanPropagation_t;

typedef enum {
  Softmax_Per_Instance = 0,
  Softmax_Per_Chanel = 1,
} SoftmaxMode_t;

typedef enum {
  None_Reduction = 0,
  Mean_Reduction = 1,
  Sum_Reduction = 2,
} CrossEntropyMode_t;

typedef enum {
  Reduction_Mean = 0,
  Reduction_Sum = 1,
} ReductionMode_t;

typedef enum {
  Reorder_To_32ic = 0,
  Reorder_To_32oc = 1,
} ConvWeightReorderMode_t;

typedef struct {
  int         dtype;
  int         ndims;
  int         shape[FW_MAX_SHAPE_DIMS];
  int         stride[FW_MAX_SHAPE_DIMS];
  int         format;
} TensorDescriptor_t;

typedef struct {
  int     oc;
  int     ic;
  int     kh;
  int     kw;
  sg_data_type_t  dtype;
} FilterDescriptor_t;

typedef struct {
  int     pad_h;
  int     pad_w;
  int     stride_h;//u
  int     stride_w;//v
  int     dilation_h;
  int     dilation_w;
  int     groups;
  sg_data_type_t  computeType;
} ConvolutionDescriptor_t;

typedef struct {
  ActivationMode_t    mode;
  NanPropagation_t    NanOpt;
  double              coef;//upper_limit when clipped
} ActivationDescriptor_t;

typedef struct {
  int             kh;
  int             kw;
  int             pad_h;
  int             pad_w;
  int             stride_h;
  int             stride_w;
  PoolingMode_t   mode;
} PoolingDescriptor_t;

bm_status_t sgdnn_conv_forward (
bm_handle_t                     handle,
const void                     *alpha,
const TensorDescriptor_t        xDesc,
const void                     *x,
const FilterDescriptor_t        wDesc,
const void                     *w,
const TensorDescriptor_t        bDesc,
const void                     *b,
const ConvolutionDescriptor_t   convDesc,
//ConvlutionFwdAlgo_t             algo,
//void                           *workspace,
//size_t                          workSpaceSizeInBytes,
const void                     *beta,
const TensorDescriptor_t        yDesc,
void                           *y );

bm_status_t sgdnn_conv_backward (
bm_handle_t                     handle,
const void                     *alpha,
const void                     *beta,
const TensorDescriptor_t        xDesc,
const void                     *x,
void                           *dx,
const FilterDescriptor_t        wDesc,
const void                     *w,
void                           *dw,
const TensorDescriptor_t        dbDesc,
void                           *db,
const TensorDescriptor_t        dyDesc,
const void                     *dy,
const ConvolutionDescriptor_t   convDesc,
bool                            dx_enable,
bool                            dw_enable,
bool                            db_enable );

bm_status_t sgdnn_batchnorm_forward (
bm_handle_t                      handle,
BatchNormMode_t                  mode,
const void                      *alpha,
const void                      *beta,
const TensorDescriptor_t         xDesc,
const void                      *x,
const TensorDescriptor_t         yDesc,
void                            *y,
const TensorDescriptor_t         bnScaleBiasMeanVarDesc,
const void                      *bnScale,
const void                      *bnBias,
double                           exponentialAverageFactor,
void                            *resultRunningMean,
void                            *resultRunningVariance,
double                           epsilon,
void                            *resultSaveMean,
void                            *resultSaveInvVariance );

bm_status_t sgdnn_batchnorm_backward (
bm_handle_t                      handle,
BatchNormMode_t                  mode,
const void                      *alphaDataDiff,
const void                      *betaDataDiff,
const void                      *alphaParamDiff,
const void                      *betaParamDiff,
const TensorDescriptor_t         xDesc,
const void                      *x,
const TensorDescriptor_t         dyDesc,
const void                      *dy,
const TensorDescriptor_t         dxDesc,
void                            *dx,
const TensorDescriptor_t         bnScaleBiasDiffDesc,
const void                      *bnScale,
void                            *resultBnScaleDiff,
void                            *resultBnBiasDiff,
double                           epsilon,
const void                      *savedMean,
const void                      *savedInvVariance,
bool                             dx_enable,
bool                             dw_enable,
bool                             db_enable );

bm_status_t sgdnn_activation_forward (
bm_handle_t                     handle,
ActivationDescriptor_t          activationDesc,
const void                     *alpha,
const TensorDescriptor_t        xDesc,
const void                     *x,
const void                     *beta,
const TensorDescriptor_t        yDesc,
void                           *y );

bm_status_t sgdnn_activation_backward (
bm_handle_t                      handle,
ActivationDescriptor_t           activationDesc,
const void                      *alpha,
const TensorDescriptor_t         yDesc,
const void                      *y,
const TensorDescriptor_t         dyDesc,
const void                      *dy,
const TensorDescriptor_t         xDesc,
const void                      *x,
const void                      *beta,
const TensorDescriptor_t         dxDesc,
void                            *dx );

bm_status_t sgdnn_dtype_convert (
bm_handle_t                      handle,
const TensorDescriptor_t         xDesc,
const void                      *xData,
const TensorDescriptor_t         yDesc,
const void                      *yData,
sg_round_mode_t                  round_mode );

bm_status_t sgdnn_conv_weight_reorder (
bm_handle_t                      handle,
const TensorDescriptor_t         xDesc,
const void                      *xData,
const TensorDescriptor_t         yDesc,
const void                      *yData,
ConvWeightReorderMode_t          reorder_mode );

bm_status_t sgdnn_binary (
bm_handle_t                 handle,
const TensorDescriptor_t    aDesc,
const void*                 A,
const TensorDescriptor_t    bDesc,
const void*                 B,
const TensorDescriptor_t    cDesc,
void*                       C,
BinaryOpMode_t              opTensorDesc );

bm_status_t sgdnn_batch_matmul (
bm_handle_t                      handle,
const TensorDescriptor_t         LDesc,
const void                      *L,
const TensorDescriptor_t         RDesc,
const void                      *R,
const TensorDescriptor_t         YDesc,
void                            *Y,
int                              L_transpose,
int                              R_transpose );

bm_status_t sgdnn_softmax_forward (
bm_handle_t                      handle,
//SoftmaxMode_t                    softmax_mode,
int                              dim,
const void                      *alpha,
const TensorDescriptor_t         xDesc,
const void                      *x,
const void                      *beta,
const TensorDescriptor_t         yDesc,
void                            *y );

bm_status_t sgdnn_softmax_backward (
bm_handle_t                      handle,
int                              dim,
const TensorDescriptor_t         yDesc,
const void                      *y,
const TensorDescriptor_t         dyDesc,
const void                      *dy,
const TensorDescriptor_t         dxDesc,
void                            *dx );

bm_status_t sgdnn_transpose (
bm_handle_t                      handle,
const TensorDescriptor_t         xDesc,
const void                      *xData,
const TensorDescriptor_t         yDesc,
void                            *yData );

bm_status_t sgdnn_pooling_forward (
bm_handle_t                 handle,
const PoolingDescriptor_t   poolingDesc,
const void                 *alpha,
const TensorDescriptor_t    xDesc,
const void                 *x,
const void                 *beta,
const TensorDescriptor_t    yDesc,
void                       *y );

bm_status_t sgdnn_pooling_backward (
bm_handle_t                 handle,
const PoolingDescriptor_t   poolingDesc,
const void                 *alpha,
const TensorDescriptor_t    yDesc,
const void                 *y,
const TensorDescriptor_t    dyDesc,
const void                 *dy,
const TensorDescriptor_t    xDesc,
const void                 *x,
const void                 *beta,
const TensorDescriptor_t    dxDesc,
void                       *dx );

bm_status_t sgdnn_gelu_backward (
bm_handle_t                     handle,
const TensorDescriptor_t        xDesc,
const void                     *x,
const TensorDescriptor_t        dyDesc,
const void                     *dy,
const TensorDescriptor_t        dxDesc,
void                           *dx );

bm_status_t sgdnn_strided_copy (
bm_handle_t                     handle,
const TensorDescriptor_t        srcDesc,
const void                      *src,
const TensorDescriptor_t        dstDesc,
void                      *dst );

bm_status_t sgdnn_where (
bm_handle_t                     handle,
const TensorDescriptor_t        condDesc,
const void                     *cond,
const TensorDescriptor_t        selfDesc,
const void                     *self,
const TensorDescriptor_t        otherDesc,
const void                     *other,
const TensorDescriptor_t        outDesc,
void                           *out );

bm_status_t sgdnn_concat (
bm_handle_t                     handle,
const TensorDescriptor_t       *inputDescs,
const void * const *            inputs,
int                             input_num,
const TensorDescriptor_t        outputDesc,
void                           *output,
int                             concat_dim );

bm_status_t sgdnn_reduce (
bm_handle_t                     handle,
const TensorDescriptor_t        xDesc,
const void                     *x,
const TensorDescriptor_t        yDesc,
void                           *y,
int                             reduce_dim_start,
int                             reduce_dim_end,
int                             keep_dim,
ReductionMode_t                 mode );

bm_status_t sgdnn_index_select (
bm_handle_t                     handle,
const TensorDescriptor_t        tableDesc,
const void                      *table,
const TensorDescriptor_t        indexDesc,
const void                      *index,
const TensorDescriptor_t        outDesc,
void                            *out,
int                             dim );

bm_status_t sgdnn_const_fill (
bm_handle_t                     handle,
const TensorDescriptor_t        srcDesc,
void                           *src,
const void*                     fill_value );

bm_status_t sgdnn_sqrt (
bm_handle_t                     handle,
const TensorDescriptor_t        inputDesc,
const void                     *input,
const TensorDescriptor_t        outputDesc,
void                           *output );

bm_status_t sgdnn_addcdiv (
bm_handle_t                     handle,
const TensorDescriptor_t        inputDesc,
const void                     *input,
const TensorDescriptor_t        tensor1Desc,
const void                     *tensor1,
const TensorDescriptor_t        tensor2Desc,
const void                     *tensor2,
const TensorDescriptor_t        outputDesc,
void                           *output,
double                          value );

bm_status_t sgdnn_addcmul (
bm_handle_t                     handle,
const TensorDescriptor_t        inputDesc,
const void                     *input,
const TensorDescriptor_t        tensor1Desc,
const void                     *tensor1,
const TensorDescriptor_t        tensor2Desc,
const void                     *tensor2,
const TensorDescriptor_t        outputDesc,
void                           *output,
double                          value );

/*
reference:
https://pytorch.org/docs/1.13/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropy#torch.nn.CrossEntropyLoss
- in:
        shape = [N, C]
-target:
        shape = [N]
- weight: default have no value
        shape = in.shape[-1] = [C]
        weight * log(softmax(in))
- reduction: default= Mean
        None : shape = target.shape = [N]
        Mean : shape = [1]
        Sum  : shape = [1]
- label_smoothing: [0, 1], default = 0
        target = one_hot(target_index)
        target[i == target_index] = 1 - label_smoothing, others = label_smothing/(len(in)-1)
 examples:
    target_index                            : 3
    label_smothing                          : 0.3
    target  = one_hot(target_index)         : [0, 0, 0, 1]
    target <- label smoothing               : [0.1, 0.1, 0.1, 0.7]
**/
bm_status_t sgdnn_cross_entropy_forward (
bm_handle_t                      handle,
const TensorDescriptor_t         inDesc,
const void                      *in,
const TensorDescriptor_t         targetDesc,
const void                      *target,
const TensorDescriptor_t         weightDesc,
const void                      *weight,
const TensorDescriptor_t         outDesc,
void                            *out,
bool                             has_weight,
CrossEntropyMode_t               reduction,
int                              ignore_index,
double                           label_smoothing
);

bm_status_t sgdnn_cross_entropy_backward (
bm_handle_t                      handle,
const TensorDescriptor_t         targetDesc,
const void                      *target,
const TensorDescriptor_t         inputDesc,
const void                      *input,
const TensorDescriptor_t         weightDesc,
const void                      *weight,
const TensorDescriptor_t         gradoutDesc,
const void                      *gradout,
const TensorDescriptor_t         gradinDesc,
void                            *gradin,
CrossEntropyMode_t               reduction,
int                              ignore_index,
double                           label_smoothing,
bool                             has_weight );

bm_status_t sgdnn_matmul (
bm_handle_t                      handle,
const TensorDescriptor_t         LDesc,
const void                      *L,
const TensorDescriptor_t         RDesc,
const void                      *R,
const TensorDescriptor_t         BDesc,
const void                      *B,
const TensorDescriptor_t         YDesc,
void                            *Y,
int                              L_transpose,
int                              R_transpose );
/**
 * refrence：
 * https://pytorch.org/docs/1.13/generated/torch.nn.Embedding.html?highlight=embedding#torch.nn.Embedding
 * - padding_idx:
 *  #TODO
 * - scale_grad_by_freq: default False
 * */
bm_status_t sgdnn_embedding_dense_backward (
bm_handle_t                       handle,
const TensorDescriptor_t          gradoutDesc,
const void                       *gradout,
const TensorDescriptor_t          indicesDesc,
const void                       *indices,
const TensorDescriptor_t          outDesc,
void                             *out,
int                               padding_idx,
bool                              scale_grad_by_freq );

bm_status_t sgdnn_scale_add (
bm_handle_t                     handle,
const TensorDescriptor_t        inputDesc,
const void                     *input,
const TensorDescriptor_t        otherDesc,
const void                     *other,
const TensorDescriptor_t        outputDesc,
void                           *output,
double                          value );

#if defined(__cplusplus)
}
#endif

#endif /* SGDNN_API_H */

