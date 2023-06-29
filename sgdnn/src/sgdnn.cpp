#include "sgdnn_api.h"
#include "sgdnn_helpers.h"
#include <assert.h>
#include <map>
#include <memory>
#include <string.h>
#include "kernel_module_data.h"
#include "bmodel.hpp"

static inline int dtype_size ( sg_data_type_t dtype ) {
  int size = 1;
  if ( dtype == SG_DTYPE_INT8 || dtype == SG_DTYPE_UINT8 ) size = 1;
  else if ( dtype == SG_DTYPE_INT16 || dtype == SG_DTYPE_UINT16 ||
            dtype == SG_DTYPE_FP16 || dtype == SG_DTYPE_BFP16 )
    size = 2;
  else if ( dtype == SG_DTYPE_FP32 || dtype == SG_DTYPE_INT32 || dtype == SG_DTYPE_UINT32 )
    size = 4;
  return size;
}

static inline sg_active_type_t tpu_active_type_convert ( ActivationMode_t active_type ) {
  sg_active_type_t atype = ACTIVE_RELU;
  switch ( active_type ) {
  case Activation_Sigmoid:    atype = ACTIVE_SIGMOID;     break;
  case Activation_Relu:       atype = ACTIVE_RELU;        break;
  case Activation_Tanh:       atype = ACTIVE_TANH;        break;
  case Activation_Elu:        atype = ACTIVE_ELU;         break;
  case Activation_Gelu:       atype = ACTIVE_GELU;        break;
  case Activation_Swish:      atype = ACTIVE_SWISH;       break;
  default:
    assert ( 0 );
    break;
  }
  return atype;
}

static inline sg_binary_type_t tpu_binary_type_convert ( BinaryOpMode_t binary_type ) {
  sg_binary_type_t optype = BINARY_ADD;
  switch ( binary_type ) {
  case OP_BINARY_ADD:          optype = BINARY_ADD;           break;
  case OP_BINARY_SUB:          optype = BINARY_SUB;           break;
  case OP_BINARY_MUL:          optype = BINARY_MUL;           break;
  case OP_BINARY_DIV:          optype = BINARY_DIV;           break;
  case OP_BINARY_MAX:          optype = BINARY_MAX;           break;
  case OP_BINARY_MIN:          optype = BINARY_MIN;           break;
  case OP_BINARY_GT:           optype = BINARY_GT;            break;
  case OP_BINARY_GE:           optype = BINARY_GE;            break;
  case OP_BINARY_LT:           optype = BINARY_LT;            break;
  case OP_BINARY_LE:           optype = BINARY_LE;            break;
  case OP_BINARY_EQ:           optype = BINARY_EQ;            break;
  case OP_BINARY_NE:           optype = BINARY_NE;            break;
  case OP_BINARY_SQUARED_DIFF: optype = BINARY_SQUARED_DIFF;  break;
  case OP_BINARY_FLOOR_MOD:    optype = BINARY_FLOOR_MOD;     break;
  case OP_BINARY_FLOOR_DIV:    optype = BINARY_FLOOR_DIV;     break;
  default:
    assert ( 0 );
    break;
  }
  return optype;
}

#define ASSERT_SAME_DIMS(A, B)            \
  assert(A.ndims == B.ndims);             \

#define ASSERT_SAME_SHAPE(A, B)           \
  assert(A.ndims == B.ndims);             \
  for (int dim = 0; dim < A.ndims; dim++) \
    assert(A.shape[dim] == B.shape[dim]); \

static std::map<bm_handle_t, tpu_kernel_module_t> tpu_kernel_module;

void tpu_module_init ( bm_handle_t handle ) {
  if ( tpu_kernel_module.find ( handle ) != tpu_kernel_module.end() ) return;
  const unsigned int *p = kernel_module_data;
  size_t length = sizeof ( kernel_module_data );
  tpu_kernel_module_t tpu_module = tpu_kernel_load_module ( handle, ( const char * ) p, length );
  tpu_kernel_module.insert ( std::pair<bm_handle_t, tpu_kernel_module_t> ( handle, tpu_module ) );
}

void tpu_module_deinit ( bm_handle_t handle ) {
  if ( tpu_kernel_module.find ( handle ) == tpu_kernel_module.end() ) return;
  assert ( tpu_kernel_module.erase ( handle ) );
}

static void sgdnn_tpu_kernel_launch (
bm_handle_t     handle,
const char*     func_name,
const void*     api,
size_t          api_size ) {
  tpu_kernel_function_t func_id;
  tpu_kernel_module_t tpu_module = tpu_kernel_module[handle];
  func_id = tpu_kernel_get_function ( handle, tpu_module, func_name );
  bm_status_t ret = tpu_kernel_launch ( handle, func_id, ( void* ) api, api_size );
  if ( ret != BM_SUCCESS ) throw ( "tpu_kernel_launch failed" );
}

#define SP(D, T) (std::shared_ptr<T>((D), std::default_delete<T []>()))
void get_coeff_data ( const std::string& modelpath, u64 addr_offset, int coeff_size, float* coeff ) {
  bmodel::ModelCtx model_ctx ( modelpath );
  //just assume resnet50 has one net
  auto params = model_ctx.model()->net()->Get ( 0 )->parameter();
  //just assume resnet50 has one netparam
  const bmodel::CoeffMem* coeff_mem = params->Get ( 0 )->coeff_mem();
#define COEFF_BLK_SIZE 0x1000000
  u8* data = new u8[COEFF_BLK_SIZE];
  auto data_sp = SP ( data, u8 );
  u64 left_size = coeff_size;
  u64 offset = 0;
  while ( left_size > 0 ) {
    u64 data_size = ( left_size >= COEFF_BLK_SIZE ? COEFF_BLK_SIZE : left_size );
    model_ctx.read_binary ( coeff_mem->binary_coeff(), offset, data, data_size );
    memcpy ( coeff + offset * sizeof ( float ), data, data_size * sizeof ( float ) );
    offset += data_size;
    left_size -= data_size;
  }
}

void set_coeff_data ( const std::string& modelpath, u64 addr_offset, int coeff_size, float* coeff ) {}

//dstValue = alpha[0]*result + beta[0]*priorDstValue
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
void                           *y ) {
  assert ( xDesc.ndims == 4 && yDesc.ndims == 4 );
  int n = xDesc.shape[0];
  int ic = xDesc.shape[1];
  int ih = xDesc.shape[2];
  int iw = xDesc.shape[3];
  int oc = wDesc.oc;
  assert ( xDesc.shape[1] == wDesc.ic );
  int kh = wDesc.kh;
  int kw = wDesc.kw;
  int pad_h = convDesc.pad_h;
  int pad_w = convDesc.pad_w;
  int stride_h = convDesc.stride_h;
  int stride_w = convDesc.stride_w;
  int dh = convDesc.dilation_h;
  int dw = convDesc.dilation_w;
  int groups = convDesc.groups;
  float alpha_ = ( ( float* ) alpha ) [0];
  assert ( alpha_ == 1.0f );
  UNUSED ( alpha_ );
  float beta_ = ( ( float* ) beta ) [0];
  assert ( beta_ == 0.0f || beta_ == 1.0f );
  bool result_add = beta_ == 1.0f;
  sg_data_type_t idtype = ( sg_data_type_t ) ( xDesc.dtype );
  sg_data_type_t wdtype = ( sg_data_type_t ) ( wDesc.dtype );
  sg_data_type_t bdtype = ( sg_data_type_t ) ( bDesc.dtype );
  sg_data_type_t odtype = ( sg_data_type_t ) ( yDesc.dtype );
  assert ( bdtype == SG_DTYPE_FP32 );
  sg_data_type_t compute_type = ( sg_data_type_t ) ( convDesc.computeType );
  if ( compute_type == SG_DTYPE_FP32 ) {
    assert ( idtype == SG_DTYPE_FP32 &&
             wdtype == SG_DTYPE_FP32 &&
             odtype == SG_DTYPE_FP32 );
    sg_api_conv_forward_t api = {
      ( unsigned long long ) x,
      ( unsigned long long ) w,
      ( unsigned long long ) b,
      ( unsigned long long ) y,
      {n, ic, ih, iw},
      groups,
      oc,
      {kh, kw},
      {stride_h, stride_w},
      {dh, dw},
      {pad_h, pad_h, pad_w, pad_w},//pad
      b != NULL ? 1 : 0,//has_bias?
      0,//if_relu
      0,//upper_limit
      result_add ? 1 : 0,
      idtype,
      odtype
    };
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_conv_forward", &api, sizeof ( api ) );
  } else if ( compute_type == SG_DTYPE_FP16 ) {
    if ( idtype == SG_DTYPE_FP32 ) {
      assert ( wdtype == SG_DTYPE_FP32 && odtype == SG_DTYPE_FP32 );
      int dtype_size = 2;
      bm_device_mem_t x_fp16, w_fp16, w_32ic_fp16;
      u64 x_fp16_size = ( u64 ) n * ic * ih * iw * dtype_size;
      u64 w_fp16_size = ( u64 ) oc * ic * kh * kw * dtype_size;
      u64 w_32ic_fp16_size = ( u64 ) oc * ALIGN ( ic, 32 ) * kh * kw * dtype_size;
      DEVICE_MEM_NEW_BUFFER ( handle, x_fp16, x_fp16_size );
      DEVICE_MEM_NEW_BUFFER ( handle, w_fp16, w_fp16_size );
      DEVICE_MEM_NEW_BUFFER ( handle, w_32ic_fp16, w_32ic_fp16_size );
      sg_api_dtype_convert_t cast_x_api;
      cast_x_api.input_global_addr = ( unsigned long long ) x;
      cast_x_api.output_global_addr = bm_mem_get_device_addr ( x_fp16 );
      memcpy ( cast_x_api.shape, xDesc.shape, xDesc.ndims * sizeof ( int ) );
      cast_x_api.dims = xDesc.ndims;
      cast_x_api.idtype = SG_DTYPE_FP32;
      cast_x_api.odtype = SG_DTYPE_FP16;
      cast_x_api.round_mode = SG_ROUND_EVEN;
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &cast_x_api, sizeof ( cast_x_api ) );
      sg_api_dtype_convert_t cast_w_api;
      cast_w_api.input_global_addr = ( unsigned long long ) w;
      cast_w_api.output_global_addr = bm_mem_get_device_addr ( w_fp16 );
      int w_shape[4] = {oc, ic, kh, kw};
      memcpy ( cast_w_api.shape, w_shape, 4 * sizeof ( int ) );
      cast_w_api.dims = 4;
      cast_w_api.idtype = SG_DTYPE_FP32;
      cast_w_api.odtype = SG_DTYPE_FP16;
      cast_w_api.round_mode = SG_ROUND_EVEN;
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &cast_w_api, sizeof ( cast_w_api ) );
      sg_api_conv_weight_reorder_t conv_weight_reorder_api = {
        bm_mem_get_device_addr ( w_fp16 ),
        bm_mem_get_device_addr ( w_32ic_fp16 ),
        {oc, ic, kh, kw},
        Reorder_To_32ic
      };
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_conv_weight_reorder", &conv_weight_reorder_api, sizeof ( conv_weight_reorder_api ) );
      sg_api_conv_forward_t api = {
        bm_mem_get_device_addr ( x_fp16 ),
        bm_mem_get_device_addr ( w_32ic_fp16 ),
        ( unsigned long long ) b,
        ( unsigned long long ) y,
        {n, ic, ih, iw},
        groups,
        oc,
        {kh, kw},
        {stride_h, stride_w},
        {dh, dw},
        {pad_h, pad_h, pad_w, pad_w},//pad
        b != NULL ? 1 : 0,//has_bias?
        0,//if_relu
        0,//upper_limit
        result_add ? 1 : 0,
        SG_DTYPE_FP16,
        odtype
      };
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_conv_forward", &api, sizeof ( api ) );
      bm_free_device ( handle, x_fp16 );
      bm_free_device ( handle, w_fp16 );
      bm_free_device ( handle, w_32ic_fp16 );
    } else if ( idtype == SG_DTYPE_FP16 ) {
      assert ( wdtype == SG_DTYPE_FP16 );
      int dtype_size = 2;
      bm_device_mem_t w_32ic_fp16;
      u64 w_32ic_fp16_size = ( u64 ) oc * ALIGN ( ic, 32 ) * kh * kw * dtype_size;
      DEVICE_MEM_NEW_BUFFER ( handle, w_32ic_fp16, w_32ic_fp16_size );
      sg_api_conv_weight_reorder_t conv_weight_reorder_api = {
        ( unsigned long long ) w,
        bm_mem_get_device_addr ( w_32ic_fp16 ),
        {oc, ic, kh, kw},
        Reorder_To_32ic
      };
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_conv_weight_reorder", &conv_weight_reorder_api, sizeof ( conv_weight_reorder_api ) );
      sg_api_conv_forward_t api = {
        ( unsigned long long ) x,
        bm_mem_get_device_addr ( w_32ic_fp16 ),
        ( unsigned long long ) b,
        ( unsigned long long ) y,
        {n, ic, ih, iw},
        groups,
        oc,
        {kh, kw},
        {stride_h, stride_w},
        {dh, dw},
        {pad_h, pad_h, pad_w, pad_w},//pad
        b != NULL ? 1 : 0,//has_bias?
        0,//if_relu
        0,//upper_limit
        result_add ? 1 : 0,
        idtype,
        odtype
      };
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_conv_forward", &api, sizeof ( api ) );
      bm_free_device ( handle, w_32ic_fp16 );
    }
  }
  return BM_SUCCESS;
}

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
bool                            db_enable
) {
  assert ( xDesc.ndims == 4 && dyDesc.ndims == 4 );
  int n = xDesc.shape[0];
  int ic = xDesc.shape[1];
  int ih = xDesc.shape[2];
  int iw = xDesc.shape[3];
  int oh = dyDesc.shape[2];
  int ow = dyDesc.shape[3];
  int oc = wDesc.oc;
  assert ( xDesc.shape[1] == wDesc.ic );
  int kh = wDesc.kh;
  int kw = wDesc.kw;
  int pad_h = convDesc.pad_h;
  int pad_w = convDesc.pad_w;
  int stride_h = convDesc.stride_h;
  int stride_w = convDesc.stride_w;
  int dilation_h = convDesc.dilation_h;
  int dilation_w = convDesc.dilation_w;
  int groups = convDesc.groups;
  assert ( groups == 1 );
  sg_data_type_t idtype = ( sg_data_type_t ) ( xDesc.dtype );
  sg_data_type_t wdtype = ( sg_data_type_t ) ( wDesc.dtype );
  sg_data_type_t odtype = ( sg_data_type_t ) ( dyDesc.dtype );
  assert ( idtype == wdtype && wdtype == odtype );
  UNUSED ( wdtype );
  UNUSED ( odtype );
  // cal buffer size
  sg_data_type_t compute_type = ( sg_data_type_t ) ( convDesc.computeType );
  bool need_buffer = idtype == SG_DTYPE_FP16 || compute_type == SG_DTYPE_FP16;
  u64 weight_reorder_size = ALIGN ( oc, 32 ) * kh * kw * ic * dtype_size ( SG_DTYPE_FP16 );
  u64 grad_out_reorder_size = ALIGN ( n, 32 ) * oh * ow * oc * dtype_size ( SG_DTYPE_FP16 );
  u64 buffer_size = need_buffer ? sg_max ( weight_reorder_size, grad_out_reorder_size ) : 0; //use for weight reorder
  bm_device_mem_t buffer_mem;
  if ( buffer_size > 0 ) {
    DEVICE_MEM_NEW_BUFFER ( handle, buffer_mem, buffer_size );
  }
  if ( ( idtype == SG_DTYPE_FP32 && compute_type == SG_DTYPE_FP32 ) ||
       ( idtype == SG_DTYPE_FP16 && compute_type == SG_DTYPE_FP16 ) ) {
    if ( dx_enable || dw_enable || db_enable ) {
      sg_api_conv_backward_t api = {
        ( unsigned long long ) x,
        ( unsigned long long ) w,
        ( unsigned long long ) dy,
        ( unsigned long long ) dx,
        ( unsigned long long ) dw,
        ( unsigned long long ) db,
        bm_mem_get_device_addr ( buffer_mem ),
        {n, ic, ih, iw},//ishape
        {n, oc, oh, ow},//oshape
        groups,
        {kh, kw},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {pad_h, pad_h, pad_w, pad_w},
        dx_enable,
        dw_enable,
        db_enable,
        idtype
      };
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_conv_backward", &api, sizeof ( api ) );
    }
    if ( buffer_size > 0 ) {
      bm_free_device ( handle, buffer_mem );
    }
  } else if ( idtype == SG_DTYPE_FP32 && compute_type == SG_DTYPE_FP16 ) {
    int dtype_size = 2;
    bm_device_mem_t x_fp16, w_fp16, dy_fp16;
    bm_device_mem_t dx_fp16, dw_fp16, db_fp16;
    u64 x_fp16_size = ( u64 ) n * ic * ih * iw * dtype_size;
    u64 w_fp16_size = ( u64 ) oc * ic * kh * kw * dtype_size;
    u64 dy_fp16_size = ( u64 ) n * oc * oh * ow * dtype_size;
    u64 dx_fp16_size = x_fp16_size;
    u64 dw_fp16_size = w_fp16_size;
    u64 db_fp16_size = oc * dtype_size;
    if ( dx_enable || dw_enable || db_enable ) {
      DEVICE_MEM_NEW_BUFFER ( handle, x_fp16, x_fp16_size );
      DEVICE_MEM_NEW_BUFFER ( handle, w_fp16, w_fp16_size );
      DEVICE_MEM_NEW_BUFFER ( handle, dy_fp16, dy_fp16_size );
    }
    if ( dx_enable ) DEVICE_MEM_NEW_BUFFER ( handle, dx_fp16, dx_fp16_size );
    if ( dw_enable ) DEVICE_MEM_NEW_BUFFER ( handle, dw_fp16, dw_fp16_size );
    if ( db_enable ) DEVICE_MEM_NEW_BUFFER ( handle, db_fp16, db_fp16_size );
    if ( dx_enable || dw_enable || db_enable ) {
      sg_api_dtype_convert_t cast_x_api;
      cast_x_api.input_global_addr = ( unsigned long long ) x;
      cast_x_api.output_global_addr = bm_mem_get_device_addr ( x_fp16 );
      memcpy ( cast_x_api.shape, xDesc.shape, xDesc.ndims * sizeof ( int ) );
      cast_x_api.dims = xDesc.ndims;
      cast_x_api.idtype = SG_DTYPE_FP32;
      cast_x_api.odtype = SG_DTYPE_FP16;
      cast_x_api.round_mode = SG_ROUND_EVEN;
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &cast_x_api, sizeof ( cast_x_api ) );
      sg_api_dtype_convert_t cast_w_api;
      cast_w_api.input_global_addr = ( unsigned long long ) w;
      cast_w_api.output_global_addr = bm_mem_get_device_addr ( w_fp16 );
      int w_shape[4] = {oc, ic, kh, kw};
      memcpy ( cast_w_api.shape, w_shape, 4 * sizeof ( int ) );
      cast_w_api.dims = 4;
      cast_w_api.idtype = SG_DTYPE_FP32;
      cast_w_api.odtype = SG_DTYPE_FP16;
      cast_w_api.round_mode = SG_ROUND_EVEN;
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &cast_w_api, sizeof ( cast_w_api ) );
      sg_api_dtype_convert_t cast_dy_api;
      cast_dy_api.input_global_addr = ( unsigned long long ) dy;
      cast_dy_api.output_global_addr = bm_mem_get_device_addr ( dy_fp16 );
      memcpy ( cast_dy_api.shape, dyDesc.shape, dyDesc.ndims * sizeof ( int ) );
      cast_dy_api.dims = dyDesc.ndims;
      cast_dy_api.idtype = SG_DTYPE_FP32;
      cast_dy_api.odtype = SG_DTYPE_FP16;
      cast_dy_api.round_mode = SG_ROUND_EVEN;
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &cast_dy_api, sizeof ( cast_dy_api ) );
      sg_api_conv_backward_t api = {
        bm_mem_get_device_addr ( x_fp16 ),
        bm_mem_get_device_addr ( w_fp16 ),
        bm_mem_get_device_addr ( dy_fp16 ),
        bm_mem_get_device_addr ( dx_fp16 ),
        bm_mem_get_device_addr ( dw_fp16 ),
        bm_mem_get_device_addr ( db_fp16 ),
        bm_mem_get_device_addr ( buffer_mem ),
        {n, ic, ih, iw},//ishape
        {n, oc, oh, ow},//oshape
        groups,
        {kh, kw},
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {pad_h, pad_h, pad_w, pad_w},
        dx_enable,
        dw_enable,
        db_enable,
        SG_DTYPE_FP16
      };
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_conv_backward", &api, sizeof ( api ) );
    }
    if ( dx_enable ) {
      sg_api_dtype_convert_t cast_dx_api;
      cast_dx_api.input_global_addr = bm_mem_get_device_addr ( dx_fp16 );
      cast_dx_api.output_global_addr = ( unsigned long long ) dx;
      memcpy ( cast_dx_api.shape, xDesc.shape, xDesc.ndims * sizeof ( int ) );
      cast_dx_api.dims = xDesc.ndims;
      cast_dx_api.idtype = SG_DTYPE_FP16;
      cast_dx_api.odtype = SG_DTYPE_FP32;
      cast_dx_api.round_mode = SG_ROUND_EVEN;
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &cast_dx_api, sizeof ( cast_dx_api ) );
    }
    if ( dw_enable ) {
      sg_api_dtype_convert_t cast_dw_api;
      cast_dw_api.input_global_addr = bm_mem_get_device_addr ( dw_fp16 );
      cast_dw_api.output_global_addr = ( unsigned long long ) dw;
      int dw_shape[4] = {oc, ic, kh, kw};
      memcpy ( cast_dw_api.shape, dw_shape, 4 * sizeof ( int ) );
      cast_dw_api.dims = 4;
      cast_dw_api.idtype = SG_DTYPE_FP16;
      cast_dw_api.odtype = SG_DTYPE_FP32;
      cast_dw_api.round_mode = SG_ROUND_EVEN;
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &cast_dw_api, sizeof ( cast_dw_api ) );
    }
    if ( db_enable ) {
      sg_api_dtype_convert_t cast_db_api;
      cast_db_api.input_global_addr = bm_mem_get_device_addr ( db_fp16 );
      cast_db_api.output_global_addr = ( unsigned long long ) db;
      memcpy ( cast_db_api.shape, dbDesc.shape, dbDesc.ndims * sizeof ( int ) );
      cast_db_api.dims = dbDesc.ndims;
      cast_db_api.idtype = SG_DTYPE_FP16;
      cast_db_api.odtype = SG_DTYPE_FP32;
      cast_db_api.round_mode = SG_ROUND_EVEN;
      sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &cast_db_api, sizeof ( cast_db_api ) );
    }
    if ( buffer_size > 0 ) {
      bm_free_device ( handle, buffer_mem );
    }
    if ( dx_enable ) bm_free_device ( handle, dx_fp16 );
    if ( dw_enable ) bm_free_device ( handle, dw_fp16 );
    if ( db_enable ) bm_free_device ( handle, db_fp16 );
    if ( dx_enable || dw_enable || db_enable ) {
      bm_free_device ( handle, x_fp16 );
      bm_free_device ( handle, w_fp16 );
      bm_free_device ( handle, dy_fp16 );
    }
  } else {
    //not support input is FP16 but compute type is FP32
    assert ( 0 );
  }
  return BM_SUCCESS;
}

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
void                            *resultSaveInvVariance )
{
  float alpha_ = ( ( float* ) alpha ) [0];
  assert ( alpha_ == 1.0f );
  float beta_ = ( ( float* ) beta ) [0];
  assert ( beta_ == 0.0f || beta_ == 1.0f );
  int n, c, h, w;
  float eps = epsilon;
  sg_data_type_t idtype = ( sg_data_type_t ) ( xDesc.dtype );
  sg_data_type_t odtype = ( sg_data_type_t ) ( yDesc.dtype );
  if ( mode == BatchNorm_Spatial )
  {
    unsigned long long input        = ( unsigned long long ) x;
    unsigned long long weight       = ( unsigned long long ) bnScale;
    unsigned long long bias         = ( unsigned long long ) bnBias;
    unsigned long long running_mean = ( unsigned long long ) resultRunningMean;
    unsigned long long running_var  = ( unsigned long long ) resultRunningVariance;
    unsigned long long batch_mean   = ( unsigned long long ) resultSaveMean;
    unsigned long long batch_invstd = ( unsigned long long ) resultSaveInvVariance;
    unsigned long long output       = ( unsigned long long ) y;
    assert ( ( xDesc.ndims == 4 && yDesc.ndims == 4 ) || ( xDesc.ndims == 3 && yDesc.ndims == 3 ) );
    if ( xDesc.ndims == 4 )
    {
      n = xDesc.shape[0];
      c = xDesc.shape[1];
      h = xDesc.shape[2];
      w = xDesc.shape[3];
    }
    else if ( xDesc.ndims == 3 )
    {
      n = xDesc.shape[0];
      c = xDesc.shape[1];
      h = 1;
      w = xDesc.shape[2];
    }
    float momentum = exponentialAverageFactor;
    if ( bnScale != nullptr || bnBias != nullptr || resultRunningMean != nullptr || resultRunningVariance != nullptr )
    {
      sg_data_type_t wdtype = ( sg_data_type_t ) ( bnScaleBiasMeanVarDesc.dtype );
      assert ( idtype == wdtype && wdtype == odtype );
      assert ( bnScaleBiasMeanVarDesc.ndims == 1 );
      assert ( bnScaleBiasMeanVarDesc.shape[0] == xDesc.shape[1] );
    }
    sg_api_batchnorm_forward_t api;
    api.input_global_addr           = input;
    api.running_mean_global_addr    = running_mean;
    api.running_var_global_addr     = running_var;
    api.weight_global_addr          = weight;
    api.bias_global_addr            = bias;
    api.updated_mean_global_addr    = running_mean;
    api.updated_var_global_addr     = running_var;
    api.batch_mean_global_addr      = batch_mean;
    api.batch_invstd_global_addr    = batch_invstd;
    api.output_global_addr          = output;
    api.momentum                    = momentum;
    api.eps                         = eps;
    api.dtype                       = idtype;
    api.shape[0]                    = n;
    api.shape[1]                    = c;
    api.shape[2]                    = h;
    api.shape[3]                    = w;
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_batchnorm_forward_v2", &api, sizeof ( api ) );
    return BM_SUCCESS;
  }
  else if ( mode == BatchNorm_Per_Layer )
  {
    unsigned long long input   = ( unsigned long long ) x;
    unsigned long long weight  = ( unsigned long long ) bnScale;
    unsigned long long bias    = ( unsigned long long ) bnBias;
    unsigned long long mean    = ( unsigned long long ) resultSaveMean;
    unsigned long long rstd    = ( unsigned long long ) resultSaveInvVariance;
    unsigned long long output  = ( unsigned long long ) y;
    assert ( resultRunningMean == nullptr && resultRunningVariance == nullptr );
    int affine = 0, save_stat = 0;
    if ( resultSaveMean != nullptr && resultSaveInvVariance != nullptr )
    {
      save_stat = 1;
    }
    if ( bnScale != nullptr && bnBias != nullptr )
    {
      affine = 1;
      sg_data_type_t wdtype = ( sg_data_type_t ) ( bnScaleBiasMeanVarDesc.dtype );
      assert ( idtype == wdtype && wdtype == odtype );
    }
    int normalized_ndim = bnScaleBiasMeanVarDesc.ndims;
    int input_ndim = xDesc.ndims;
    int axis = input_ndim - normalized_ndim;
    sg_api_layernorm_forward_t api;
    api.input_global_addr   = input;
    api.weight_global_addr  = weight;
    api.bias_global_addr    = bias;
    api.output_global_addr  = output;
    api.mean_global_addr    = mean;
    api.rstd_global_addr    = rstd;
    api.dims                = input_ndim;
    api.axis                = axis;
    api.eps                 = eps;
    api.affine              = affine;
    api.save_stat           = save_stat;
    api.dtype               = idtype;
    for ( int i = 0; i < input_ndim; i++ ) {
      api.shape[i] = xDesc.shape[i];
    }
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_layernorm_forward", &api, sizeof ( api ) );
    return BM_SUCCESS;
  }
  return BM_ERR_NOFEATURE;
}

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
bool                             db_enable )
{
  unsigned long long grad_output   = ( unsigned long long ) dy;
  unsigned long long input         = ( unsigned long long ) x;
  unsigned long long weight        = ( unsigned long long ) bnScale;
  unsigned long long saved_mean    = ( unsigned long long ) savedMean;
  unsigned long long saved_invstd  = ( unsigned long long ) savedInvVariance;
  unsigned long long grad_input    = ( unsigned long long ) dx;
  unsigned long long grad_weight   = ( unsigned long long ) resultBnScaleDiff;
  unsigned long long grad_bias     = ( unsigned long long ) resultBnBiasDiff;
  float alpha_data = ( ( float* ) alphaDataDiff ) [0];
  assert ( alpha_data == 1.0f );
  float alpha_param = ( ( float* ) alphaParamDiff ) [0];
  assert ( alpha_param == 1.0f );
  float beta_data = ( ( float* ) betaDataDiff ) [0];
  assert ( beta_data == 0.0f );
  float beta_param = ( ( float* ) betaParamDiff ) [0];
  assert ( beta_param == 0.0f );
  int n, c, h, w;
  if ( mode == BatchNorm_Spatial )
  {
    assert ( dyDesc.ndims == 4 );
    assert ( xDesc.ndims == 4 );
    assert ( dxDesc.ndims == 4 );
    n = xDesc.shape[0];
    c = xDesc.shape[1];
    h = xDesc.shape[2];
    w = xDesc.shape[3];
  }
  else if ( mode == BatchNorm_Per_Layer )
  {
    assert ( dyDesc.ndims == 3 );
    assert ( xDesc.ndims == 3 );
    if ( dx_enable ) assert ( dxDesc.ndims == 3 );
    n = xDesc.shape[0];
    c = xDesc.shape[1];
    h = 1;
    w = xDesc.shape[2];
  }
  else
  {
    assert ( 0 );
  }
  assert ( bnScaleBiasDiffDesc.ndims == 1 );
  sg_data_type_t dydtype = ( sg_data_type_t ) ( dyDesc.dtype );
  sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
  sg_data_type_t dxdtype = ( sg_data_type_t ) ( dxDesc.dtype );
  sg_data_type_t wdtype = ( sg_data_type_t ) ( bnScaleBiasDiffDesc.dtype );
  // if dtype is fp16, it will convert to fp32 in local
  assert ( xdtype == dydtype );
  if ( dx_enable )              { assert ( xdtype == dxdtype );}
  if ( dw_enable || db_enable ) { assert ( xdtype == wdtype );}
  sg_api_batchnorm_backward_t api = {
    grad_output,
    input,
    weight,
    saved_mean,
    saved_invstd,
    grad_input,
    grad_weight,
    grad_bias,
    {n, c, h, w},
    dx_enable,
    dw_enable,
    db_enable,
    xdtype
  };
  if ( mode == BatchNorm_Spatial ) sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_batchnorm_backward", &api, sizeof ( api ) );
  else if ( mode == BatchNorm_Per_Layer ) sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_layernorm_backward", &api, sizeof ( api ) );
  else {assert ( 0 );}
  return BM_SUCCESS;
}

bm_status_t sgdnn_pooling_forward (
bm_handle_t                 handle,
const PoolingDescriptor_t   poolingDesc,
const void                 *alpha,
const TensorDescriptor_t    xDesc,
const void                 *x,
const void                 *beta,
const TensorDescriptor_t    yDesc,
void                       *y
) {
  assert ( * ( float * ) alpha == 1.f );
  assert ( * ( float * ) beta == 0.f );
  assert ( xDesc.ndims == 4 && yDesc.ndims == 4 );
  int n = xDesc.shape[0];
  int c = xDesc.shape[1];
  int ih = xDesc.shape[2];
  int iw = xDesc.shape[3];
  int oh = yDesc.shape[2];
  int ow = yDesc.shape[3];
  int pooling_mode = ( PoolingMode_t ) poolingDesc.mode;
  int kh = poolingDesc.kh;
  int kw = poolingDesc.kw;
  int pad_h = poolingDesc.pad_h;
  int pad_w = poolingDesc.pad_w;
  int stride_h = poolingDesc.stride_h;
  int stride_w = poolingDesc.stride_w;
  sg_data_type_t idtype = ( sg_data_type_t ) xDesc.dtype;
  sg_data_type_t odtype = ( sg_data_type_t ) yDesc.dtype;
  assert ( idtype == odtype );
  UNUSED ( odtype );
  sg_api_pooling_forward_t api = {
    ( unsigned long long ) x,
    ( unsigned long long ) y,
    0,//max_mask
    n, c, ih, iw,
    oh, ow,
    kh, kw,
    pad_h, pad_w, pad_h, pad_w,
    stride_h, stride_w,
    1, 1,//dilation_h && dilation_w
    pooling_mode == Pooling_AVERAGE,
    0,//avgpool_mode
    0,//if_mask_max
    0,//if_relu
    0,//relu_upper_limit
    idtype
  };
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_pooling_forward", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

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
void                       *dx
) {
  assert ( * ( float * ) alpha == 1.f );
  assert ( * ( float * ) beta == 0.f );
  ASSERT_SAME_SHAPE ( xDesc, dxDesc );
  ASSERT_SAME_SHAPE ( yDesc, dyDesc );
  assert ( xDesc.ndims == 4 && yDesc.ndims == 4 );
  int n = xDesc.shape[0];
  int c = xDesc.shape[1];
  int ih = xDesc.shape[2];
  int iw = xDesc.shape[3];
  int oh = yDesc.shape[2];
  int ow = yDesc.shape[3];
  int pooling_mode = ( PoolingMode_t ) poolingDesc.mode;
  int kh = poolingDesc.kh;
  int kw = poolingDesc.kw;
  int pad_h = poolingDesc.pad_h;
  int pad_w = poolingDesc.pad_w;
  int stride_h = poolingDesc.stride_h;
  int stride_w = poolingDesc.stride_w;
  assert ( xDesc.dtype == yDesc.dtype );
  assert ( xDesc.dtype == dxDesc.dtype );
  assert ( yDesc.dtype == dyDesc.dtype );
  sg_data_type_t dtype = ( sg_data_type_t ) ( xDesc.dtype );
  if ( pooling_mode == Pooling_MAX ) {
    sg_api_maxpool_backward_t api = {
      ( unsigned long long ) x,
      ( unsigned long long ) y,
      ( unsigned long long ) dy,
      ( unsigned long long ) dx,
      {n, c, ih, iw},
      {n, c, oh, ow},
      {kh, kw},
      {stride_h, stride_w},
      {pad_h, pad_w},
      {1, 1},//{dilation_h, dilation_w},
      0,//ceil_mode
      dtype
    };
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_maxpool_backward", &api, sizeof ( api ) );
  } else if ( pooling_mode == Pooling_AVERAGE ) {
    sg_api_avgpool_backward_t api = {
      ( unsigned long long ) dy,
      ( unsigned long long ) dx,
      {n, c, ih, iw},
      {n, c, oh, ow},
      {kh, kw},
      {stride_h, stride_w},
      {pad_h, pad_w},
      0,//ceil_mode
      1,//count_include_pad
      kh * kw,//divisor_override
      dtype
    };
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_avgpool_backward", &api, sizeof ( api ) );
  }
  return BM_SUCCESS;
}

bm_status_t sgdnn_binary (
bm_handle_t                 handle,
const TensorDescriptor_t    aDesc,
const void*                 A,
const TensorDescriptor_t    bDesc,
const void*                 B,
const TensorDescriptor_t    cDesc,
void*                       C,
BinaryOpMode_t              opTensorDesc )
{
  sg_binary_type_t binary_type = tpu_binary_type_convert ( opTensorDesc );
  // if (aDesc.ndims > 0 && bDesc.ndims > 0 && aDesc.ndims > bDesc.ndims)
  // {
  //     TensorDescriptor_t bDescSaved = bDesc;
  //     int dimgap = aDesc.ndims - bDesc.ndims;
  //     int i = 0;
  //     for (; i < dimgap; ++i)
  //     {
  //         const_cast<TensorDescriptor_t &>(bDesc).shape[i] = 1;
  //     }
  //     for (; i < aDesc.ndims; ++i)
  //     {
  //         const_cast<TensorDescriptor_t &>(bDesc).shape[i] = bDescSaved.shape[i - dimgap];
  //     }
  //     const_cast<TensorDescriptor_t &>(bDesc).ndims = aDesc.ndims;
  // }
  // else if (aDesc.ndims > 0 && bDesc.ndims > 0 && aDesc.ndims < bDesc.ndims)
  // {
  //     TensorDescriptor_t aDescSaved = aDesc;
  //     int dimgap = bDesc.ndims - aDesc.ndims;
  //     int i = 0;
  //     for (; i < dimgap; ++i)
  //     {
  //         const_cast<TensorDescriptor_t &>(aDesc).shape[i] = 1;
  //     }
  //     for (; i < bDesc.ndims; ++i)
  //     {
  //         const_cast<TensorDescriptor_t &>(aDesc).shape[i] = aDescSaved.shape[i - dimgap];
  //     }
  //     const_cast<TensorDescriptor_t &>(aDesc).ndims = bDesc.ndims;
  // }
  if ( aDesc.ndims && bDesc.ndims && cDesc.ndims )
  {
    sg_data_type_t dtype_A = ( sg_data_type_t ) ( aDesc.dtype );
    sg_data_type_t dtype_B = ( sg_data_type_t ) ( bDesc.dtype );
    sg_data_type_t dtype_C = ( sg_data_type_t ) ( cDesc.dtype );
    assert ( dtype_A == dtype_B && dtype_B == dtype_C );
    sg_api_bcbinary_float_t api;
    api.A_global_addr = ( unsigned long long ) A;
    api.B_global_addr = ( unsigned long long ) B;
    api.res_global_addr = ( unsigned long long ) C;
    for ( int i = 0; i < aDesc.ndims; ++i )
    {
      api.A_shape[i] = aDesc.shape[i];
    }
    for ( int i = 0; i < bDesc.ndims; ++i )
    {
      api.B_shape[i] = bDesc.shape[i];
    }
    api.A_dims = aDesc.ndims;
    api.B_dims = bDesc.ndims;
    api.dtype = dtype_A;
    api.binary_type = binary_type;
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_bcbinary_float", &api, sizeof ( api ) );
  }
  else if ( ( !aDesc.ndims && bDesc.ndims ) || ( !bDesc.ndims && aDesc.ndims ) )
  {
    const void* tensor = !aDesc.ndims ? B : A;
    const void* scalar = !aDesc.ndims ? A : B;
    TensorDescriptor_t tensorDesc = !aDesc.ndims ? bDesc : aDesc;
    TensorDescriptor_t scalarDesc = !aDesc.ndims ? aDesc : bDesc;
    float const_value = ( ( float* ) scalar ) [0];
    sg_data_type_t tensor_dtype = ( sg_data_type_t ) ( tensorDesc.dtype );
    sg_data_type_t scalar_dtype = ( sg_data_type_t ) ( scalarDesc.dtype );
    switch ( scalar_dtype )
    {
    case SG_DTYPE_INT8:     const_value = ( ( s8* ) scalar ) [0];  break;
    case SG_DTYPE_UINT8:    const_value = ( ( u8* ) scalar ) [0];  break;
    case SG_DTYPE_INT16:    const_value = ( ( s16* ) scalar ) [0];  break;
    case SG_DTYPE_UINT16:   const_value = ( ( u16* ) scalar ) [0];  break;
    case SG_DTYPE_INT32:    const_value = ( ( s32* ) scalar ) [0];  break;
    case SG_DTYPE_UINT32:   const_value = ( ( u32* ) scalar ) [0];  break;
    case SG_DTYPE_FP32:                                       break;
    case SG_DTYPE_FP16:     assert ( 0 );                        break;
    case SG_DTYPE_BFP16:    assert ( 0 );                        break;
    default:                assert ( 0 );                        break;
    }
    int n, c, h, w;
    if ( tensorDesc.ndims == 4 )
    {
      n = tensorDesc.shape[0];
      c = tensorDesc.shape[1];
      h = tensorDesc.shape[2];
      w = tensorDesc.shape[3];
    }
    else if ( tensorDesc.ndims == 2 )
    {
      n = 1;
      c = tensorDesc.shape[0];
      h = 1;
      w = tensorDesc.shape[1];
    }
    else if ( tensorDesc.ndims == 1 )
    {
      n = 1;
      c = tensorDesc.shape[0];
      h = 1;
      w = 1;
    }
    else
    {
      assert ( false );
    }
    bool is_inversed = !aDesc.ndims;
    sg_api_const_binary_float_t api = {
      ( unsigned long long ) tensor,
      ( unsigned long long ) C,
      {n, c, h, w},
      4,
      binary_type,
      tensor_dtype,
      const_value,
      is_inversed
    };
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_const_binary", &api, sizeof ( api ) );
  }
  else
  {
    assert ( 0 );
  }
  return BM_SUCCESS;
}

bm_status_t sgdnn_activation_forward (
bm_handle_t                     handle,
ActivationDescriptor_t          activationDesc,
const void                     *alpha,
const TensorDescriptor_t        xDesc,
const void                     *x,
const void                     *beta,
const TensorDescriptor_t        yDesc,
void                           *y )
{
  if ( activationDesc.mode == Activation_Relu )
  {
    unsigned long long input    = ( unsigned long long ) x;
    unsigned long long output   = ( unsigned long long ) y;
    float alpha_ = ( ( float* ) alpha ) [0];
    assert ( alpha_ == 1.0f );
    float beta_ = ( ( float* ) beta ) [0];
    assert ( beta_ == 0.0f || beta_ == 1.0f );
    float upper_limit = activationDesc.coef;
    assert ( xDesc.ndims == 4 );
    assert ( yDesc.ndims == 4 );
    int n = xDesc.shape[0];
    int c = xDesc.shape[1];
    int h = xDesc.shape[2];
    int w = xDesc.shape[3];
    sg_data_type_t ydtype = ( sg_data_type_t ) ( yDesc.dtype );
    sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
    assert ( xdtype == ydtype );
    sg_api_relu_forward_t api = {
      input,
      output,
      {n, c, h, w},
      upper_limit,
      xdtype
    };
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_relu_forward", &api, sizeof ( api ) );
    return BM_SUCCESS;
  }
  else if ( activationDesc.mode == Activation_Gelu && xDesc.dtype == SG_DTYPE_FP16 )
  {
    assert ( xDesc.ndims == yDesc.ndims );
    sg_data_type_t ydtype = ( sg_data_type_t ) ( yDesc.dtype );
    sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
    assert ( xdtype == ydtype );
    sg_api_gelu_forward_t api;
    api.x_global_addr =  ( unsigned long long ) x;
    api.y_global_addr =  ( unsigned long long ) y;
    api.dim = xDesc.ndims;
    api.dtype = xdtype;
    for ( int i = 0; i < xDesc.ndims; ++i )
    {
      assert ( xDesc.shape[i] == yDesc.shape[i] );
      api.shape[i] = xDesc.shape[i];
    }
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_gelu_forward", &api, sizeof ( api ) );
    return BM_SUCCESS;
  }
  else
  {
    sg_active_type_t active_type =  tpu_active_type_convert ( activationDesc.mode ) ;
    unsigned long long input    = ( unsigned long long ) x;
    unsigned long long output   = ( unsigned long long ) y;
    assert ( xDesc.ndims == yDesc.ndims );
    sg_data_type_t ydtype = ( sg_data_type_t ) ( yDesc.dtype );
    sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
    assert ( xdtype == ydtype );
    sg_api_active_forward_t api;
    api.in_global_addr = input;
    api.out_global_addr = output;
    api.shape_dim = xDesc.ndims;
    api.dtype = xdtype;
    api.active_type = active_type;
    for ( int i = 0; i < xDesc.ndims; ++i )
    {
      assert ( xDesc.shape[i] == yDesc.shape[i] );
      api.shape[i] = xDesc.shape[i];
    }
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_active_forward", &api, sizeof ( api ) );
    return BM_SUCCESS;
  }
  return BM_ERR_NOFEATURE;
}

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
void                            *dx )
{
  assert ( activationDesc.mode == Activation_Relu );
  if ( activationDesc.mode == Activation_Relu )
  {
    unsigned long long input         = ( unsigned long long ) x;
    unsigned long long grad_output   = ( unsigned long long ) dy;
    unsigned long long grad_input    = ( unsigned long long ) dx;
    float alpha_ = ( ( float* ) alpha ) [0];
    assert ( alpha_ == 1.0f );
    float beta_ = ( ( float* ) beta ) [0];
    assert ( beta_ == 0.0f || beta_ == 1.0f );
    float upper_limit = activationDesc.coef;
    //TODO: clipped_relu_backward
    assert ( upper_limit == 0 );
    assert ( dyDesc.ndims == 4 );
    assert ( xDesc.ndims == 4 );
    assert ( dxDesc.ndims == 4 );
    int n = xDesc.shape[0];
    int c = xDesc.shape[1];
    int h = xDesc.shape[2];
    int w = xDesc.shape[3];
    sg_data_type_t dydtype = ( sg_data_type_t ) ( dyDesc.dtype );
    sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
    sg_data_type_t dxdtype = ( sg_data_type_t ) ( dxDesc.dtype );
    assert ( dydtype == xdtype && xdtype == dxdtype );
    sg_api_relu_backward_t api = {
      input,
      grad_output,
      grad_input,
      {n, c, h, w},
      xdtype
    };
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_relu_backward", &api, sizeof ( api ) );
    return BM_SUCCESS;
  }
  return BM_ERR_NOFEATURE;
}

//y = cast(x)
bm_status_t sgdnn_dtype_convert (
bm_handle_t                      handle,
const TensorDescriptor_t         xDesc,
const void                      *xData,
const TensorDescriptor_t         yDesc,
const void                      *yData,
sg_round_mode_t                  round_mode
) {
  assert ( xDesc.ndims == yDesc.ndims );
  for ( int idx = 0; idx < xDesc.ndims; idx++ ) {
    assert ( xDesc.shape[idx] == yDesc.shape[idx] );
  }
  sg_api_dtype_convert_t api;
  api.input_global_addr = ( unsigned long long ) xData;
  api.output_global_addr = ( unsigned long long ) yData;
  memcpy ( api.shape, xDesc.shape, xDesc.ndims * sizeof ( int ) );
  api.dims = xDesc.ndims;
  api.idtype = ( sg_data_type_t ) xDesc.dtype;
  api.odtype = ( sg_data_type_t ) yDesc.dtype;
  api.round_mode = ( sg_round_mode_t ) round_mode;
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_dtype_convert", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

//y = x -> 32ic or y = x -> 32oc
bm_status_t sgdnn_conv_weight_reorder (
bm_handle_t                      handle,
const TensorDescriptor_t         xDesc,
const void                      *xData,
const TensorDescriptor_t         yDesc,
const void                      *yData,
ConvWeightReorderMode_t          reorder_mode
) {
  assert ( xDesc.ndims == 4 && yDesc.ndims == 4 );
  int n = xDesc.shape[0];
  int c = xDesc.shape[1];
  int h = xDesc.shape[2];
  int w = xDesc.shape[3];
  sg_api_conv_weight_reorder_t api = {
    ( unsigned long long ) xData,
    ( unsigned long long ) yData,
    {n, c, h, w},
    reorder_mode
  };
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_conv_weight_reorder", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_batch_matmul (
bm_handle_t                      handle,
const TensorDescriptor_t         LDesc,
const void                      *L,
const TensorDescriptor_t         RDesc,
const void                      *R,
const TensorDescriptor_t         YDesc,
void                            *Y,
int                              L_transpose,
int                              R_transpose )
{
  assert ( L_transpose == 0 );
  assert ( LDesc.ndims == 3 && RDesc.ndims == 3 && YDesc.ndims == 3 );
  assert ( LDesc.shape[0] == RDesc.shape[0] );
  assert ( LDesc.shape[0] == YDesc.shape[0] );
  assert ( LDesc.shape[1] == YDesc.shape[1] );
  assert ( RDesc.shape[2] == YDesc.shape[2] );
  assert ( LDesc.shape[2] == RDesc.shape[1] );
  assert ( LDesc.dtype == RDesc.dtype );
  assert ( LDesc.dtype == YDesc.dtype );
  sg_api_batch_matmul_t api;
  api.L_addr = ( unsigned long long ) L;
  api.R_addr = ( unsigned long long ) R;
  api.Y_addr = ( unsigned long long ) Y;
  api.B_addr = 0;
  api.batch_num = YDesc.shape[0];
  api.L_row_num = LDesc.shape[1];
  api.L_col_num = LDesc.shape[2];
  api.R_col_num = RDesc.shape[2];
  api.L_trans = L_transpose;
  api.R_trans = R_transpose;
  api.dtype = ( sg_data_type_t ) ( YDesc.dtype );
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_batch_matmul", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_softmax_forward (
bm_handle_t                      handle,
int                              dim,
const void                      *alpha,
const TensorDescriptor_t         xDesc,
const void                      *x,
const void                      *beta,
const TensorDescriptor_t         yDesc,
void                            *y )
{
  assert ( * ( ( float * ) alpha ) == 1.f );
  assert ( * ( ( float * ) beta ) == 0.f );
  sg_data_type_t ydtype = ( sg_data_type_t ) ( yDesc.dtype );
  sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
  assert ( xdtype == ydtype );
  assert ( xDesc.ndims == yDesc.ndims );
  for ( int i = 0; i < xDesc.ndims; ++i )
  {
    assert ( xDesc.shape[i] == yDesc.shape[i] );
  }
  if ( xdtype == SG_DTYPE_FP32 || xdtype == SG_DTYPE_FP16 )
  {
    sg_api_softmax_forward_t api;
    api.input_global_addr = ( unsigned long long ) x;
    api.output_global_addr = ( unsigned long long ) y;
    for ( int i = 0; i < xDesc.ndims; ++i )
    {
      api.shape[i] = xDesc.shape[i];
    }
    api.dims = xDesc.ndims;
    api.compute_dim = dim;
    api.scale_val = 1.f;
    api.dtype = xdtype;
    sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_softmax_forward", &api, sizeof ( api ) );
  }
  else
  {
    assert ( false );
  }
  return BM_SUCCESS;
}

bm_status_t sgdnn_softmax_backward (
bm_handle_t                      handle,
int                              dim,
const TensorDescriptor_t         yDesc,
const void                      *y,
const TensorDescriptor_t         dyDesc,
const void                      *dy,
const TensorDescriptor_t         dxDesc,
void                            *dx )
{
  sg_data_type_t ydtype  = ( sg_data_type_t ) ( yDesc.dtype );
  sg_data_type_t dydtype = ( sg_data_type_t ) ( dyDesc.dtype );
  sg_data_type_t dxdtype = ( sg_data_type_t ) ( dxDesc.dtype );
  assert ( ydtype == dydtype && dydtype == dxdtype );
  assert ( yDesc.ndims == dyDesc.ndims && dyDesc.ndims == dxDesc.ndims && yDesc.ndims == 4 );
  assert ( dim == 3 );
  for ( int i = 0; i < yDesc.ndims; ++i )
  {
    assert ( yDesc.shape[i] == dyDesc.shape[i] );
    assert ( yDesc.shape[i] == dxDesc.shape[i] );
  }
  sg_api_softmax_backward_t api;
  api.output_global_addr      = ( unsigned long long ) y;
  api.grad_output_global_addr = ( unsigned long long ) dy;
  api.grad_input_global_addr  = ( unsigned long long ) dx;
  api.input_n = yDesc.shape[0];
  api.input_c = yDesc.shape[1];
  api.input_h = yDesc.shape[2];
  api.input_w = yDesc.shape[3];
  api.dim = dim;
  api.dtype = ydtype;
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_softmax_backward", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_transpose (
bm_handle_t                      handle,
const TensorDescriptor_t         xDesc,
const void                      *xData,
const TensorDescriptor_t         yDesc,
void                            *yData )
{
  assert ( xDesc.ndims == yDesc.ndims && xDesc.ndims == 2 );
  assert ( xDesc.shape[0] == yDesc.shape[1] );
  assert ( xDesc.shape[1] == yDesc.shape[0] );
  sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
  sg_data_type_t ydtype = ( sg_data_type_t ) ( yDesc.dtype );
  assert ( xdtype == ydtype );
  sg_api_transpose_t api;
  api.input_global_mem_addr = ( unsigned long long ) xData;
  api.output_global_mem_addr = ( unsigned long long ) yData;
  api.buffer_global_mem_addr = 0;
  api.dims = 2;
  api.sgdtype = xdtype;
  api.input_shape[0] = xDesc.shape[0];
  api.input_shape[1] = xDesc.shape[1];
  api.order[0] = 1;
  api.order[1] = 0;
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_transpose", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_gelu_backward (
bm_handle_t                     handle,
const TensorDescriptor_t        xDesc,
const void                     *x,
const TensorDescriptor_t        dyDesc,
const void                     *dy,
const TensorDescriptor_t        dxDesc,
void                           *dx )
{
  assert ( xDesc.ndims == dxDesc.ndims && xDesc.ndims == dyDesc.ndims );
  sg_data_type_t dxdtype = ( sg_data_type_t ) ( dxDesc.dtype );
  sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
  sg_data_type_t dydtype = ( sg_data_type_t ) ( dyDesc.dtype );
  assert ( dxdtype == xdtype && xdtype == dydtype );
  sg_api_gelu_backward_t api;
  api.dx_global_addr = ( unsigned long long ) dx;
  api.dy_global_addr = ( unsigned long long ) dy;
  api.x_global_addr = ( unsigned long long ) x;
  api.dim = xDesc.ndims;
  api.dtype = xdtype;
  for ( int i = 0; i < xDesc.ndims; ++i )
  {
    assert ( xDesc.shape[i] == dyDesc.shape[i] );
    assert ( xDesc.shape[i] == dxDesc.shape[i] );
    api.shape[i] = xDesc.shape[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_gelu_backward", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_reduce (
bm_handle_t                     handle,
const TensorDescriptor_t        xDesc,
const void                     *x,
const TensorDescriptor_t        yDesc,
void                           *y,
int                             reduce_dim_start,
int                             reduce_dim_end,
int                             keep_dim,
ReductionMode_t                 mode )
{
  if ( reduce_dim_start < 0 )
  {
    reduce_dim_start += xDesc.ndims;
  }
  if ( reduce_dim_end < 0 )
  {
    reduce_dim_end += xDesc.ndims;
  }
  if ( reduce_dim_end < reduce_dim_start )
  {
    int tmp = reduce_dim_end;
    reduce_dim_end = reduce_dim_start;
    reduce_dim_start = tmp;
  }
  assert ( reduce_dim_end > reduce_dim_start );
  if ( keep_dim )
  {
    assert ( xDesc.ndims == yDesc.ndims );
  }
  else
  {
    assert ( xDesc.ndims == yDesc.ndims + ( reduce_dim_end - reduce_dim_start ) );
  }
  sg_data_type_t xdtype = ( sg_data_type_t ) ( xDesc.dtype );
  sg_data_type_t ydtype = ( sg_data_type_t ) ( yDesc.dtype );
  assert ( xdtype == ydtype );
  sg_api_reduce_t api;
  if ( keep_dim )
  {
    for ( int i = 0; i < xDesc.ndims; ++i )
    {
      if ( i < reduce_dim_start )
      {
        assert ( xDesc.shape[i] == yDesc.shape[i] );
      }
      else if ( i >= reduce_dim_start && i < reduce_dim_end )
      {
        assert ( yDesc.shape[i] == 1 );
      }
      else if ( i >= reduce_dim_end )
      {
        assert ( xDesc.shape[i] == yDesc.shape[i] );
      }
      api.shape[i] = xDesc.shape[i];
    }
  }
  else
  {
    for ( int i = 0; i < xDesc.ndims; ++i )
    {
      if ( i < reduce_dim_start )
      {
        assert ( xDesc.shape[i] == yDesc.shape[i] );
      }
      else if ( i >= reduce_dim_end )
      {
        assert ( xDesc.shape[i] == yDesc.shape[i - ( reduce_dim_end - reduce_dim_start )] );
      }
      api.shape[i] = xDesc.shape[i];
    }
  }
  api.shape_dim = xDesc.ndims;
  api.input_global_addr = ( unsigned long long ) x;
  api.output_global_addr = ( unsigned long long ) y;
  api.reduce_dim_start = reduce_dim_start;
  api.reduce_dim_end = reduce_dim_end;
  api.dtype = xdtype;
  api.reduction_mode = mode;
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_reduce", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_strided_copy (
bm_handle_t                     handle,
const TensorDescriptor_t        srcDesc,
const void                      *src,
const TensorDescriptor_t        dstDesc,
void                            *dst )
{
  sg_data_type_t srcDtype = ( sg_data_type_t ) ( srcDesc.dtype );
  sg_data_type_t dstDtype = ( sg_data_type_t ) ( srcDesc.dtype );
  assert ( srcDtype == dstDtype );
  assert ( srcDesc.ndims == dstDesc.ndims );
  sg_api_strided_copy_t api;
  api.dtype = srcDtype;
  api.shape_dim = srcDesc.ndims;
  api.in_global_addr = ( unsigned long long ) src;
  api.out_global_addr = ( unsigned long long ) dst;
  for ( int i = 0; i < srcDesc.ndims; i++ ) {
    assert ( srcDesc.shape[i] == dstDesc.shape[i] );
    api.shape[i]      = srcDesc.shape[i];
    api.in_stride[i]  = srcDesc.stride[i];
    api.out_stride[i] = dstDesc.stride[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_strided_copy", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_where (
bm_handle_t                     handle,
const TensorDescriptor_t        condDesc,
const void                     *cond,
const TensorDescriptor_t        selfDesc,
const void                     *self,
const TensorDescriptor_t        otherDesc,
const void                     *other,
const TensorDescriptor_t        outDesc,
void                           *out )
{
  sg_data_type_t condDtype = ( sg_data_type_t ) ( condDesc.dtype );
  sg_data_type_t selfDtype = ( sg_data_type_t ) ( selfDesc.dtype );
  sg_data_type_t otherDtype = ( sg_data_type_t ) ( otherDesc.dtype );
  sg_data_type_t outDtype = ( sg_data_type_t ) ( outDesc.dtype );
  assert ( selfDtype == otherDtype );
  assert ( selfDtype == outDtype );
  assert ( condDesc.ndims > 0 );
  assert ( condDesc.ndims == outDesc.ndims );
  if ( selfDesc.ndims > 0 )
  {
    assert ( selfDesc.ndims == outDesc.ndims );
  }
  if ( otherDesc.ndims > 0 )
  {
    assert ( otherDesc.ndims == outDesc.ndims );
  }
  sg_api_where_t api;
  api.cond_global_addr = ( unsigned long long ) cond;
  api.self_global_addr = ( unsigned long long ) self;
  api.other_global_addr = ( unsigned long long ) other;
  api.out_global_addr = ( unsigned long long ) out;
  api.self_is_scalar = selfDesc.ndims == 0;
  api.other_is_scalar = otherDesc.ndims == 0;
  api.cond_dtype = condDtype;
  api.dtype = selfDtype;
  api.shape_dim = selfDesc.ndims;
  for ( int i = 0; i < outDesc.ndims; i++ ) {
    api.cond_shape[i] = condDesc.shape[i];
    api.out_shape[i] = outDesc.shape[i];
    if ( selfDesc.ndims > 0 )
    {
      api.self_shape[i] = selfDesc.shape[i];
    }
    if ( otherDesc.ndims > 0 )
    {
      api.other_shape[i] = otherDesc.shape[i];
    }
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_where", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_concat (
bm_handle_t                     handle,
const TensorDescriptor_t       *inputDescs,
const void * const *            inputs,
int                             input_num,
const TensorDescriptor_t        outputDesc,
void                           *output,
int                             concat_dim )
{
  sg_data_type_t outDtype = ( sg_data_type_t ) ( outputDesc.dtype );
  if ( concat_dim < 0 )
  {
    concat_dim += outputDesc.ndims;
  }
  int concat_dim_shape = 0;
  for ( int i = 0; i < input_num; ++i )
  {
    sg_data_type_t inDtype = ( sg_data_type_t ) ( inputDescs[i].dtype );
    assert ( inDtype == outDtype );
    assert ( inputDescs[i].ndims == outputDesc.ndims );
    for ( int j = 0; j < outputDesc.ndims; ++j )
    {
      if ( j != concat_dim )
      {
        assert ( inputDescs[i].shape[j] == outputDesc.shape[j] );
      }
      else
      {
        concat_dim_shape += inputDescs[i].shape[j];
      }
    }
  }
  assert ( concat_dim_shape == outputDesc.shape[concat_dim] );
  sg_api_concat_t api;
  for ( int i = 0; i < input_num; ++i )
  {
    api.input_global_addrs[i] = ( unsigned long long ) inputs[i];
    for ( int j = 0; j < inputDescs[i].ndims; ++j )
    {
      api.input_shapes[i][j] = inputDescs[i].shape[j];
    }
  }
  api.output_global_addr = ( unsigned long long ) output;
  api.input_num = input_num;
  api.shape_dim = outputDesc.ndims;
  api.concat_dim = concat_dim;
  api.dtype = outDtype;
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_concat", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_index_select (
bm_handle_t                     handle,
const TensorDescriptor_t        tableDesc,
const void                      *table,
const TensorDescriptor_t        indexDesc,
const void                      *index,
const TensorDescriptor_t        outDesc,
void                            *out,
int                             dim ) {
  sg_api_index_select_t api;
  api.input_global_addr     = ( unsigned long long ) table;
  api.index_global_addr     = ( unsigned long long ) index;
  api.output_global_addr    = ( unsigned long long ) out;
  api.shape_dims            = tableDesc.ndims;
  api.index_num             = 1;
  api.axis                  = dim;
  api.const_val             = 0;
  api.dtype                 = ( sg_data_type_t ) tableDesc.dtype;
  for ( int i = 0; i < tableDesc.ndims; i++ ) {
    api.input_shape[i] = tableDesc.shape[i];
  }
  for ( int i = 0; i < indexDesc.ndims; i++ ) {
    api.index_num *= indexDesc.shape[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_index_select", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_const_fill (
bm_handle_t                     handle,
const TensorDescriptor_t        srcDesc,
void                           *src,
const void*                     fill_value ) {
  sg_api_constant_fill_t api;
  api.out_global_addr       = ( unsigned long long ) src;
  api.shape_dim             = srcDesc.ndims;
  api.filled_sgdtype        = ( sg_data_type_t ) srcDesc.dtype;
  api.filled_value          = 0;
  memcpy ( &api.filled_value, fill_value, dtype_size ( api.filled_sgdtype ) );
  for ( int i = 0; i < srcDesc.ndims; i++ ) {
    api.shape[i] = srcDesc.shape[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_const_fill", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_sqrt (
bm_handle_t                     handle,
const TensorDescriptor_t        inputDesc,
const void                     *input,
const TensorDescriptor_t        outputDesc,
void                           *output )
{
  sg_api_sqrt_t api;
  api.input_global_addr = ( unsigned long long ) input;
  api.output_global_addr = ( unsigned long long ) output;
  api.dim = inputDesc.ndims;
  api.dtype = ( sg_data_type_t ) inputDesc.dtype;
  assert ( inputDesc.ndims == outputDesc.ndims );
  assert ( inputDesc.dtype == outputDesc.dtype );
  for ( int i = 0; i < inputDesc.ndims; i++ ) {
    assert ( inputDesc.shape[i] == outputDesc.shape[i] );
    api.shape[i] = inputDesc.shape[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_sqrt", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

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
double                          value )
{
  sg_api_addcdiv_t api;
  api.input_global_addr = ( unsigned long long ) input;
  api.tensor1_global_addr = ( unsigned long long ) tensor1;
  api.tensor2_global_addr = ( unsigned long long ) tensor2;
  api.output_global_addr = ( unsigned long long ) output;
  api.dim = inputDesc.ndims;
  api.dtype = ( sg_data_type_t ) inputDesc.dtype;
  api.value = ( float ) value;
  assert ( inputDesc.ndims == outputDesc.ndims );
  assert ( inputDesc.ndims == tensor1Desc.ndims );
  assert ( inputDesc.ndims == tensor2Desc.ndims );
  assert ( inputDesc.dtype == outputDesc.dtype );
  assert ( inputDesc.dtype == tensor1Desc.dtype );
  assert ( inputDesc.dtype == tensor2Desc.dtype );
  for ( int i = 0; i < inputDesc.ndims; i++ ) {
    assert ( inputDesc.shape[i] == outputDesc.shape[i] );
    assert ( inputDesc.shape[i] == tensor1Desc.shape[i] );
    assert ( inputDesc.shape[i] == tensor2Desc.shape[i] );
    api.shape[i] = inputDesc.shape[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_addcdiv", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

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
double                          value )
{
  sg_api_addcmul_t api;
  api.input_global_addr = ( unsigned long long ) input;
  api.tensor1_global_addr = ( unsigned long long ) tensor1;
  api.tensor2_global_addr = ( unsigned long long ) tensor2;
  api.output_global_addr = ( unsigned long long ) output;
  api.dim = inputDesc.ndims;
  api.dtype = ( sg_data_type_t ) inputDesc.dtype;
  api.value = ( float ) value;
  assert ( inputDesc.ndims == outputDesc.ndims );
  assert ( inputDesc.ndims == tensor1Desc.ndims );
  assert ( inputDesc.ndims == tensor2Desc.ndims );
  assert ( inputDesc.dtype == outputDesc.dtype );
  assert ( inputDesc.dtype == tensor1Desc.dtype );
  assert ( inputDesc.dtype == tensor2Desc.dtype );
  for ( int i = 0; i < inputDesc.ndims; i++ ) {
    assert ( inputDesc.shape[i] == outputDesc.shape[i] );
    assert ( inputDesc.shape[i] == tensor1Desc.shape[i] );
    assert ( inputDesc.shape[i] == tensor2Desc.shape[i] );
    api.shape[i] = inputDesc.shape[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_addcmul", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_scale_add (
bm_handle_t                     handle,
const TensorDescriptor_t        inputDesc,
const void                     *input,
const TensorDescriptor_t        otherDesc,
const void                     *other,
const TensorDescriptor_t        outputDesc,
void                           *output,
double                          value )
{
  sg_api_scale_add_t api;
  api.input_global_addr = ( unsigned long long ) input;
  api.other_global_addr = ( unsigned long long ) other;
  api.output_global_addr = ( unsigned long long ) output;
  api.dim = inputDesc.ndims;
  api.dtype = ( sg_data_type_t ) inputDesc.dtype;
  api.value = ( float ) value;
  assert ( inputDesc.ndims == outputDesc.ndims );
  assert ( inputDesc.ndims == otherDesc.ndims );
  assert ( inputDesc.dtype == outputDesc.dtype );
  assert ( inputDesc.dtype == otherDesc.dtype );
  for ( int i = 0; i < inputDesc.ndims; i++ ) {
    assert ( inputDesc.shape[i] == outputDesc.shape[i] );
    assert ( inputDesc.shape[i] == otherDesc.shape[i] );
    api.shape[i] = inputDesc.shape[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_scale_add", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_mulc (
bm_handle_t                     handle,
const TensorDescriptor_t        inputDesc,
const void                     *input,
const TensorDescriptor_t        outputDesc,
void                           *output,
double                          value )
{
  sg_api_mulc_t api;
  api.input_global_addr = ( unsigned long long ) input;
  api.output_global_addr = ( unsigned long long ) output;
  api.dim = inputDesc.ndims;
  api.dtype = ( sg_data_type_t ) inputDesc.dtype;
  api.value = ( float ) value;
  assert ( inputDesc.ndims == outputDesc.ndims );
  assert ( inputDesc.dtype == outputDesc.dtype );
  for ( int i = 0; i < inputDesc.ndims; i++ ) {
    assert ( inputDesc.shape[i] == outputDesc.shape[i] );
    api.shape[i] = inputDesc.shape[i];
  }
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_mulc", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

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
double                           label_smoothing )
{
  sg_api_cross_entropy_loss_forward_t api;
  assert ( ignore_index < 0 );
  assert ( inDesc.ndims == 2 );
  api.batch_num = inDesc.shape[0];
  api.class_num = inDesc.shape[1];
  api.dtype = ( sg_data_type_t ) inDesc.dtype;
  assert ( api.dtype == SG_DTYPE_FP32 || api.dtype == SG_DTYPE_FP16 );
  assert ( targetDesc.dtype == 31 || targetDesc.dtype == SG_DTYPE_INT32 );
  if ( targetDesc.dtype == 31 )
  {
    api.target_is_int64 = 1;
  }
  else
  {
    api.target_is_int64 = 0;
  }
  assert ( targetDesc.ndims == 1 );
  assert ( targetDesc.shape[0] == api.batch_num );
  if ( has_weight )
  {
    assert ( weightDesc.ndims == 1 );
    assert ( weightDesc.shape[0] == api.class_num );
    assert ( weightDesc.dtype == inDesc.dtype );
    api.weight_global_addr = ( unsigned long long ) weight;
  }
  else
  {
    api.weight_global_addr = 0;
  }
  assert ( reduction == Mean_Reduction || reduction == Sum_Reduction );
  if ( reduction == Mean_Reduction )
  {
    assert ( weight == nullptr );
    api.reduction = 0;
  }
  else if ( reduction == Sum_Reduction )
  {
    api.reduction = 1;
  }
  assert ( outDesc.ndims == 0 );
  assert ( outDesc.dtype == inDesc.dtype );
  api.label_smoothing = label_smoothing;
  api.input_global_addr = ( unsigned long long ) in;
  api.target_global_addr = ( unsigned long long ) target;
  api.output_global_addr = ( unsigned long long ) out;
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_cross_entropy_loss_forward", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_cross_entropy_backward (
bm_handle_t                      handle,
const TensorDescriptor_t         targetDesc,
const void                      *target,
const TensorDescriptor_t         inDesc,
const void                      *in,
const TensorDescriptor_t         weightDesc,
const void                      *weight,
const TensorDescriptor_t         gradoutDesc,
const void                      *gradout,
const TensorDescriptor_t         gradinDesc,
void                            *gradin,
CrossEntropyMode_t               reduction,
int                              ignore_index,
double                           label_smoothing,
bool                             has_weight ) {
  sg_api_cross_entropy_loss_backward_t api;
  assert ( ignore_index < 0 );
  assert ( inDesc.ndims == 2 );
  api.batch_num = inDesc.shape[0];
  api.class_num = inDesc.shape[1];
  api.dtype = ( sg_data_type_t ) inDesc.dtype;
  assert ( api.dtype == SG_DTYPE_FP32 || api.dtype == SG_DTYPE_FP16 );
  assert ( targetDesc.dtype == 31 || targetDesc.dtype == SG_DTYPE_INT32 );
  if ( targetDesc.dtype == 31 )
  {
    api.target_is_int64 = 1;
  }
  else
  {
    api.target_is_int64 = 0;
  }
  assert ( targetDesc.ndims == 1 );
  assert ( targetDesc.shape[0] == api.batch_num );
  if ( has_weight )
  {
    assert ( weightDesc.ndims == 1 );
    assert ( weightDesc.shape[0] == api.class_num );
    assert ( weightDesc.dtype == inDesc.dtype );
    api.weight_global_addr = ( unsigned long long ) weight;
  }
  else
  {
    api.weight_global_addr = 0;
  }
  assert ( reduction == Mean_Reduction || reduction == Sum_Reduction );
  if ( reduction == Mean_Reduction )
  {
    assert ( weight == nullptr );
    api.reduction = 0;
  }
  else if ( reduction == Sum_Reduction )
  {
    api.reduction = 1;
  }
  assert ( gradinDesc.dtype == inDesc.dtype );
  assert ( gradinDesc.ndims == inDesc.ndims );
  for ( int i = 0; i < inDesc.ndims; ++i )
  {
    assert ( gradinDesc.shape[i] == inDesc.shape[i] );
  }
  assert ( gradoutDesc.ndims == 0 );
  api.label_smoothing = label_smoothing;
  api.input_global_addr = ( unsigned long long ) in;
  api.target_global_addr = ( unsigned long long ) target;
  api.grad_output_global_addr = ( unsigned long long ) gradout;
  api.grad_input_global_addr = ( unsigned long long ) gradin;
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_cross_entropy_loss_backward", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

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
int                              R_transpose )
{
  assert ( L_transpose == 0 );
  assert ( LDesc.ndims == 2 && RDesc.ndims == 2 && YDesc.ndims == 2 );
  if ( B != nullptr )
  {
    assert ( BDesc.ndims == 1 );
  }
  assert ( LDesc.shape[0] == YDesc.shape[0] );
  assert ( RDesc.shape[1] == YDesc.shape[1] );
  assert ( LDesc.shape[1] == RDesc.shape[0] );
  if ( B != nullptr )
  {
    assert ( BDesc.shape[0] == YDesc.shape[1] );
  }
  assert ( LDesc.dtype == RDesc.dtype );
  assert ( LDesc.dtype == YDesc.dtype );
  if ( B != nullptr )
  {
    assert ( LDesc.dtype == BDesc.dtype );
  }
  sg_api_batch_matmul_t api;
  api.L_addr = ( unsigned long long ) L;
  api.R_addr = ( unsigned long long ) R;
  api.Y_addr = ( unsigned long long ) Y;
  api.B_addr = ( unsigned long long ) B;
  api.batch_num = 1;
  api.L_row_num = LDesc.shape[0];
  api.L_col_num = LDesc.shape[1];
  api.R_col_num = RDesc.shape[1];
  api.L_trans = L_transpose;
  api.R_trans = R_transpose;
  api.dtype = ( sg_data_type_t ) ( YDesc.dtype );
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_batch_matmul", &api, sizeof ( api ) );
  return BM_SUCCESS;
}

bm_status_t sgdnn_embedding_dense_backward (
bm_handle_t                       handle,
const TensorDescriptor_t          gradoutDesc,
const void                       *gradout,
const TensorDescriptor_t          indicesDesc,
const void                       *indices,
const TensorDescriptor_t          outDesc,
void                             *out,
int                               padding_idx,
bool                              scale_grad_by_freq ) {
  assert ( scale_grad_by_freq == false ); // not support scale_grad_by_freq now.
  sg_api_emb_backward_t api;
  api.gradout_global_addr          = ( unsigned long long ) gradout;
  api.index_global_addr            = ( unsigned long long ) indices;
  api.output_global_addr           = ( unsigned long long ) out;
  api.gradout_dim                  = gradoutDesc.ndims;
  api.idx_dim                      = indicesDesc.ndims;
  api.out_dim                      = outDesc.ndims;
  api.grad_dtype                   = ( sg_data_type_t ) gradoutDesc.dtype;
  api.idx_dtype                    = ( sg_data_type_t ) indicesDesc.dtype;
  for ( int i = 0; i < api.gradout_dim; i++ ) {
    api.gradout_shape[i] = gradoutDesc.shape[i];
  }
  for ( int i = 0; i < api.idx_dim; i++ ) {
    api.idx_shape[i] = indicesDesc.shape[i];
  }
  for ( int i = 0; i < api.out_dim; i++ ) {
    api.out_shape[i] = outDesc.shape[i];
  }
  assert ( api.idx_dtype ==  SG_DTYPE_INT32 );
  assert ( api.out_dim == 2 );
  assert ( api.gradout_shape[api.gradout_dim - 1] == api.out_shape[1] );
  for ( int i = 0; i < api.idx_dim; i++ ) {
    assert ( api.idx_shape[i] == api.gradout_shape[i] );
  }
  int window_size = 64;
  assert ( window_size % 64 == 0 );
  int NUM_Index = 1;
  for ( int i = 0; i < api.idx_dim; i++ ) { NUM_Index *= api.idx_shape[i]; }
  bm_device_mem_t sorted_index, sorted_index_index;
  bm_device_mem_t from_index, to_index;
  bm_status_t status;
  status = bm_malloc_device_byte ( handle, &sorted_index, NUM_Index * sizeof ( int ) );
  assert ( status == BM_SUCCESS );
  status = bm_malloc_device_byte ( handle, &sorted_index_index, NUM_Index * sizeof ( int ) );
  assert ( status == BM_SUCCESS );
  status = bm_malloc_device_byte ( handle, &from_index, window_size * sizeof ( int ) );
  assert ( status == BM_SUCCESS );
  status = bm_malloc_device_byte ( handle, &to_index, window_size * sizeof ( int ) );
  assert ( status == BM_SUCCESS );
  api.sorted_index_global_addr = bm_mem_get_device_addr ( sorted_index );
  api.sorted_index_index_global_addr = bm_mem_get_device_addr ( sorted_index_index );
  api.from_index_global_addr = bm_mem_get_device_addr ( from_index );
  api.to_index_global_addr = bm_mem_get_device_addr ( to_index );
  api.window_size = window_size;
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_emb_backward", &api, sizeof ( api ) );
  bm_free_device ( handle, sorted_index );
  bm_free_device ( handle, sorted_index_index );
  bm_free_device ( handle, from_index );
  bm_free_device ( handle, to_index );
  return BM_SUCCESS;
}

bm_status_t sgdnn_norm2 (
bm_handle_t                     handle,
const TensorDescriptor_t        inputDesc,
const void                     *input,
const TensorDescriptor_t        outputDesc,
void                           *output )
{
  sg_api_norm2_t api;
  api.dim = inputDesc.ndims;
  if ( outputDesc.ndims != 0 )
  {
    assert ( outputDesc.ndims == inputDesc.ndims );
    for ( int i = 0; i < outputDesc.ndims; ++i )
    {
      assert ( outputDesc.shape[i] == 1 );
    }
  }
  for ( int i = 0; i < inputDesc.ndims; ++i )
  {
    api.shape[i] = inputDesc.shape[i];
  }
  assert ( inputDesc.dtype == outputDesc.dtype );
  api.input_global_addr = ( unsigned long long ) input;
  api.output_global_addr = ( unsigned long long ) output;
  api.dtype = ( sg_data_type_t ) ( inputDesc.dtype );
  sgdnn_tpu_kernel_launch ( handle, "tpu_kernel_api_norm2", &api, sizeof ( api ) );
  return BM_SUCCESS;
}
