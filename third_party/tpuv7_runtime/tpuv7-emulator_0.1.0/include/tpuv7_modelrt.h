#ifndef __TPU_NET__
#define __TPU_NET__
#include "tpuv7_rt.h"

typedef unsigned long size_t;
/* tpurt_data_type_t holds the type for a scalar value */
typedef enum tpu_data_type_e {
  TPU_FLOAT32 = 0,
  TPU_FLOAT16 = 1,
  TPU_INT8 = 2,
  TPU_UINT8 = 3,
  TPU_INT16 = 4,
  TPU_UINT16 = 5,
  TPU_INT32 = 6,
  TPU_UINT32 = 7,
  TPU_BFLOAT16 = 8,
  TPU_INT4 = 9,
  TPU_UINT4 = 10,
} tpuRtDataType_t;

/* store location definitions */
typedef enum tpu_store_mode_e {
  TPU_STORE_IN_SYSTEM = 0, /* default, if not sure, use 0 */
  TPU_STORE_IN_DEVICE = 1,
} tpuRtStoreMode_t;

/* tpu_shape_t holds the shape info */
#define TPU_MAX_DIMS_NUM 8
typedef struct tpu_shape_s {
  int num_dims;
  int dims[TPU_MAX_DIMS_NUM];
} tpuRtShape_t;

/*
tpu_tensor_t holds a multi-dimensional array of elements of a single data type
and tensor are in device memory */
typedef struct tpu_tensor_s {
  tpuRtDataType_t dtype;
  tpuRtShape_t shape;
  void *data;//data in device mem
  unsigned char reserved[64];
} tpuRtTensor_t;

/* --------------------------------------------------------------------------*/
/* network information structure */

/* tpu_stage_info_t holds input/output shapes and device mems; every network can contain one or more
 * stages */
typedef struct tpu_stage_info_s {
  tpuRtShape_t * input_shapes;  /* input_shapes[0] / [1] / ... / [input_num-1] */
  tpuRtShape_t * output_shapes; /* output_shapes[0] / [1] / ... / [output_num-1] */
  void * input_mems;
  void * output_mems;
} tpuRtStageInfo_t;

/* tpu_tensor_info_t holds all information of one net.
 * scale for float type is 1.0 as default */
typedef struct tpu_net_info_s {
  char const* name;              /* net name */
  bool is_dynamic;               /* dynamic or static */
  int input_num;                 /* number of inputs */
  char const** input_names;      /* input_names[0] / [1] / .../ [input_num-1] */
  tpuRtDataType_t * input_dtypes;  /* input_dtypes[0] / [1] / .../ [input_num-1] */
  float * input_scales;           /* input_scales[0] / [1] / .../ [input_num-1] */
  int output_num;                /* number of outputs */
  char const** output_names;     /* output_names[0] / [1] / .../ [output_num-1] */
  tpuRtDataType_t const* output_dtypes; /* output_dtypes[0] / [1] / .../ [output_num-1] */
  float * output_scales;          /* output_scales[0] / [1] / .../ [output_num-1] */
  int stage_num;                 /* number of stages */
  tpuRtStageInfo_t * stages;       /* stages[0] / [1] / ... / [stage_num-1] */
  size_t * max_input_bytes;       /* max_input_bytes[0]/ [1] / ... / [input_num-1] */
  size_t * max_output_bytes;      /* max_output_bytes[0] / [1] / ... / [output_num-1] */
  int * input_zero_point;         /* input_zero_point[0] / [1] / .../ [input_num-1] */
  int * output_zero_point;        /* output_zero_point[0] / [1] / .../ [input_num-1] */
  unsigned char reserved[64];
} tpuRtNetInfo_t;

typedef void* tpuRtNet_t;
typedef void* tpuRtNetContext_t;

void * tpuRtInitNet(void);
tpuRtStatus_t tpuRtCreateNetContext(tpuRtNetContext_t *context);
tpuRtStatus_t tpuRtDestroyNetContext(tpuRtNetContext_t context);
tpuRtStatus_t tpuRtLoadNet(const char* net_path, tpuRtNetContext_t context, tpuRtNet_t *net);
tpuRtStatus_t tpuRtLoadNetFromMem(void* net_data, size_t size, tpuRtNet_t *net);
tpuRtStatus_t tpuRtUnloadNet(tpuRtNet_t *net);
tpuRtNetInfo_t tpuRtGetNetInfo(tpuRtNet_t *net);
//tpuRtStatus_t tpuRtGetNetInfo(tpuRtNet_t *net, tpuRtNetInfo_t *info);
//tpuRtNetInfo_t tpuRtGetNetInfo(tpuRtNet_t *net, tpuRtNetContext_t context);
tpuRtStatus_t tpuRtLaunchNetAsync(tpuRtNet_t* net, const tpuRtTensor_t input[], tpuRtTensor_t output[], const char*net_name, tpuRtStream_t stream);
tpuRtStatus_t tpuRtLaunchNet(tpuRtNet_t* net, const tpuRtTensor_t input[], tpuRtTensor_t output[], tpuRtStream_t stream);

#endif  //end of __TPU_NET__
