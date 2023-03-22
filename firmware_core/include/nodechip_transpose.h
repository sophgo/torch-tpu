#ifndef NODECHIP_TRANSPOSE_H
#define NODECHIP_TRANSPOSE_H

#include "sg_api_struct.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"
#include "string.h"
#ifdef __cplusplus
extern "C" {
#endif

#define GET_OPTIMIZED_FACTORIZATION(factor, polynomial)       \
    {                                                         \
        factor = 0;                                           \
        if(polynomial > NPU_NUM){                             \
            for(int i = NPU_NUM; i >= (NPU_NUM/2); i--) {     \
                if(polynomial % i == 0) {                     \
                    factor = i;                               \
                    break;                                    \
                }                                             \
            }                                                 \
        }                                                     \
    }

#define HALF_LOCAL_MEM_SIZE  (LOCAL_MEM_SIZE >> 1)
#define QUARTER_LOCAL_MEM_SIZE  (LOCAL_MEM_SIZE >> 2)
typedef enum {
    TRANS_GENERAL = 0,
    TRANS_NPU_N_SWITCH_W,
    TRANS_GDMA_NCH,
    TRANS_NPU_H_SWITCH_W,
} trans_axis;

typedef struct {
    trans_axis trans_method;
    int N;
    int C;
    int H;
    int W;
    int max_trans_counts;
} trans_info_t;

trans_info_t get_transpose_info(int fixed_dim, int left_dim, int right_dim,  int  gdma_format);

void nodechip_permute_xyz2xzy(
        global_addr_t     input_global_addr,
        global_addr_t     output_global_addr,
        int               input_x,
        int               input_y,
        int               input_z,
        data_type_t       dtype
);

void nodechip_permute_tpu_nw(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        int           N,
        int           C,
        int           H,
        int           W,
        int           total_trans_times,
        data_type_t   dtype
);

void nodechip_permute_tpu_hw(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        int           N,
        int           C,
        int           H,
        int           W,
        int           total_trans_times,
        data_type_t   dtype
);

void nodechip_permute_gdma(
        global_addr_t    input_global_addr,
        global_addr_t    output_global_addr,
        const int*       input_shape,
        const int*       order,
        int              dims,
        data_type_t      dtype
);
/*
INLINE static bm_fw_status_t bm_api_cv_transpose(
    unsigned char              *api_buf,
    int                         size)
{
  bm_api_cv_transpose_t *api = (bm_api_cv_transpose_t*)api_buf;
  ASSERT(size == sizeof(bm_api_cv_transpose_t));
  int gdma_format = GDMA_VALUE_FORMAT_INT8;
  switch (api->type_len) {
    case 1: gdma_format = GDMA_VALUE_FORMAT_INT8; break;
    case 2: gdma_format = GDMA_VALUE_FORMAT_FLOAT16; break;
    case 4: gdma_format = GDMA_VALUE_FORMAT_FLOAT32; break;
    default: ASSERT(0);
  }
  CMD_ID_NODE id_node;
  resync_cmd_id(&id_node);
  nodechip_permute_xyz2xzy(
              api->input_global_mem_addr,
              api->output_global_mem_addr,
              api->channel,
              api->height,
              api->width,
              gdma_format,
              &id_node
              );
  poll_all_engine_done(&id_node);
  return BM_FW_SUCCESS;
}
*/
void nodechip_transpose(
    global_addr_t                input_global_addr,
    global_addr_t                output_global_addr,
    int*                         input_shape,
    int*                         order,
    int                          dims,
    global_addr_t                buffer_global_addr,
    unsigned long long*          buffer_size, //if not NULL, just calculate buffer_size, not compute
    data_type_t                  dtype
);

// inline static sg_fw_status_t sg_api_transpose(
//     unsigned char              *api_buf,
//     int                         size)
// {
//   sg_api_transpose_t *api = (sg_api_transpose_t*)api_buf;
//   TPUKERNEL_ASSERT(size == sizeof(sg_api_transpose_t));
//   tpu_initialize();
//   int input_shape[8]={0}, order[8]={0};
//   memcpy(input_shape, api->input_shape, sizeof(int) * 8);
//   memcpy(order, api->order, sizeof(int) * 8);
//   nodechip_transpose(
//               api->input_global_mem_addr,
//               api->output_global_mem_addr,
//               input_shape,
//               order,
//               api->dims,
//               api->buffer_global_mem_addr,
//               NULL,
//               tpu_type_convert(api->sgdtype)
//               );
//   tpu_poll();
//   return SG_FW_SUCCESS;
// }


#ifdef __cplusplus
}
#endif

#endif // NODECHIP_TRANSPOSE_H

