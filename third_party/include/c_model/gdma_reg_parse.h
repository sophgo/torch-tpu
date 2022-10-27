#ifndef __GDMA_REG_PARSE_H__
#define __GDMA_REG_PARSE_H__
#include "gdma_reg_def.h"
#include "gdma_reg_value.h"

typedef struct {
  u32 intr_en;
  u32 stride_enable;
  u32 nchw_copy;
  u32 cmd_short;
  u32 decompress_enable;
  u32 eng_sync_id_en;
  u32 cmd_id;
  u32 cmd_type;
  u32 special_function;
  u32 fill_constant_en;
  u32 src_data_format;
  u32 mask_data_format; // Also as index_data_format for DMA_nonzero
  u32 eng_sync_id;
  u32 constant_value;
  u32 src_nstride;
  u32 src_cstride; // Also as length for DMA_general
  u32 src_hstride;
  u32 src_wstride;
  u32 dst_nstride; // Also as bias_i for DMA_nonzero
  u32 dst_cstride;
  u32 dst_hstride;
  u32 dst_wstride;
  u32 src_nsize;
  u32 src_csize;
  u32 src_hsize;
  u32 src_wsize;
  u32 dst_nsize;
  u32 dst_csize;
  u32 dst_hsize;
  u32 dst_wsize;
  u32 src_start_addr_l32;
  u32 src_start_addr_h8;
  u32 dst_start_addr_l32;
  u32 dst_start_addr_h8;
  u32 mask_start_addr_l32;
  u32 mask_start_addr_h8;
  u32 localmem_mask_l32;  // Also as the index_start_addr_l32 for Gather/Scatter
  u32 localmem_mask_h32;  // Also as the index_start_addr_h8 for Gather/Scatter
  // Special field for DMA_gather and DMA_scatter
  // GDMA Des need to be reassigned when meet DMA_gather or DMA_scatter
  u32 index_cstride;
  u32 index_hstride;
  u32 index_csize;
  u32 index_hsize;
} gdma_des_t;

static inline void parse_gdma_des_struct(const P_COMMAND cmd, gdma_des_t *des) {
  des->intr_en = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_INTR_EN);
  des->stride_enable = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_STRIDE_ENABLE);
  des->nchw_copy = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_NCHW_COPY);
  des->cmd_short = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_CMD_SHORT);
  des->decompress_enable = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DECOMPRESS_ENABLE);
  des->eng_sync_id_en = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_CMD_ID_EN);
  des->cmd_id = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_CMD_ID);
  des->cmd_type = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_CMD_TYPE);
  des->special_function = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_CMD_SPECIAL_FUNCTION);
  des->fill_constant_en = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_FILL_CONSTANT_EN);
  des->src_data_format = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_DATA_FORMAT);
  des->mask_data_format = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_MASK_DATA_FORMAT);
  des->eng_sync_id = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_CMD_ID_DEP);
  des->constant_value = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_CONSTANT_VALUE);
  des->src_nstride = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_NSTRIDE);
  des->src_cstride = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_CSTRIDE);
  des->src_hstride = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_HSTRIDE);
  des->src_wstride = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_WSTRIDE);
  des->dst_nstride = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_NSTRIDE);
  des->dst_cstride = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_CSTRIDE);
  des->dst_hstride = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_HSTRIDE);
  des->dst_wstride = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_WSTRIDE);
  des->src_nsize = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_NSIZE);
  des->src_csize = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_CSIZE);
  des->src_hsize = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_HSIZE);
  des->src_wsize = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_WSIZE);
  des->dst_nsize = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_NSIZE);
  des->dst_csize = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_CSIZE);
  des->dst_hsize = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_HSIZE);
  des->dst_wsize = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_WSIZE);
  des->src_start_addr_l32 = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_START_ADDR_L32);
  des->src_start_addr_h8 = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_SRC_START_ADDR_H8);
  des->dst_start_addr_l32 = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_START_ADDR_L32);
  des->dst_start_addr_h8 = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_DST_START_ADDR_H8);
  des->mask_start_addr_l32 = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_MASK_START_ADDR_L32);
  des->mask_start_addr_h8 = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_MASK_START_ADDR_H8);
  des->localmem_mask_l32 = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_LOCALMEM_MASK_L32);
  des->localmem_mask_h32 = (u32)get_reg_id_val(cmd, (reg_id_t)GDMA_ID_LOCALMEM_MASK_H32);
  des->index_cstride = 0;
  des->index_hstride = 0;
  des->index_csize = 0;
  des->index_hsize = 0;

  if (des->cmd_type == GDMA_VALUE_TYPE_GATHER ||
      des->cmd_type == GDMA_VALUE_TYPE_SCATTER) {
    u32 src_c_str = des->src_nstride;
    u32 src_h_str = des->src_cstride;
    u32 dst_c_str = des->src_hstride;
    u32 dst_h_str = des->src_wstride;
    u32 index_c_str = des->dst_nstride;
    u32 index_h_str = des->dst_cstride;
    u32 src_c = (des->dst_hstride >> 16);
    u32 src_h = des->dst_wstride;
    u32 src_w = des->src_nsize;
    u32 dst_h = (des->src_wsize << 16) | (des->src_hsize & 0xffff);
    u32 index_c = (des->dst_csize);

    des->src_nstride = 0;
    des->src_cstride = src_c_str;
    des->src_hstride = src_h_str;
    des->src_wstride = 1;
    des->dst_nstride = 0;
    des->dst_cstride = dst_c_str;
    des->dst_hstride = dst_h_str;
    des->dst_wstride = 1;
    des->src_nsize = 1;
    des->src_csize = src_c;
    des->src_hsize = src_h;
    des->src_wsize = src_w;
    des->dst_nsize = 1;
    des->dst_csize = src_c > index_c ? src_c : index_c;
    des->dst_hsize = dst_h;
    des->dst_wsize = src_w;
    des->index_cstride = index_c_str;
    des->index_hstride = index_h_str;
    des->index_csize = index_c;
    des->index_hsize = des->cmd_type == GDMA_VALUE_TYPE_GATHER ? dst_h : src_h;
  } else if (des->cmd_type == GDMA_VALUE_TYPE_GENERAL) {
    if (des->special_function == 1) { // Broadcast
      des->special_function = GDMA_VALUE_FUNC_BROADCAST;
    }
  }
}

#endif  // __GDMA_REG_PARSE_H__

