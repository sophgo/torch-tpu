
#include "sg_api_struct.h"
#include "tpu_kernel.h"

typedef struct pooling_secs_info {
  int nsecs;
  int hsecs;
  float Ratio;
} pooling_secs_info_t;

// stride>15
typedef struct pooling_secs_info_stride_gt_15
{
  int n_c_slice;
  int n_c_loops;
  int ohslice;
  int ohloops;
  int owslice;
  int owloops;
} pooling_secs_info_stride_gt_15_t;

extern void nodechip_pooling_local(
    local_addr_t    bottom_addr,
    local_addr_t    top_addr,
    const int*      bottom_dim,
    const int*      top_dim,
    int    kh,
    int    kw,
    int    up_pad_h,
    int    left_pad_w,
    int    down_pad_h,
    int    right_pad_w,
    int    stride_h,
    int    stride_w,
    int    dh,
    int    dw,
    int    is_avg_pooling,
    int    avg_pooling_mode,
    int    if_relu,
    float  relu_upper_limit,
    data_type_t dtype);

static void global_upsample_nearest2d_backward_data_split_stride_gt_15(
    const dim4 *ishape,
    const dim4 *oshape,
    const int kh,
    const int kw,
    const int cell_h,
    const int cell_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    data_type_t dtype,
    pooling_secs_info_stride_gt_15_t *p_secs)
{
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;

  p_secs->n_c_slice = ishape->n * ishape->c;
  p_secs->n_c_loops = 1;
  p_secs->ohslice = oshape->h;
  p_secs->ohloops = 1;
  p_secs->owslice = oshape->w;
  p_secs->owloops = 1;

  // local memory will be devided into 5 parts.
  // TODO: could apply another bank to load buffer
  // bank0: buffer + buffer2 + output_buffer,  bank1: ping-in, bank2:pong-in, bank3:ping-out, bank4:pong-out
  // step 01: attempt to load full feature map

  int ifmap_single_size =
      tpu_aligned_feature_size(ishape->h, ishape->w, dtype);
  int ofmap_single_size =
      tpu_aligned_feature_size(oshape->h, oshape->w, dtype);
  int buffer1_size = ALIGN(cell_h * oshape->h * ishape->w, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);
  int buffer2_size = ALIGN(cell_h * oshape->h * cell_w * oshape->w, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);
  int output_buffer_size = ofmap_single_size;

  //   int min_n_c_loops = 1;
  int max_n_c_loops = DIV_UP(ishape->n * ishape->c, NPU_NUM);
  int n_c_loops = 1;
  int n_c_slice = 1;

  for (n_c_loops = max_n_c_loops; n_c_loops >= 1; n_c_loops--)
  {
    // int n_c_sec = DIV_UP(ishape->n * ishape->c, NPU_NUM);
    n_c_slice = MIN(DIV_UP(max_n_c_loops, n_c_loops) * NPU_NUM, ishape->n * ishape->c);
    int local_mem = ALIGN(DIV_UP(n_c_slice, NPU_NUM) * ifmap_single_size, BANK_SIZE) * 2 +
                    ALIGN(DIV_UP(n_c_slice, NPU_NUM) * ofmap_single_size, BANK_SIZE) * 2 +
                    ALIGN(DIV_UP(n_c_slice, NPU_NUM) * (buffer1_size + buffer2_size + output_buffer_size), BANK_SIZE);
    if (local_mem > LOCAL_MEM_SIZE)
      break;
  }
  n_c_loops++;
  if (n_c_loops <= max_n_c_loops)
  {
    p_secs->n_c_slice = MIN(DIV_UP(max_n_c_loops, n_c_loops) * NPU_NUM, ishape->n * ishape->c);
    p_secs->n_c_loops = n_c_loops;
    return;
  }
  else
  {
    // step 02: attempt to split oh
    p_secs->n_c_slice = MIN(1 * NPU_NUM, ishape->n * ishape->c);
    p_secs->n_c_loops = max_n_c_loops;
    // int min_oh_loops = 1;
    int max_oh_loops = oshape->h;
    int ohloops = 1;
    int ohslice = 1;

    for (ohloops = max_oh_loops; ohloops >= 1; ohloops--)
    {
      ohslice = DIV_UP(max_oh_loops, ohloops);
      ifmap_single_size = tpu_aligned_feature_size(kh_ext + (ohslice - 1) * stride_h, ishape->w, dtype);
      ofmap_single_size = tpu_aligned_feature_size(ohslice, oshape->w, dtype);
      buffer1_size = ALIGN(cell_h * ohslice * ishape->w, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);
      buffer2_size = ALIGN(cell_h * ohslice * cell_w * oshape->w, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);
      output_buffer_size = ofmap_single_size;
      int local_mem = ALIGN(ifmap_single_size, BANK_SIZE) * 2 + ALIGN(ofmap_single_size, BANK_SIZE) * 2 +
                      ALIGN(buffer1_size + buffer2_size + output_buffer_size, BANK_SIZE);
      if (local_mem > LOCAL_MEM_SIZE)
        break;
    }
    ohloops++;
    if (ohloops <= max_oh_loops)
    {
      p_secs->ohslice = DIV_UP(max_oh_loops, ohloops);
      p_secs->ohloops = ohloops;
      return;
    }
    else
    {
      // step 03: attempt to split ow, with low efficiency of GDMA
      p_secs->ohslice = 1;
      p_secs->ohloops = oshape->h;
      //   int min_ow_loops = 1;
      int max_ow_loops = oshape->w;
      int owloops = 1;
      int owslice = 1;
      for (owloops = max_ow_loops; owloops >= 1; owloops--)
      {
        owslice = DIV_UP(max_ow_loops, owloops);
        ifmap_single_size = tpu_aligned_feature_size(kh_ext, kw_ext + (owslice - 1) * stride_w, dtype);
        ofmap_single_size = tpu_aligned_feature_size(1, owslice, dtype);
        // TODO: if ihslice=1, do not need buffer 1, buffer1_size = 0
        buffer1_size = ALIGN(cell_h * ohslice * ishape->w, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);
        buffer2_size = ALIGN(cell_h * 1 * cell_w * owslice, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);
        output_buffer_size = ofmap_single_size;

        int local_mem = ALIGN(ifmap_single_size, BANK_SIZE) * 2 + ALIGN(ofmap_single_size, BANK_SIZE) * 2 +
                        ALIGN(buffer1_size + buffer2_size + output_buffer_size, BANK_SIZE);
        if (local_mem > LOCAL_MEM_SIZE)
          break;
      }
      owloops++;
      if (owloops <= max_ow_loops)
      {
        p_secs->owslice = DIV_UP(max_ow_loops, owloops);
        p_secs->owloops = owloops;
        return;
      }
      else
      {
        TPUKERNEL_LOG("Not supported pooling parameters.\n");
        TPUKERNEL_ASSERT(0);
      }
    }
  }
}

static void extract_w_pooling(
    local_addr_t buffer_local_addr,
    local_addr_t buffer_local_addr2,
    local_addr_t output_buffer_addr,
    local_addr_t output_addr,
    const dim2 *kernel,
    const dim2 *stride,
    const dim2 *dilation,
    const padding_t *pad,
    dim2 cur_pooling_kernel,
    dim2 cur_pooling_stride,
    dim2 cur_pooling_dilation,
    dim4 cur_pooling_shape,
    padding_t cur_pooling_pad,
    dim4 cur_output_shape,
    const int cell_w,
    const int kw_ext,
    const int iw_used, // iwslice
    const int ic,
    bool h_first_cell,
    data_type_t dtype,
    const int is_avg_pooling,
    const int avg_pooling_mode)
{
  bool w_first_cell = true;
  if (stride->w <= 15 && dilation->w <= 15)
  {
    // w_slice do not need copy, skip w-dim extraction
    cur_pooling_shape.w = iw_used;
    cur_pooling_kernel.w = kernel->w;
    cur_pooling_pad.left = pad->left;
    cur_pooling_pad.right = pad->right;
    cur_pooling_stride.w = stride->w;
    cur_pooling_dilation.w = dilation->w;
    dim4 cur_input_stride;
    cur_input_stride.w = 1;
    cur_input_stride.h = iw_used;
    cur_input_stride.c = ALIGN(cur_pooling_shape.h * cur_input_stride.h, tpu_eu_num(dtype));
    cur_input_stride.n = DIV_UP(ic, NPU_NUM) * cur_input_stride.c;
    // 3.pooling
    if (is_avg_pooling)
    {
      scalar_t scale = {.f32 = 1.f / (cur_pooling_kernel.h * cur_pooling_kernel.w)};
      if (h_first_cell && w_first_cell)
      {
        // initialize output //
        tpu_bdc_fp_avg_pool2d(
            output_addr, // do not need output buffer
            buffer_local_addr,
            &cur_pooling_shape,
            &cur_pooling_kernel,
            &cur_pooling_pad,
            &cur_pooling_stride,
            &cur_pooling_dilation,
            dtype,
            tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
        w_first_cell = false;
      }
      else
      {
        // add cell output
        tpu_bdc_fp_avg_pool2d(
            output_buffer_addr, // need output buffer
            buffer_local_addr,
            &cur_pooling_shape,
            &cur_pooling_kernel,
            &cur_pooling_pad,
            &cur_pooling_stride,
            &cur_pooling_dilation,
            dtype,
            tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
        tpu_bdc_fp_add(
            output_addr,
            output_addr,
            output_buffer_addr,
            &cur_output_shape,
            NULL, NULL, NULL, dtype);
      }
    }
    else
    {
      // max pooling
      scalar_t pad_value = {.u32 = FP_NEG_MAX(dtype)};
      if (h_first_cell && w_first_cell)
      {
        tpu_bdc_fp_max_pool2d(
            output_addr,
            buffer_local_addr,
            &cur_pooling_shape,
            &cur_pooling_kernel,
            &cur_pooling_pad,
            &cur_pooling_stride,
            &cur_pooling_dilation,
            dtype,
            pad_value);
        w_first_cell = false;
      }
      else
      {
        tpu_bdc_fp_max_pool2d(
            output_buffer_addr,
            buffer_local_addr,
            &cur_pooling_shape,
            &cur_pooling_kernel,
            &cur_pooling_pad,
            &cur_pooling_stride,
            &cur_pooling_dilation,
            dtype,
            pad_value);
        // cell max
        tpu_bdc_max(
            output_addr,
            output_addr,
            output_buffer_addr,
            &cur_output_shape,
            NULL, NULL, NULL, dtype);
      }
    }
  }
  else if (stride->w <= 15 && dilation->w > 15 && kernel->w > 15)
  {
    /* code TODO: */
    TPUKERNEL_ASSERT_INFO(false, "not support now");
  }
  else
  {
    // 2.second extract: W
    for (int cell_w_idx = 0; cell_w_idx < (kernel->w / cell_w); cell_w_idx++)
    {
      // each cell data have different padding
      cur_pooling_pad.left = 0;
      cur_pooling_pad.right = 0;
      int iwslice_ext = iw_used + pad->left + pad->right;
      int w_cpy_times = (iwslice_ext - kw_ext) / stride->w + 1; // output->w
      int cell_w_ext = cell_w * dilation->w;
      dim4 cpy_src_stride, cpy_dst_stride;
      cpy_src_stride.w = dilation->w;
      cpy_src_stride.h = iw_used;
      cpy_src_stride.c = ALIGN(cur_pooling_shape.h * iw_used, tpu_eu_num(dtype));
      cpy_src_stride.n = DIV_UP(ic, NPU_NUM) * cpy_src_stride.c;
      cpy_dst_stride.w = 1;
      cpy_dst_stride.h = cell_w * w_cpy_times;
      // dst_stride.h need reshape, if has padding
      if (pad->left > 0 && cell_w_idx * cell_w_ext < pad->left)
      {
        TPUKERNEL_ASSERT(pad->left <= 15); // if pad > 16, need more complicated implement.
        int zero_pad = DIV_UP(MIN(pad->left, (cell_w_idx + 1) * cell_w_ext) - cell_w_idx * cell_w_ext,
                              dilation->w);
        zero_pad = MIN(cell_w, zero_pad);
        zero_pad = MAX(0, zero_pad);
        cpy_dst_stride.h -= zero_pad;
      }
      if (pad->right > 0 && ((w_cpy_times - 1) * stride->w + (cell_w_idx + 1) * cell_w_ext) > (pad->left + iw_used))
      {
        int zero_pad = ((w_cpy_times - 1) * stride->w + (cell_w_idx + 1) * cell_w_ext - pad->left - iw_used) / dilation->w;
        zero_pad = MIN(cell_w, zero_pad);
        zero_pad = MAX(0, zero_pad);
        cpy_dst_stride.h -= zero_pad;
      }
      cpy_dst_stride.c = ALIGN(cur_pooling_shape.h * cpy_dst_stride.h, tpu_eu_num(dtype));
      cpy_dst_stride.n = DIV_UP(ic, NPU_NUM) * cpy_dst_stride.c;

      // cpy extract
      unsigned int w_dst_offset = 0;
      for (int stride_idx = 0; stride_idx < w_cpy_times; stride_idx++)
      {
        dim4 cpy_shape = {1, ic, cur_pooling_shape.h, cell_w};
        int left_pad = 0;
        if (pad->left > 0 &&
            stride_idx * stride->w + cell_w_idx * cell_w_ext < pad->left)
        { // left has padding
          int zero_pad = DIV_UP(
              MIN(pad->left, stride_idx * stride->w + cell_w_idx * cell_w_ext + cell_w_ext) -
                  (stride_idx * stride->w + cell_w_idx * cell_w_ext),
              dilation->w); // end_bound - start_bound
          zero_pad = MIN(cell_w, zero_pad);
          zero_pad = MAX(0, zero_pad);
          left_pad = MAX(0, zero_pad);
          cpy_shape.w = MAX(0, cell_w - zero_pad); // if cell_w - slice_pad < 0, need not to cpy
          cur_pooling_pad.left += zero_pad;
        }

        if (pad->right > 0 &&
            stride_idx * stride->w + cell_w_idx * cell_w_ext + cell_w_ext >
                (pad->left + iw_used))
        { // right have padding case
          int zero_pad = ((stride_idx * stride->w + cell_w_idx * cell_w_ext + cell_w_ext) - pad->left - iw_used) / dilation->w;
          zero_pad = MIN(cell_w, zero_pad);
          zero_pad = MAX(0, zero_pad);
          cpy_shape.w = MAX(0, cpy_shape.w - zero_pad);
          cur_pooling_pad.right += zero_pad;
        }
        if (cpy_shape.w == 0 || cpy_shape.h == 0)
          continue; // origin pooling need padding, and
        tpu_bdc_cpy(
            buffer_local_addr2 + w_dst_offset * tpu_data_type_size(dtype), // dst_addr
            buffer_local_addr +
                MAX(0, (stride_idx * stride->w - pad->left + cell_w_idx * cell_w_ext + left_pad * dilation->w)) * tpu_data_type_size(dtype), // src_addr
            &cpy_shape,
            &cpy_dst_stride,
            &cpy_src_stride,
            dtype);
        w_dst_offset += cpy_shape.w;
      }
      cur_pooling_shape.w = cell_w * ((iwslice_ext - kw_ext) / stride->w + 1) - cur_pooling_pad.left - cur_pooling_pad.right;
      if (cur_pooling_shape.w == 0)
        continue;

      // 3.pooling
      if (is_avg_pooling)
      {
        scalar_t scale = {.f32 = 1.f / (cur_pooling_kernel.h * cur_pooling_kernel.w)};
        if (h_first_cell && w_first_cell)
        {
          tpu_bdc_fp_avg_pool2d(
              output_addr, // do not need buffer
              buffer_local_addr2,
              &cur_pooling_shape,
              &cur_pooling_kernel,
              &cur_pooling_pad,
              &cur_pooling_stride,
              &cur_pooling_dilation,
              dtype,
              tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
          w_first_cell = false;
        }
        else
        {
          tpu_bdc_fp_avg_pool2d(
              output_buffer_addr,
              buffer_local_addr2,
              &cur_pooling_shape,
              &cur_pooling_kernel,
              &cur_pooling_pad,
              &cur_pooling_stride,
              &cur_pooling_dilation,
              dtype,
              tpu_cast(scale, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
          // add cell output
          tpu_bdc_fp_add(
              output_addr,
              output_addr,
              output_buffer_addr,
              &cur_output_shape,
              NULL, NULL, NULL, dtype);
        }
      }
      else
      {
        // max pooling
        scalar_t pad_value = {.u32 = FP_NEG_MAX(dtype)};
        if (h_first_cell && w_first_cell)
        {
          tpu_bdc_fp_max_pool2d(
              output_addr,
              buffer_local_addr2,
              &cur_pooling_shape,
              &cur_pooling_kernel,
              &cur_pooling_pad,
              &cur_pooling_stride,
              &cur_pooling_dilation,
              dtype,
              pad_value);
          w_first_cell = false;
        }
        else
        {
          tpu_bdc_fp_max_pool2d(
              output_buffer_addr,
              buffer_local_addr2,
              &cur_pooling_shape,
              &cur_pooling_kernel,
              &cur_pooling_pad,
              &cur_pooling_stride,
              &cur_pooling_dilation,
              dtype,
              pad_value);
          // cell max
          tpu_bdc_max(
              output_addr,
              output_addr,
              output_buffer_addr,
              &cur_output_shape,
              NULL, NULL, NULL, dtype);
        }
      }
    }
  }
}



void nodechip_upsample_nearest2d_backward_parallel_special_case(
    global_addr_t ifmap_offset_global,
    global_addr_t ofmap_offset_global,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int kh,
    int kw,
    int pad_h,
    int pad_w,
    int pad_h_after,
    int pad_w_after,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int is_avg_pooling,
    int avg_pooling_mode,
    int if_relu,
    float relu_upper_limit,
    int scale,
    data_type_t dtype)
{
  int kh_ext = dilation_h * (kh - 1) + 1;
  int kw_ext = dilation_w * (kw - 1) + 1;
  int ih_ext = (input_h - 1) + pad_h + pad_h_after + 1;
  int iw_ext = (input_w - 1) + pad_w + pad_w_after + 1;
  const scalar_t scale_c = { .f32 = scale };

  // for ceil_mode
  bool bottom_ceil_mode = false;
  bool right_ceil_mode = false;
  int pad_bottom_org = pad_h_after;
  int pad_right_org = pad_w_after;

  // bottom ceil_mode
  if (output_h > (ih_ext - kh_ext) / stride_h + 1)
  {
    int ceil_h_ext = (output_h - 1) * stride_h + kh_ext;
    pad_h_after += ceil_h_ext - ih_ext;
    bottom_ceil_mode = true;
  }
  // right ceil_mode
  if (output_w > (iw_ext - kw_ext) / stride_w + 1)
  {
    int ceil_w_ext = (output_w - 1) * stride_w + kw_ext;
    pad_w_after += ceil_w_ext - iw_ext;
    right_ceil_mode = true;
  }

  // cal pooling cell, if stride > 15. cell = kernel's max factor in [0,16)
  int cell_h = kh, cell_w = kw;
  if (stride_h > 15){
    for (int i = 15; i >= 1; i--){
      if (kh % i == 0){
        cell_h = i;break;
      }
    }
  }
  if (stride_w > 15){
    for (int i = 15; i >= 1; i--){
      if (kw % i == 0){
        cell_w = i;break;
      }
    }
  }

  // 1. split input
  pooling_secs_info_stride_gt_15_t secs_info;
  dim4 ishape = {input_n, input_c, input_h, input_w};
  dim4 oshape = {input_n, input_c, output_h, output_w};
  // local memory will be devided into 5 parts.
  // bank0: buffer + buffer2 + output_buffer,  bank1: ping-in, bank2:pong-in, bank3:ping-out, bank4:pong-out
  global_upsample_nearest2d_backward_data_split_stride_gt_15(
      &ishape,
      &oshape,
      kh,
      kw,
      cell_h,
      cell_w,
      stride_h,
      stride_w,
      dilation_h,
      dilation_w,
      dtype,
      &secs_info);

  int n_c_slice = secs_info.n_c_slice;
  int ohslice = secs_info.ohslice;
  int owslice = secs_info.owslice;
  int ihslice = MIN((ohslice - 1) * stride_h + kh_ext, ishape.h);
  int iwslice = MIN((owslice - 1) * stride_w + kw_ext, ishape.w);

  // get input/output's size per NPU
  unsigned int isize = DIV_UP(n_c_slice, NPU_NUM) *
                       tpu_aligned_feature_size(ihslice, iwslice, dtype);
  unsigned int osize = DIV_UP(n_c_slice, NPU_NUM) *
                       tpu_aligned_feature_size(ohslice, owslice, dtype); // pooling result need add, so mid type only can be FP32

  unsigned int buffer1_size = DIV_UP(n_c_slice, NPU_NUM) * ALIGN(cell_h * ohslice * ishape.w, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);
  unsigned int buffer2_size = DIV_UP(n_c_slice, NPU_NUM) * ALIGN(cell_h * ohslice * cell_w * owslice, tpu_eu_num(dtype)) * tpu_data_type_size(dtype);
  unsigned int output_buffer_size = osize;

//   printf("buffer1_size = %d \n", buffer1_size);
//   printf("buffer2_size = %d \n", buffer2_size);
//   printf("output_buffer_size = %d \n", output_buffer_size);
//   printf("isize = %d \n", isize);
//   printf("osize = %d \n", osize);

  local_addr_t buffer_addr = 0;
  local_addr_t buffer_addr2 = buffer_addr + buffer1_size;
  local_addr_t output_buffer_addr = buffer_addr2 + buffer2_size;
  local_addr_t iaddr_ping = ALIGN(output_buffer_addr + output_buffer_size, BANK_SIZE);
  local_addr_t iaddr_pong = ALIGN(iaddr_ping + isize, BANK_SIZE);
  local_addr_t oaddr_ping = ALIGN(iaddr_pong + isize, BANK_SIZE);
  local_addr_t oaddr_pong = ALIGN(oaddr_ping + osize, BANK_SIZE);

  TPUKERNEL_ASSERT(oaddr_pong + osize <= (unsigned int)LOCAL_MEM_SIZE);

  bool ping = true;
  //  int last_n_c_start = 0;
  int last_n_c_slice = 0;
  int last_ohstart = 0;
  int last_ohslice = 0;
  int last_owstart = 0;
  int last_owslice = 0;

  //(in,ic,ih,iw)->(1, in*ic, ih, iw)
  dim4 istride = {
      ishape.n * ishape.c * ishape.h * ishape.w,
      ishape.h * ishape.w,
      ishape.w,
      1};
  dim4 ostride = {
      ishape.n * ishape.c * output_h * output_w,
      output_h * output_w,
      output_w,
      1};
  padding_t pad = {.top = pad_h, .bottom = pad_h_after, .left = pad_w, .right = pad_w_after};
  dim2 kernel = {.h = kh, .w = kw};
  dim2 stride = {.h = stride_h, .w = stride_w};
  dim2 dilation = {.h = dilation_h, .w = dilation_w};

  int n_c_start = 0;
  int n_c_end = 0;
  // 1. split N_C

  for (int n_c_idx = 0; n_c_idx < secs_info.n_c_loops; n_c_idx++)
  {
    // nc boundary: [n_c_start, n_c_end)
    n_c_start = n_c_end;
    n_c_slice = (n_c_idx == (secs_info.n_c_loops - 1)) ? (ishape.n * ishape.c - n_c_end) : n_c_slice;
    n_c_end = n_c_start + n_c_slice;

    // ping pong
    bool parallel_branch = false;
    int ohstart = 0;
    int ohend = 0;
    padding_t slice_pad;
    // 2. OH dim
    for (int ohidx = 0; ohidx < secs_info.ohloops; ohidx++)
    {
      // oh boundary: [ohstart, ohend)
      // ih boundary: [ihstart, ihend)
      ohstart = ohend;
      ohslice = output_h / secs_info.ohloops + ((output_h % secs_info.ohloops) > ohidx);
      ohend = ohstart + ohslice;
      int ihstart = ohstart * stride_h - pad.top;
      int ihend = (ohend - 1) * stride_h + kh_ext - pad.top;
      slice_pad.top = ihstart < 0 ? -ihstart : 0;
      slice_pad.bottom = ihend > ishape.h ? ihend - ishape.h : 0;
      ihstart = ihstart < 0 ? 0 : ihstart;
      ihend = ihend > ishape.h ? ishape.h : ihend;
      ihslice = ihend - ihstart;

      int owstart = 0;
      int owend = 0;
      // 3. OW
      for (int owidx = 0; owidx < secs_info.owloops; owidx++)
      {
        // ow boundary: [owstart, owend)
        // iw boundary: [iwstart, iwend)
        owstart = owend;
        owslice = output_w / secs_info.owloops + ((output_w % secs_info.owloops) > owidx);
        owend = owstart + owslice;
        int iwstart = owstart * stride_w - pad.left;
        int iwend = (owend - 1) * stride_w + kw_ext - pad.left;
        slice_pad.left = iwstart < 0 ? -iwstart : 0;
        slice_pad.right = iwend > ishape.w ? iwend - ishape.w : 0;
        iwstart = iwstart < 0 ? 0 : iwstart;
        iwend = iwend > ishape.w ? ishape.w : iwend;
        iwslice = iwend - iwstart;

        // move input to local memory
        dim4 islice_shape = {1, n_c_slice, ihslice, iwslice};
        dim4 oslice_shape = {1, n_c_slice, ohslice, owslice};
        tpu_gdma_cpy_S2L(
            ping ? iaddr_ping : iaddr_pong,
            ifmap_offset_global + (n_c_start * istride.c +
                                   ihstart * istride.h + iwstart) *
                                      tpu_data_type_size(dtype),
            &islice_shape,
            NULL,
            &istride,
            dtype);
        // ping pong
        if (parallel_branch)
        {
          tpu_parallel_end();
        }
        tpu_parallel_start();
        parallel_branch = true;

        // pooling//
        if (stride_h <= 15 && dilation_h <= 15)
        {
          // h_slice do not need copy
          dim2 cur_pooling_kernel = {.h = kh,
                                     .w = stride_w > 15 ? cell_w : kw};
          dim2 cur_pooling_stride = {.h = stride_h,
                                     .w = stride_w > 15 ? cell_w : stride_w};
          dim2 cur_pooling_dilation = {.h = dilation_h, .w = stride_w > 15 ? 1 : dilation_w};

          extract_w_pooling(
              ping ? iaddr_ping : iaddr_pong, // buffer1
              buffer_addr2,                   // buffer2
              output_buffer_addr,
              ping ? oaddr_ping : oaddr_pong,
              &kernel,
              &stride,
              &dilation,
              &slice_pad,
              cur_pooling_kernel,
              cur_pooling_stride,
              cur_pooling_dilation,
              islice_shape, // cur_pooling_shape
              slice_pad,    // cur_pooling_pad
              oslice_shape,
              cell_w,
              kw_ext,
              iwslice,   // iw_used
              n_c_slice, // ic
              true,      // cell_h_idx=0, only one h_cell
              dtype,
              is_avg_pooling,
              avg_pooling_mode);
        }
        else if (stride_h <= 15 && dilation_h > 15 && kh > 15)
        {
          /* code TODO: */
          TPUKERNEL_LOG("Not supported pooling parameters.\n");
          TPUKERNEL_ASSERT(0);
        }
        else
        {
          dim2 cur_pooling_kernel = {.h = cell_h,
                                     .w = stride_w > 15 ? cell_w : kernel.w};
          dim2 cur_pooling_stride = {.h = cell_h,
                                     .w = cell_w};
          dim2 cur_pooling_dilation = {.h = 1, .w = stride_w > 15 ? 1 : dilation_w};
          // 1. first extract: H
          bool h_first_cell = true;
          for (int cell_h_idx = 0; cell_h_idx < (kh / cell_h); cell_h_idx++)
          {
            int cell_h_ext = cell_h * dilation_h;
            dim4 cur_pooling_shape = {.n = 1, .c = n_c_slice};
            padding_t cur_pooling_pad = {.top = 0, .bottom = 0, .left = 0, .right = 0};
            dim4 cpy_src_stride, cpy_dst_stride;
            cpy_src_stride.w = 1;
            cpy_src_stride.h = dilation_h * iwslice;
            cpy_src_stride.c = ALIGN(ihslice * cpy_src_stride.h, tpu_eu_num(dtype));
            cpy_src_stride.n = DIV_UP(n_c_slice, NPU_NUM) * cpy_src_stride.c;
            cpy_dst_stride.w = 1;
            cpy_dst_stride.h = iwslice;
            int c_stride = cell_h * DIV_UP(ihslice, stride_h);

            if (slice_pad.top > 0 && cell_h_idx * cell_h_ext < slice_pad.top)
            { // top has padding
                int zero_pad = DIV_UP(
                    MIN(slice_pad.top, cell_h_idx * cell_h_ext + cell_h_ext) - cell_h_idx * cell_h_ext,
                    dilation_h);
                zero_pad = MIN(cell_h, zero_pad);
                zero_pad = MAX(0, zero_pad);
                c_stride -= zero_pad;
            }
            if (slice_pad.bottom > 0 && ((oslice_shape.h - 1) * stride.h + cell_h_idx * cell_h_ext + cell_h_ext) >
                                            (slice_pad.top + ihslice))
            { // bottom has padding
                int zero_pad = (((oslice_shape.h - 1) * stride.h + cell_h_idx * cell_h_ext + cell_h_ext) - slice_pad.top - ihslice) / dilation_h;
                zero_pad = MIN(cell_h, zero_pad);
                zero_pad = MAX(0, zero_pad);
                c_stride -= zero_pad;
            }
            cpy_dst_stride.c = ALIGN(iwslice * c_stride,
                                     tpu_eu_num(dtype));
            cpy_dst_stride.n = DIV_UP(n_c_slice, NPU_NUM) * cpy_dst_stride.c;

            unsigned int h_dst_offset = 0;
            for (int stride_idx = 0; stride_idx < oslice_shape.h; stride_idx++)
            {
                dim4 cpy_shape = {1, n_c_slice, cell_h, iwslice};
                int top_pad = 0;
                if (slice_pad.top > 0 &&
                    stride_idx * stride_h + cell_h_idx * cell_h_ext < slice_pad.top)
                { // top has padding
                    int zero_pad = DIV_UP(
                        MIN(slice_pad.top, stride_idx * stride_h + cell_h_idx * cell_h_ext + cell_h_ext) -
                            (stride_idx * stride_h + cell_h_idx * cell_h_ext),
                        dilation_h);
                    zero_pad = MIN(cell_h, zero_pad);
                    zero_pad = MAX(0, zero_pad);
                    top_pad = zero_pad;
                    cpy_shape.h = MAX(0, cell_h - zero_pad);
                    cur_pooling_pad.top += zero_pad;
                }
                if (slice_pad.bottom > 0 &&
                    stride_idx * stride_h + cell_h_idx * cell_h_ext + cell_h_ext >
                        (slice_pad.top + ihslice))
                { // bottom has padding
                    int zero_pad = ((stride_idx * stride_h + cell_h_idx * cell_h_ext + cell_h_ext) - slice_pad.top - ihslice) / dilation_h;
                    zero_pad = MIN(cell_h, zero_pad);
                    zero_pad = MAX(0, zero_pad);
                    cpy_shape.h = MAX(0, cpy_shape.h - zero_pad);
                    cur_pooling_pad.bottom += zero_pad;
                }
                if (cpy_shape.w == 0 || cpy_shape.h == 0)
                    continue;
                tpu_bdc_cpy(
                    buffer_addr + h_dst_offset * tpu_data_type_size(dtype), // dst_addr
                    (ping ? iaddr_ping : iaddr_pong) +
                        MAX(0, (stride_idx * stride_h - slice_pad.top + cell_h_idx * cell_h_ext + top_pad * dilation_h)) * iwslice * tpu_data_type_size(dtype), // src_addr
                    &cpy_shape,
                    &cpy_dst_stride,
                    &cpy_src_stride,
                    dtype);
                h_dst_offset += (cpy_shape.h * iwslice);
            }
            cur_pooling_shape.h = cell_h * oslice_shape.h - cur_pooling_pad.top - cur_pooling_pad.bottom;
            if (cur_pooling_shape.h == 0)
                continue; // cell is in padding

            extract_w_pooling(
                buffer_addr,  // buffer1
                buffer_addr2, // buffer2
                output_buffer_addr,
                ping ? oaddr_ping : oaddr_pong,
                &kernel,
                &stride,
                &dilation,
                &slice_pad,
                cur_pooling_kernel,
                cur_pooling_stride,
                cur_pooling_dilation,
                cur_pooling_shape, // cur_pooling_shape
                cur_pooling_pad,   // cur_pooling_pad
                oslice_shape,      // cur_output_shape
                cell_w,
                kw_ext,
                iwslice,      // iw_used
                n_c_slice,    // ic
                h_first_cell, // h_first_cell
                dtype,
                is_avg_pooling,
                avg_pooling_mode);
            h_first_cell = false;
          }
        }

        int cell_num = kh * kw / cell_h / cell_w;
        if (is_avg_pooling)
        {
          scalar_t scale = {.f32 = 1.f / cell_num};
          tpu_bdc_fp_mul_C(
              ping ? oaddr_ping : oaddr_pong,
              ping ? oaddr_ping : oaddr_pong,
              scale,
              &oslice_shape,
              NULL,
              NULL,
              dtype);
          dim4 compensate_stride;
          tpu_aligned_stride(&compensate_stride, 0, &oslice_shape, dtype);
          if (avg_pooling_mode == 1)
          {
            // vertical pad compensate
            for (int y = 0; y < oslice_shape.h; y++)
            {
                int h = kh;
                int h0 = y * stride_h - slice_pad.top;
                int h1 = h0 + kh;
                if (h0 < 0)
                    h -= -h0;
                if (h1 > islice_shape.h)
                    h -= (h1 - islice_shape.h);
                if (h == kh)
                    continue;

                scalar_t scale_factor = {.f32 = (float)kh / h};
                dim4 shape1 = {.n = oslice_shape.n, .c = oslice_shape.c, .h = 1, .w = oslice_shape.w};
                tpu_bdc_fp_mul_C(
                    (ping ? oaddr_ping : oaddr_pong) + y * oslice_shape.w * tpu_data_type_size(dtype),
                    (ping ? oaddr_ping : oaddr_pong) + y * oslice_shape.w * tpu_data_type_size(dtype),
                    tpu_cast(scale_factor, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                    &shape1,
                    &compensate_stride, &compensate_stride,
                    dtype);
            }
            // horizontal pad compensate
            for (int x = 0; x < oslice_shape.w; x++)
            {
                int w = kw;
                int w0 = x * stride_w - slice_pad.left;
                int w1 = w0 + kw;
                if (w0 < 0)
                    w -= -w0;
                if (w1 > islice_shape.w)
                    w -= (w1 - islice_shape.w);
                if (w == kw)
                    continue;

                scalar_t scale_factor = {.f32 = (float)kw / w};
                dim4 shape1 = {.n = oslice_shape.n, .c = oslice_shape.c, .h = oslice_shape.h, .w = 1};
                tpu_bdc_fp_mul_C(
                    (ping ? oaddr_ping : oaddr_pong) + x * tpu_data_type_size(dtype),
                    (ping ? oaddr_ping : oaddr_pong) + x * tpu_data_type_size(dtype),
                    tpu_cast(scale_factor, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                    &shape1,
                    &compensate_stride, &compensate_stride,
                    dtype);
            }
          }
          else if (avg_pooling_mode == 0)
          {
            // Bottom compensate
            if (bottom_ceil_mode && slice_pad.bottom > 0)
            {
                scalar_t scale_factor =
                    {.f32 = (float)kh / (kh - slice_pad.bottom + pad_bottom_org)};
                dim4 shape1 = {.n = oslice_shape.n, .c = oslice_shape.c, .h = 1, .w = oslice_shape.w};
                tpu_bdc_fp_mul_C(
                    (ping ? oaddr_ping : oaddr_pong) + (oslice_shape.h - 1) * oslice_shape.w * tpu_data_type_size(dtype),
                    (ping ? oaddr_ping : oaddr_pong) + (oslice_shape.h - 1) * oslice_shape.w * tpu_data_type_size(dtype),
                    tpu_cast(scale_factor, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                    &shape1,
                    &compensate_stride, &compensate_stride,
                    dtype);
            }
            // Right compensate
            if (right_ceil_mode && slice_pad.right > 0)
            {
                scalar_t scale_factor =
                    {.f32 = (float)kw / (kw - slice_pad.right + pad_right_org)};
                dim4 shape1 = {.n = oslice_shape.n, .c = oslice_shape.c, .h = oslice_shape.h, .w = 1};
                tpu_bdc_fp_mul_C(
                    (ping ? oaddr_ping : oaddr_pong) + (oslice_shape.w - 1) * tpu_data_type_size(dtype),
                    (ping ? oaddr_ping : oaddr_pong) + (oslice_shape.w - 1) * tpu_data_type_size(dtype),
                    tpu_cast(scale_factor, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                    &shape1,
                    &compensate_stride, &compensate_stride,
                    dtype);
            }
          }
        }
        if (if_relu)
        {
          scalar_t C = {.f32 = 0};
          tpu_bdc_max_C(
              ping ? oaddr_ping : oaddr_pong,
              ping ? oaddr_ping : oaddr_pong,
              tpu_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
              &oslice_shape,
              NULL,
              NULL,
              dtype);
          if (relu_upper_limit > 0)
          {
            scalar_t upper_limit = {.f32 = relu_upper_limit};
            tpu_bdc_min_C(
                ping ? oaddr_ping : oaddr_pong,
                ping ? oaddr_ping : oaddr_pong,
                tpu_cast(upper_limit, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                &oslice_shape,
                NULL,
                NULL,
                dtype);
          }
        }
        ////////////pooling end//////////////////////

        // move output to global_mem
        if (n_c_idx > 0 || ohidx > 0 || owidx > 0)
        {
          dim4 last_oshape = {1, last_n_c_slice, last_ohslice, last_owslice};
          tpu_bdc_fp_mul_C(
            ping ? oaddr_pong : oaddr_ping,
            ping ? oaddr_pong : oaddr_ping,
            tpu_cast(scale_c, dtype, DT_FP32, RM_HALF_TO_EVEN),
            &last_oshape,
            NULL,
            NULL,
            dtype
          );
          tpu_gdma_cpy_L2S(
              ofmap_offset_global + ((n_c_idx * secs_info.n_c_slice) * ostride.c +
                                     last_ohstart * ostride.h + last_owstart) *
                                        tpu_data_type_size(dtype),
              ping ? oaddr_pong : oaddr_ping,
              &last_oshape,
              &ostride,
              NULL,
              dtype);
        }
        ping = !ping;
        // save current info used for moving output to global memory next loop
        // last_n_c_start = n_c_start;
        last_n_c_slice = n_c_slice;
        last_ohstart = ohstart;
        last_ohslice = ohslice;
        last_owstart = owstart;
        last_owslice = owslice;
      } // --3.OW
    }   // -- 2.OH
    tpu_parallel_end();

    // move the last output to global memory
    if (last_n_c_slice != 0)
    {
      dim4 last_oshape = {1, last_n_c_slice, last_ohslice, last_owslice};
      tpu_bdc_fp_mul_C(
            ping ? oaddr_pong : oaddr_ping,
            ping ? oaddr_pong : oaddr_ping,
            tpu_cast(scale_c, dtype, DT_FP32, RM_HALF_TO_EVEN),
            &last_oshape,
            NULL,
            NULL,
            dtype
          );
      tpu_gdma_cpy_L2S(
          ofmap_offset_global + ((n_c_idx * secs_info.n_c_slice) * ostride.c +
                                 last_ohstart * ostride.h + last_owstart) *
                                    tpu_data_type_size(dtype),
          ping ? oaddr_pong : oaddr_ping,
          &last_oshape,
          &ostride,
          NULL,
          dtype);
    }
  } // 1. -- N_C
}
extern int global_pooling_data_split(
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int kh,
    int kw,
    int dh,
    int dw,
    int pad_h,
    int pad_w,
    int stride_h,
    data_type_t dtype,
    pooling_secs_info_t *pooling_secs_info);

void nodechip_upsample_nearest2d_backward_parallel(
    const global_addr_t   ifmap_offset_global,
    const global_addr_t   ofmap_offset_global,
    const int             input_n,
    const int             input_c,
    const int             input_h,
    const int             input_w,
    const int             output_h,
    const int             output_w,
    const int             kh,
    const int             kw,
    const int             top_pad_h,
    const int             left_pad_w,
    int                   bottom_pad_h,
    int                   right_pad_w,
    int                   stride_h,
    int                   stride_w,
    const int             dilation_h,
    const int             dilation_w,
    const int             is_avg_pooling,
    const int             avg_pooling_mode,
    const int             if_relu,
    const float           relu_upper_limit,
    data_type_t           dtype,
    const int             c_step,
    const int             h_step,
    float                 Ratio,
    int                   scale
) {
    // local mem layout
    // 1 input feature map
    // 2 output feature map
    // 3 index tensor
//      int new_kh = (kh - 1) * dh + 1;
    TPUKERNEL_ASSERT(dilation_h == 1);
    TPUKERNEL_ASSERT(dilation_w == 1);
    if (input_h == kh && input_w == kw && top_pad_h == 0 && bottom_pad_h == 0 &&
        left_pad_w == 0 && right_pad_w == 0) {
        stride_h = 1;
        stride_w = 1;
    }
    int bottom_pad_h_output =
        MAX(0, (output_h - 1) * stride_h + kh - input_h - top_pad_h);
    const dim4 in_shape = {.n = input_n, .c = input_c,
                           .h = input_h, .w = input_w
                          };
    const dim4 out_shape = {.n = input_n, .c = input_c,
                            .h = output_h, .w = output_w
                           };
    const dim2 stride = {.h = stride_h, .w = stride_w};
    const padding_t pad = {.top = top_pad_h, .bottom = bottom_pad_h,
                           .left = left_pad_w, .right = right_pad_w
                          };
    const int type_size = tpu_data_type_size(dtype);
    const scalar_t scale_c = {.f32 = scale};
    dim4 bottom_global_stride ;
    tpu_continuous_stride(&bottom_global_stride, &in_shape);
    dim4 top_global_stride ;
    tpu_continuous_stride(&top_global_stride, &out_shape);

    // ceil(c/npu_num) * h * w local memory requirements
    float inRatio[1] = {0.0f}; // input's percentage
    inRatio[0] = Ratio;
    // need to update
    int ifmap_offset_local[2] = {0, 0};
    int ofmap_offset_local = 0;
    int offset_local_end = 0;

    int ofmap_align_size = 0;

    int local_mem_inputbanks = (int)(inRatio[0]*LOCAL_MEM_BANKS)/2; /* 2 is pipeline stage num and ratio is for internel mem div(such as input and output)*/
    if (local_mem_inputbanks == 0)
        local_mem_inputbanks = 1;
    ifmap_offset_local[0] = 0;
    ifmap_offset_local[1] =  LOCAL_MEM_SIZE / LOCAL_MEM_BANKS * local_mem_inputbanks;
    ofmap_offset_local = LOCAL_MEM_SIZE / LOCAL_MEM_BANKS * 2 * local_mem_inputbanks;

    // ------------- The parallel patten: two - stage software pipeline -----------------------
    // stage 0:  in0        in1       in0       in1       in0       ...... no_in
    // stage 1:  no_npu&out npu0&out0 npu1&out1 npu0&out0 npu1&out1 ...... npu0&out0  finish
    // ---------------------------------------------------------------------------------------
    // seperate data according to c dimension if a local_mem_bank can store a feature map
    // seperate data according to h dimension if a local_mem_bank can't store a feature map
    // ifmap (n, c, h, w) is regarded as (1, n * c, h, w)

    // In fact, c_step and h_step should be decided carefully to obtain the maximum performance
    // In the following, [0] indicate stage 0, [1] indicate stage 1
    int c_start[2];
    int h_start[2];
    int c_slice[2] = {0, 0};
    int h_slice[2] = {0, 0};
    int new_top_pad_h[2] = {0, 0};
    int new_bottom_pad_h[2] = {0, 0};
    int new_bottom_pad_h_output[2] = {0, 0};
    int input_h_start[2] = {0, 0};
    int output_h_start[2] = {0, 0};
    int new_output_h[2] = {0, 0};
    int loffset_idx[2] = {0, 0};
    int h_end;
    int loop_first = 1;
    int loop_last = 0;
    int cnt = 0;
    for (c_start[0] = 0, c_start[1] = 0,
            h_start[0] = 0 - pad.top, h_start[1] = 0 - pad.top;
            c_start[1] < in_shape.n * in_shape.c; ) {
        // update stage 0 parameters
        if (c_start[0] + c_step - 1 < in_shape.n * in_shape.c) {
            c_slice[0] = c_step;
        } else if (c_start[0] < in_shape.n * in_shape.c) {
            c_slice[0] = in_shape.n * in_shape.c - c_start[0];
        } else {
            c_slice[0] = 0;
        }
        input_h_start[0] = (h_start[0] < 0) ? 0 : h_start[0];
        output_h_start[0] = (h_start[0] + pad.top) / stride.h;
        new_top_pad_h[0] = 0 - h_start[0];
        if (new_top_pad_h[0] < 0) new_top_pad_h[0] = 0;
        // calculate h_slice, h_slice is the number of feature_map_height processed one time
        h_end = h_start[0] + h_step - stride.h + kh - 1;
        if (h_end < in_shape.h) {
            h_slice[0] = h_end - input_h_start[0] + 1;
            new_bottom_pad_h[0] = 0;
            new_bottom_pad_h_output[0] = new_bottom_pad_h[0];
        } else if (h_end >= in_shape.h && h_end < in_shape.h + pad.bottom) {
            h_slice[0] = in_shape.h - input_h_start[0];
            new_bottom_pad_h[0] = h_end - in_shape.h + 1;
            new_bottom_pad_h_output[0] = new_bottom_pad_h[0];
        } else {
            h_slice[0] = in_shape.h - input_h_start[0];
            new_bottom_pad_h[0] = pad.bottom;
            new_bottom_pad_h_output[0] = bottom_pad_h_output;
        }
        loffset_idx[0] = cnt % 2;
        new_output_h[0] =
            (h_slice[0] + new_top_pad_h[0] + new_bottom_pad_h_output[0] -kh) /
            stride.h + 1;

        ofmap_align_size =
            ((c_slice[0] + NPU_NUM - 1) / NPU_NUM) *
            tpu_aligned_feature_size(new_output_h[0], out_shape.w, dtype);
        offset_local_end = ofmap_offset_local + ofmap_align_size;
        if (offset_local_end > LOCAL_MEM_SIZE)
            TPUKERNEL_ASSERT(0);

        // Pipeline loop
        tpu_parallel_start();

        // ---------------------------- Pipeline stage 0 ----------------------------
        if (loop_last == 0) {
            // Load ifmap from global mem to local mem
            dim4 bottom_shape_tmp = {.n = 1, .c = c_slice[0],
                                     .h = h_slice[0], .w = in_shape.w
                                    };
            tpu_gdma_cpy_S2L(
                ifmap_offset_local[loffset_idx[0]],
                ifmap_offset_global +
                ((global_addr_t)c_start[0] * in_shape.h + input_h_start[0]) *
                in_shape.w * type_size,
                &bottom_shape_tmp,
                0, &bottom_global_stride,
                dtype);
        }

        // ---------------------------- Pipeline stage 1 ----------------------------
        if (loop_first == 0) {
            int c_remained = c_slice[1];
            int c_start = 0;
            while (c_remained > 0) {
                int __c = c_remained;
                int __n = 1;
                while (__c >= (1 << 16)) {
                    ++__n;
                    __c = c_remained / __n;
                }
                if (__n > 1)
                    __c = __c / NPU_NUM * NPU_NUM;
                TPUKERNEL_ASSERT(__c > 0);

                unsigned int ofmap_local_addr =
                    (ofmap_offset_local + c_start *
                     tpu_aligned_feature_size(new_output_h[1], output_w , dtype));
                unsigned int ifmap_local_addr =
                    (ifmap_offset_local[loffset_idx[1]] + c_start *
                     tpu_aligned_feature_size(h_slice[1], input_w , dtype));
                int in_dim[4] = {__n, __c, h_slice[1], in_shape.w};
                int out_dim[4] = {__n, __c, new_output_h[1], output_w};
                nodechip_pooling_local(
                    ifmap_local_addr,
                    ofmap_local_addr,
                    in_dim,
                    out_dim,
                    kh,
                    kw,
                    new_top_pad_h[1],
                    left_pad_w,
                    new_bottom_pad_h[1],
                    right_pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    is_avg_pooling,
                    avg_pooling_mode,
                    if_relu,
                    relu_upper_limit,
                    dtype);

                c_remained -= __n * __c;
                c_start += __n * __c / NPU_NUM;
            }
        }

        tpu_parallel_end();

        if (loop_first == 0) {
            dim4 top_shape_tmp = {.n = 1, .c = c_slice[1],
                                  .h = new_output_h[1], .w = output_w
                                 };
            tpu_bdc_fp_mul_C(
                ofmap_offset_local,
                ofmap_offset_local,
                tpu_cast(scale_c, dtype, DT_FP32, RM_HALF_TO_EVEN),
                &top_shape_tmp,
                NULL,
                NULL,
                dtype
            );
            tpu_gdma_cpy_L2S(
                ofmap_offset_global +
                ((unsigned long long)c_start[1] * output_h + output_h_start[1]) *
                output_w * type_size,
                ofmap_offset_local,
                &top_shape_tmp,
                &top_global_stride, 0,
                dtype);
        }

        // update stage 1 parameters
        c_start[1] = c_start[0];
        h_start[1] = h_start[0];
        c_slice[1] = c_slice[0];
        h_slice[1] = h_slice[0];
        new_top_pad_h[1] = new_top_pad_h[0];
        new_bottom_pad_h[1] = new_bottom_pad_h[0];
        input_h_start[1] = input_h_start[0];
        output_h_start[1] = output_h_start[0];
        loffset_idx[1] = loffset_idx[0];
        new_output_h[1] = new_output_h[0];

        // update c_start[0] and h_start[0]
        h_start[0] += h_step;
        if (h_start[0] > input_h + bottom_pad_h_output - kh) {
            c_start[0] += c_step;
            h_start[0] = 0 - pad.top;
        }

        // update loop parameters
        loop_first = 0;
        if (c_start[0] >= input_n * input_c) {
            loop_last = 1;
        }

        cnt++;
    }
}

void nodechip_upsample_nearest2d_backward_with_data_split(
    global_addr_t      ifmap_offset_global,
    global_addr_t      ofmap_offset_global,
    int             input_n,
    int             input_c,
    int             input_h,
    int             input_w,
    int             output_h,
    int             output_w,
    int             kh,
    int             kw,
    int             pad_h,
    int             pad_w,
    int             pad_h_after,
    int             pad_w_after,
    int             stride_h,
    int             stride_w,
    int             dilation_h,
    int             dilation_w,
    int             is_avg_pooling,
    int             avg_pooling_mode,
    int             if_relu,
    float           relu_upper_limit,
    int             scale,
    data_type_t     dtype
) {
        if (stride_h > 15 || stride_w > 15) {
            nodechip_upsample_nearest2d_backward_parallel_special_case(
                ifmap_offset_global,
                ofmap_offset_global,
                input_n,
                input_c,
                input_h,
                input_w,
                output_h,
                output_w,
                kh,
                kw,
                pad_h,
                pad_w,
                pad_h_after,
                pad_w_after,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                is_avg_pooling,
                avg_pooling_mode,
                if_relu,
                relu_upper_limit,
                scale,
                dtype);
        } else {
            pooling_secs_info_t pooling_secs_info;
            int result = global_pooling_data_split(
                            input_n, input_c, input_h, input_w,
                            output_h, output_w,
                            kh, kw,
                            dilation_h, dilation_w,
                            pad_h, pad_w,
                            stride_h,
                            dtype,
                            &pooling_secs_info);

            if (result == 0) {
                TPUKERNEL_LOG("Not supported pooling parameters.\n");
                TPUKERNEL_ASSERT(0);
            }
            nodechip_upsample_nearest2d_backward_parallel(
                ifmap_offset_global,
                ofmap_offset_global,
                input_n, input_c, input_h, input_w,
                output_h, output_w,
                kh, kw,
                pad_h, pad_w, pad_h_after, pad_w_after,
                stride_h, stride_w,
                dilation_h, dilation_w,
                is_avg_pooling,
                avg_pooling_mode,
                if_relu, relu_upper_limit,
                dtype,
                pooling_secs_info.nsecs,
                pooling_secs_info.hsecs,
                pooling_secs_info.Ratio,
                scale
            );
        }
}

int tpu_kernel_api_upsample_nearest2d_backward(const void *args) {
  sg_api_upsample2d_backward_t *api = (sg_api_upsample2d_backward_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_FP16 ||
                   api->dtype == DT_BFP16);

  nodechip_upsample_nearest2d_backward_with_data_split(
        api->input_global_addr,
        api->output_global_addr,
        api->input_shape[0],
        api->input_shape[1],
        api->input_shape[2],
        api->input_shape[3],
        api->output_shape[2],
        api->output_shape[3],
        api->pooling_desc.kh,
        api->pooling_desc.kw,
        api->pooling_desc.pad_h,
        api->pooling_desc.pad_w,
        api->pooling_desc.pad_h,
        api->pooling_desc.pad_w,
        api->pooling_desc.stride_h,
        api->pooling_desc.stride_w,
        1/*dilation_h*/, 1/*dilation_w*/,
        api->pooling_desc.mode == POOLING_AVG,
        0/*avg_pooling_mode*/, 0/*if_relu*/,
        -1/*relu_upper_limit*/,
        api->scalar, api->dtype);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_upsample_nearest2d_backward);
