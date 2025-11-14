#include <stdlib.h>


#include "sg_api_struct.h"
#include "tpu_kernel.h"

// void nodechip_topk(global_addr_t input_global_addr,
//                    global_addr_t value_global_addr,
//                    global_addr_t index_global_addr, int *shape, int dim, int
//                    k, int axis, bool largest, bool sorted, int64_t length,
//                    data_type_t dtype) {
//   if (length == 0)
//     return;

//   const int batch_stride = shape[axis];
//   const int batch = length / batch_stride;
//   global_addr_t src_value_addr = input_global_addr;
//   global_addr_t dst_value_addr = value_global_addr;
//   global_addr_t dst_index_addr = index_global_addr;

//   for (int i = 0; i < batch; ++i) {
//     src_value_addr = input_global_addr + i * batch_stride * sizeof(int);
//     dst_value_addr = value_global_addr + i * k * sizeof(int);
//     dst_index_addr = index_global_addr + i * k * sizeof(int);

//     tpu_hau_sort_natural_index(dst_value_addr, dst_index_addr,
//     src_value_addr,
//                                batch_stride, k, largest, dtype);
//     tpu_hau_poll();
//   }
// }

extern void nodechip_batch_topk(system_addr_t bottom_value_addr,
                                system_addr_t bottom_index_addr, // is always int32
                                system_addr_t top_value_addr,
                                system_addr_t top_index_addr, // is always int32
                                system_addr_t buffer_value_addr,
                                system_addr_t buffer_index_addr,
                                bool bottom_index_valid, int k, int descending, int batchs,
                                bool is_batch_same, int *batch_num, int batch_stride,
                                data_type_t dtype);

int tpu_kernel_api_topk(const void *args) {
  sg_api_topk_t *api = (sg_api_topk_t *)args;
  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_INT32 ||
                   api->dtype == DT_UINT32);

  int64_t length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }
  if (api->axis != api->dim - 1) {
    TPUKERNEL_ASSERT_INFO(false, "not support axis != dim-1 now");
  }

  const int batch_stride = api->shape[api->axis];
  const int batchs = length / batch_stride;
  int *batch_nums = (int *)malloc(batchs * sizeof(int));
  for (int i = 0; i < batchs; i++)
    batch_nums[i] = batch_stride;
  tpu_initialize();
  nodechip_batch_topk(api->input_global_addr, (system_addr_t)(-1),
                      api->value_global_addr, api->index_global_addr, 0, 0, false,
                      api->k, api->largest, batchs, true, batch_nums,
                      batch_stride, api->dtype);
  tpu_poll();
  free(batch_nums);
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_topk);

static void swap_2byte(void *a_ptr, void *b_ptr, int size)
{
    int16_t *a = (int16_t *)a_ptr;
    int16_t *b = (int16_t *)b_ptr;

    int16_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void swap_4byte(void *a_ptr, void *b_ptr, int size)
{
    int32_t *a = (int32_t *)a_ptr;
    int32_t *b = (int32_t *)b_ptr;

    int32_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void swap_8byte(void *a_ptr, void *b_ptr, int size)
{
    int64_t *a = (int64_t *)a_ptr;
    int64_t *b = (int64_t *)b_ptr;

    int64_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static int greater_2byte(const void *a, const void *b, int size)
{
    return *(int16_t *)a > *(int16_t *)b;
}

static void swap(void *a_ptr, void *b_ptr, int size)
{
    char *a = (char *)a_ptr;
    char *b = (char *)b_ptr;

    char tmp;
    for (int i = 0; i < size; ++i)
    {
        tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

static int greater(const void *a, const void *b, int size)
{
    if (size == 2)
        return *(const int16_t *)a > *(const int16_t *)b;
    else if (size == 4)
        return *(const int32_t *)a > *(const int32_t *)b;

    return 0;
}

#include <string.h>

void partial_sort_cpu(
    uint64_t input_addr,
    uint64_t value_output_addr,
    uint64_t index_output_addr,
    int k,
    uint64_t axis_size,
    uint64_t axis_stride,
    data_type_t dtype)
{
    const char *input = (char *)tpu_global_mem_addr(input_addr);
    char *value_output = (char *)tpu_global_mem_addr(value_output_addr);
    int32_t *index_output = (int32_t *)tpu_global_mem_addr(index_output_addr);
    int dt_size = tpu_data_type_size(dtype);

    // Only supports signed comparision, i.e., INT, FP32, BF32

    char *input_buffer = (char *)malloc(axis_size * dt_size);
    int32_t *index_buffer = (int32_t *)malloc(axis_size * sizeof(int32_t));

    if (axis_stride == 1)
    {
        memcpy(input_buffer, input, axis_size * dt_size);
        for (int i = 0; i < (int)axis_size; ++i)
            index_buffer[i] = i;
    } else {
        for (int i = 0; i < (int)axis_size; ++i)
        {
            memcpy(
                input_buffer + i * dt_size,
                input + i * axis_stride * dt_size,
                dt_size);
            index_buffer[i] = i;
        }
    }

    void (*swap_fn)(void *, void *, int);
    int (*greater_fn)(const void *, const void *, int);
    if (dt_size == 2)
    {
        swap_fn = swap_2byte;
        greater_fn = greater_2byte;
    } else {
        swap_fn = swap;
        greater_fn = greater;
    }

    char *a, *b;
    for (int i = 0; i < k; ++i)
    {
        for (int j = axis_size - 1; j > i; --j)
        {
            a = input_buffer + j * dt_size;
            b = a - dt_size;
            if (greater_fn(b, a, dt_size))
                continue;
            swap_fn(a, b, dt_size);
            swap_4byte(index_buffer + j, index_buffer + j - 1, sizeof(int32_t));
        }
    }

    if (axis_stride == 1)
    {
        memcpy(value_output, input_buffer, k * dt_size);
        memcpy(index_output, index_buffer, k * sizeof(int32_t));
    } else {
        for (int i = 0; i < k; ++i)
        {
            memcpy(
                value_output + i * axis_stride * dt_size,
                input_buffer + i * dt_size,
                dt_size);
            index_output[i * axis_stride] = index_buffer[i];
        }
    }
}

void tpu_topk(
    global_addr_t input_global,
    int outer_size, int axis_size, int inner_size,
    int k,
    global_addr_t value_output,
    global_addr_t index_output,
    data_type_t dtype)
{
    TPUKERNEL_ASSERT_INFO(inner_size == 1, "tpu Top-K only support inner size 1, got %d", inner_size);

    int input_size_in_lane = DIV_UP(outer_size, NPU_NUM) * axis_size * inner_size * sizeof(float);
    int mask_size_in_lane = DIV_UP(outer_size, NPU_NUM) * axis_size * axis_size;
    TPUKERNEL_ASSERT_INFO(
        mask_size_in_lane < (LOCAL_BANK_SIZE * (LOCAL_MEM_BANKS - 4)),
        "Too large for tpu Top-K, outer_size=%d, axis_size=%d", outer_size, axis_size);  // TODO tiling
    TPUKERNEL_ASSERT_INFO(
        input_size_in_lane < LOCAL_BANK_SIZE,
        "Too large for tpu Top-K, outer_size=%d, axis_size=%d", outer_size, axis_size); // TODO tiling

    local_addr_t input_local = 0;
    local_addr_t seq_local = LOCAL_BANK_SIZE;
    local_addr_t buf0 = seq_local + LOCAL_BANK_SIZE;
    local_addr_t buf1 = buf0 + LOCAL_BANK_SIZE;
    local_addr_t buf2 = buf1 + LOCAL_BANK_SIZE;

    tpu_parallel_start();

    dim4 input_shape = {.n = 1, .c = outer_size, .h = 1, .w = axis_size};
    tpu_gdma_cpy_S2L(input_local, input_global, &input_shape, NULL, NULL, dtype);

    tpu_bdc_arithmetic_sequence_bcast(seq_local, outer_size, 0, 1, axis_size);
    dim4 seq_shape = {.n = 1, .c = NPU_NUM, .h = 1, .w = axis_size};
    tpu_bdc_cast(buf1, seq_local, &seq_shape, NULL, NULL, DT_FP32, DT_UINT32, RM_HALF_TO_EVEN);
    scalar_t C = {.f32 = 1e-6};
    tpu_bdc_fp_mul_C(buf0, buf1, C, &seq_shape, NULL, NULL, DT_FP32);
    local_addr_t bias_local = buf0;

    tpu_parallel_end();

    local_addr_t f32_input = buf2;
    tpu_bdc_cast(f32_input, input_local, &input_shape, NULL, NULL, DT_FP32, dtype, RM_HALF_TO_EVEN);

    dim4 seq_stride;
    tpu_aligned_stride(&seq_stride, 0, &input_shape, dtype);
    seq_stride.c = 0;
    local_addr_t biased_input_local = buf1;
    tpu_bdc_fp_add(biased_input_local, f32_input, bias_local, &input_shape, NULL, NULL, &seq_stride, DT_FP32);

    dim4 bw_stride, bh_stride;
    tpu_aligned_stride(&bw_stride, 0, &input_shape, dtype);
    bh_stride = bw_stride;
    bw_stride.h = 1;
    bw_stride.w = 0;
    bh_stride.w = 1;
    bh_stride.h = 0;
    scalar_t true_val = {.u8 = 1};
    dim4 mask_shape = {.n = 1, .c = outer_size, .h = axis_size, .w = axis_size};
    local_addr_t mask_local = buf2;
    tpu_bdc_less(
        mask_local,
        biased_input_local, biased_input_local,
        true_val,
        &mask_shape, NULL,
        &bw_stride, &bh_stride,
        DT_UINT8, DT_FP32);

    local_addr_t scatter_index = buf0;
    dim2 kernel = {.h = 1, .w = axis_size};
    padding_t padding = {0};
    dim2 stride = {.h = 1, .w = 1};
    dim2 dilation = {.h = 1, .w = 1};
    const int pool_eu_num = 32; // Use 32 outta 64
    if (axis_size % pool_eu_num == 0 && kernel.w / pool_eu_num < 15)
    {
        kernel.w /= pool_eu_num;
        stride.w = kernel.w;
        tpu_bdc_int8_avg_pool2d(
            mask_local,
            mask_local,
            &mask_shape,
            &kernel,
            &padding,
            &stride,
            &dilation,
            DT_UINT8,
            DT_UINT8,
            1, 0);
        mask_shape.w = pool_eu_num;
        kernel.w = pool_eu_num;
        tpu_bdc_int8_avg_pool2d(
            scatter_index,
            mask_local,
            &mask_shape,
            &kernel,
            &padding,
            &stride,
            &dilation,
            DT_UINT16,
            DT_UINT8,
            1, 0);
    } else {
        tpu_bdc_int8_avg_pool2d(
            scatter_index,
            mask_local,
            &mask_shape,
            &kernel,
            &padding,
            &stride,
            &dilation,
            DT_UINT16,
            DT_UINT8,
            1, 0);
    }

    local_addr_t output_local = buf1;
    tpu_bdc_batch_bcast_w_scatter(
        output_local,
        input_local,
        scatter_index,
        &input_shape,
        axis_size,
        dtype,
        DT_UINT16,
        false);

    tpu_parallel_start();

    dim4 output_shape = {.n = 1, .c = outer_size, .h = 1, .w = k};
    dim4 output_local_stride;
    tpu_aligned_stride(&output_local_stride, 0, &input_shape, dtype);
    tpu_gdma_cpy_L2S(
        value_output,
        output_local,
        &output_shape,
        NULL,
        &output_local_stride,
        dtype);

    local_addr_t index_local = mask_local;
    tpu_bdc_batch_bcast_w_scatter(
        index_local,
        seq_local,
        scatter_index,
        &input_shape,
        axis_size,
        DT_UINT32,
        DT_UINT16,
        false);

    tpu_parallel_end();

    tpu_aligned_stride(&output_local_stride, 0, &input_shape, DT_UINT32);
    tpu_gdma_cpy_L2S(
        index_output,
        index_local,
        &output_shape,
        NULL,
        &output_local_stride,
        DT_UINT32);
}

#ifdef ENABLE_MULTI_CORE
int scalar_topk_multi_core(sg_api_topk_t *api)
{
    int axis = api->axis >= 0 ? api->axis : api->axis + api->dim;

    unsigned outer_num = 1, inner_num = 1;
    unsigned outer_stride = 1, axis_stride = 1, output_stride = 1;
    for (int i = 0; i < api->dim; ++i)
    {
        if (i < axis) {
            outer_num *= api->shape[i];
        } else {
            outer_stride *= api->shape[i];
            output_stride *= i == axis ? api->k : api->shape[i];
        }

        if (i > axis) {
            inner_num *= api->shape[i];
            axis_stride *= api->shape[i];
        }
    }

    int tile_num = tpu_group_num() * tpu_workitem_num();
    int tile_index = tpu_group_index() * tpu_workitem_num() + tpu_workitem_index();
    unsigned tiling_size = DIV_UP(outer_num, tile_num);

    if (tile_index * tiling_size > outer_num)
        return 0;

    uint64_t input = api->input_global_addr;
    uint64_t value_output = api->value_global_addr;
    uint64_t index_output = api->index_global_addr;

    int dt_size = tpu_data_type_size(api->dtype);

    input += tile_index * tiling_size * outer_stride * dt_size;
    value_output += tile_index * tiling_size * output_stride * dt_size;
    index_output += tile_index * tiling_size * output_stride * sizeof(int32_t);

    unsigned real_tile_size = MIN(tiling_size, outer_num - tiling_size * tile_index);
    if (real_tile_size <= 0)
        return 0;
#if 0
    for (int i = 0; i < (int)real_tile_size; ++i)
    {
        for (int j = 0; j < (int)inner_num; ++j)
        {
            partial_sort_cpu(
                input + (i * outer_stride + j) * dt_size,
                value_output + (i * output_stride + j) * dt_size,
                index_output + (i * output_stride + j) * sizeof(int32_t),
                api->k,
                api->shape[axis],
                axis_stride,
                api->dtype);
        }
    }
#else
    // TODO
    // Adjust local tiling size by inner shape & pipeline
    const unsigned local_tile = 128;
    int local_tile_num = DIV_UP(real_tile_size, local_tile);
    for (int i = 0; i < local_tile_num; ++i)
    {
        int real_local_tile = MIN(local_tile, real_tile_size - local_tile * i);
        tpu_topk(
            input + i * local_tile * outer_stride * dt_size,
            real_local_tile,
            api->shape[axis],
            inner_num,
            api->k,
            value_output + i * local_tile * output_stride * dt_size,
            index_output + i * local_tile *output_stride * sizeof(int32_t),
            api->dtype);
    }
#endif

    return 0;
}

int tpu_kernel_api_topk_multi_core(const void *args) {
  sg_api_topk_t *api = (sg_api_topk_t *)args;

  int64_t length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  if (api->shape[api->axis] <= 256)
  {
      tpu_initialize();
      int ret = scalar_topk_multi_core(api);
      tpu_sync_all();
      tpu_poll();
      return ret;
  }

  TPUKERNEL_ASSERT(api->dtype == DT_FP32 || api->dtype == DT_INT32 ||
                   api->dtype == DT_UINT32);

  if (api->axis != api->dim - 1) {
    TPUKERNEL_ASSERT_INFO(false, "not support axis != dim-1 now");
  }
  const int batch_stride = api->shape[api->axis];
  const int batchs = length / batch_stride;
  int *batch_nums = (int *)malloc(batchs * sizeof(int));
  for (int i = 0; i < batchs; i++)
    batch_nums[i] = batch_stride;
  tpu_initialize();
  int core_idx = tpu_core_index();
  if (core_idx == 0) {
    nodechip_batch_topk(api->input_global_addr, (system_addr_t)(-1),
                        api->value_global_addr, api->index_global_addr, 0,0, false,
                        api->k, api->largest, batchs, true, batch_nums,
                        batch_stride, api->dtype);
  }
  tpu_poll();
  free(batch_nums);
  return 0;
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_topk_multi_core);
#endif
