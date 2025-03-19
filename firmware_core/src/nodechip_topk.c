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

void partial_sort(
    const char *input,
    char *value_output,
    int32_t *index_output,
    int k,
    uint64_t axis_size,
    uint64_t axis_stride,
    int dt_size)
{
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

#ifdef BACKEND_SG2260
int scalar_topk_multi_core(sg_api_topk_t *api)
{
    int axis = api->axis > 0 ? api->axis : api->axis + api->dim;

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

    const char *input = (char *)tpu_global_mem_addr(api->input_global_addr);
    char *value_output = (char *)tpu_global_mem_addr(api->value_global_addr);
    char *index_output = (char *)tpu_global_mem_addr(api->index_global_addr);

    int dt_size = tpu_data_type_size(api->dtype);

    input += tile_index * tiling_size * outer_stride * dt_size;
    value_output += tile_index * tiling_size * output_stride * dt_size;
    index_output += tile_index * tiling_size * output_stride * sizeof(int32_t);

    unsigned real_tile_size = MIN(tiling_size, outer_num - tiling_size * tile_index);
    for (int i = 0; i < (int)real_tile_size; ++i)
    {
        for (int j = 0; j < (int)inner_num; ++j)
        {
            partial_sort(
                input + (i * outer_stride + j) * dt_size,
                value_output + (i * output_stride + j) * dt_size,
                ((int32_t *)index_output) + i * output_stride + j,
                api->k,
                api->shape[axis],
                axis_stride,
                dt_size);
        }
    }

    return 0;
}

int tpu_kernel_api_topk_multi_core(const void *args) {
  sg_api_topk_t *api = (sg_api_topk_t *)args;

  int64_t length = 1;
  for (int i = 0; i < api->dim; ++i) {
    length *= api->shape[i];
  }

  if (length < 1024 * 8)
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
