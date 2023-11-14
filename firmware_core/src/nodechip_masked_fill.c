#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include "config.h"

extern void nodechip_masked_fill(
    global_addr_t  input_global_addr,
    global_addr_t  mask_global_addr,
    global_addr_t  output_global_addr,
    const int*     input_shape,
    const int*     mask_shape,
    int            input_dims,
    int            mask_dims,
    float          value,
    data_type_t    dtype);


void tpu_kernel_api_masked_fill ( const void * args )
{
    sg_api_masked_fill_t *api = ( sg_api_masked_fill_t * ) args;
    tpu_initialize();
    nodechip_masked_fill(
        api->input_global_addr,
        api->mask_global_addr,
        api->out_global_addr,
        api->input_shape,
        api->mask_shape,
        api->input_dims,
        api->mask_dims,
        api->value,
        api->dtype
    );
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_masked_fill);

#ifdef FIRMWARE_BACKEND_2260
void tpu_kernel_api_masked_fill_multi_core ( const void * args )
{
    sg_api_masked_fill_t *api = ( sg_api_masked_fill_t * ) args;

    int is_bcast[FW_MAX_SHAPE_DIMS] = {0};
    for (int i = 0; i < api->mask_dims; ++i)
        is_bcast[api->input_dims-1-i] = (api->mask_shape[api->mask_dims-1-i] == 1) && api->mask_shape[api->mask_dims-1-i] != api->input_shape[api->input_dims-1-i];
    for (int i = api->mask_dims; i < api->input_dims; ++i)
        is_bcast[api->input_dims-1-i] = 1;

    int num_edges = 0;
    int new_input_shape[3] = {1, 1, 1};
    int new_mask_shape[3] = {1, 1, 1};

    // merge shapes to the right. attempt to increase utilization
    for (int i = 0; i < api->input_dims - 1; ++i)
    {    
        TPUKERNEL_ASSERT(num_edges <= 2);
        new_input_shape[2 - num_edges] *= api->input_shape[api->input_dims-1-i];
        new_mask_shape[2 - num_edges] *= api->mask_shape[api->mask_dims-1-i];
        if (is_bcast[api->input_dims-1-i] != is_bcast[api->input_dims-2-i])
            ++num_edges;
    }
    new_input_shape[2 - num_edges] *= api->input_shape[0];
    new_mask_shape[2 - num_edges] *= api->mask_shape[0];

    // broadcast cases. Same as backend. mode is the dim to split.
    int mode = -1;
    if (num_edges == 0 && !is_bcast[api->input_dims-1]) { // no broadcast
        mode = 2;
    } else if ((num_edges == 0 && is_bcast[api->input_dims-1])
            || (num_edges == 1 && !is_bcast[api->input_dims-1])) {
        mode = 1;
    } else if (num_edges == 1 && is_bcast[api->input_dims-1]) {
        mode = 1;
    } else if (num_edges == 2 && !is_bcast[api->input_dims-1] && !is_bcast[0]) {
        mode = 0;
    } else {
        TPUKERNEL_ASSERT_INFO(false, "not implemented");
    }

    tpu_initialize();
    int core_num = tpu_core_num();
    int core_idx = tpu_core_index();
    int length = new_input_shape[mode];
    int input_length_other = new_input_shape[0] * new_input_shape[1] * new_input_shape[2] / length;
    int mask_length_other = new_mask_shape[0] * new_mask_shape[1] * new_mask_shape[2] / length;

    int length_slice = DIV_UP(length, core_num);
    int length_secs = DIV_UP(length, length_slice);
    TPUKERNEL_ASSERT(length_secs <= core_num);
    int cur_length_slice = length_slice;
    if (core_idx == length_secs - 1)
        cur_length_slice = length - length_slice * (length_secs - 1);
    if (core_idx * length_slice < length) {
        int cur_input_shape[3] = {new_input_shape[0], new_input_shape[1], new_input_shape[2]};
        int cur_mask_shape[3] = {new_mask_shape[0], new_mask_shape[1], new_mask_shape[2]};
        cur_input_shape[mode] = cur_length_slice;
        cur_mask_shape[mode] = cur_length_slice;
        if (new_input_shape[mode] != new_mask_shape[mode]) // the split dim need broadcast
        {
            TPUKERNEL_ASSERT(new_mask_shape[mode] == 1);
            cur_mask_shape[mode] = 1;
            mask_length_other = 0;
        }
        
        nodechip_masked_fill(
            api->input_global_addr + (length_slice * input_length_other * core_idx) * tpu_data_type_size(api->dtype),
            api->mask_global_addr + (length_slice * mask_length_other * core_idx) * tpu_data_type_size(api->dtype),
            api->out_global_addr + (length_slice * input_length_other * core_idx) * tpu_data_type_size(api->dtype),
            cur_input_shape,
            cur_mask_shape,
            3,
            3,
            api->value,
            api->dtype
        );
    }
    tpu_poll();
}
TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_masked_fill_multi_core);

/* simple version, only split the first dim */
// void tpu_kernel_api_masked_fill_multi_core ( const void * args )
// {
//     sg_api_masked_fill_t *api = ( sg_api_masked_fill_t * ) args;

//     tpu_initialize();
//     int core_num = tpu_core_num();
//     int core_idx = tpu_core_index();
//     int input_shape[FW_MAX_SHAPE_DIMS] = {0};
//     int mask_shape[FW_MAX_SHAPE_DIMS] = {0};
//     int input_length_other = 1;
//     int mask_length_other = 1;
//     for(int i = 0; i < api->input_dims; ++i)
//     {
//         input_shape[i] = api->input_shape[i];
//         if (i != 0) input_length_other *= api->input_shape[i];
//     }
//     for(int i = 0; i < api->mask_dims; ++i)
//     {
//         mask_shape[i] = api->mask_shape[i];
//         if (i != 0) mask_length_other *= api->mask_shape[i];
//     }    

//     int length = input_shape[0];
//     int length_slice = DIV_UP(length, core_num);
//     int length_secs = DIV_UP(length, length_slice);
//     TPUKERNEL_ASSERT(length_secs <= core_num);
//     int cur_length_slice = length_slice;
//     if (core_idx == length_secs - 1)
//         cur_length_slice = length - length_slice * (length_secs - 1);
//     if (core_idx * length_slice < length) {
//         input_shape[0] = cur_length_slice;
//         mask_shape[0] = cur_length_slice;
//         if (api->input_shape[0] != api->mask_shape[0])
//         {
//             TPUKERNEL_ASSERT(api->mask_shape[0] == 1);
//             mask_shape[0] = 1;
//             mask_length_other = 0;
//         }
        
//         nodechip_masked_fill(
//             api->input_global_addr + (length_slice * input_length_other * core_idx) * tpu_data_type_size(api->dtype),
//             api->mask_global_addr + (length_slice * mask_length_other * core_idx) * tpu_data_type_size(api->dtype),
//             api->out_global_addr + (length_slice * input_length_other * core_idx) * tpu_data_type_size(api->dtype),
//             input_shape,
//             mask_shape,
//             api->input_dims,
//             api->mask_dims,
//             api->value,
//             api->dtype
//         );
//     }
//     tpu_poll();
// }
// TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_masked_fill_multi_core);
#endif
