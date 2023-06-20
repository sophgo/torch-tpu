#include "sg_api_struct.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"
#include <string.h>
#include <stdlib.h>

typedef struct List_Node{
    int         index;
    struct List_Node    *prv;
    struct List_Node    *next;
} ListNode;
typedef struct _List{
    ListNode    head;
    ListNode    tail;
} List;

bool idx_in_window(int idx,  int* window, int pos ){
    if (pos == 0) return false;
    if (window[pos-1] == idx) return true;
    return false;
}

ListNode* delete_nodel(ListNode* pnode){
    pnode->next->prv = pnode->prv;
    pnode->prv->next = pnode->next;
    ListNode *to_deleted = pnode; 
    pnode = pnode->next;
    free(to_deleted);
    return pnode;
}


void nodechip_embedding_backward(
    global_addr_t gradout_global_addr,
    global_addr_t index_global_addr,
    global_addr_t out_global_addr,
    global_addr_t sorted_index_global_addr,
    global_addr_t sorted_index_index_global_addr,
    global_addr_t from_index_global_addr,
    global_addr_t to_index_global_addr,
    global_addr_t from_buffer_global_addr,
    global_addr_t to_buffer_global_addr, 
    int           window_size,
    const int     *gradout_shape,
    const int     *index_shape,
    const int     *out_shape,
    int           gradout_dim,
    int           idx_dim,
    int           out_dim,
    data_type_t   grad_dtype,
    data_type_t   index_dtype){
    TPUKERNEL_ASSERT(index_dtype == DT_INT32);
    // 1. fill out_global with zero
    scalar_t zero_ = {.u32 = 0};
    dim4 o_shape = {.n = 1, .c = 1, .h = 1, .w = out_shape[out_dim -1]};
    for (int i = 0; i < out_dim - 1; i++){
        o_shape.h *= out_shape[i];
    }
    dim4 go_shape = {.n = 1, .c = 1, .h = 1, .w = gradout_shape[out_dim -1]};
    for (int i =0; i < gradout_dim -1; i++){
        go_shape.h *= gradout_shape[i];
    }

    tpu_gdma_set_C_system(
        out_global_addr,
        zero_,
        &o_shape,
        NULL,
        grad_dtype);
    
    int NUM_index = 1;
    //printf("index_shape: ");
    for (int i = 0; i < idx_dim; i++){
        NUM_index *= index_shape[i];
        //printf("  %d,", index_shape[i]);
    }
    //printf("\n");    
    // 2. sort index
    tpu_hau_sort_natural_index(sorted_index_global_addr, 
                               sorted_index_index_global_addr,
                               index_global_addr,
                               NUM_index,NUM_index, false, DT_INT32);
    tpu_hau_poll();
#if 1
    tpu_invalidate_cache(sorted_index_global_addr, ALIGN( NUM_index * sizeof(int), 64));
    tpu_invalidate_cache(sorted_index_index_global_addr, ALIGN(NUM_index * sizeof(int), 64));

    int window_cur = 0;
    int *in_idx_window =  (int*)malloc( window_size * sizeof(int) );
    int *out_idx_window = (int*)malloc( window_size * sizeof(int) );
    List reserved_list;
    reserved_list.head.next = &reserved_list.tail; reserved_list.tail.prv = &reserved_list.head;
    ListNode* ptr_reserved_idx = reserved_list.head.next;
    int start_idx = 0;
    //printf("===NUM_idx : %d \n", NUM_index);
    while ( start_idx < NUM_index )
    {
        //printf("start_idx: %d \n", start_idx);
        if (ptr_reserved_idx == &reserved_list.tail){
            int idx_out = ((int*)tpu_global_mem_addr(sorted_index_global_addr))[start_idx];
            int idx_in  = ((int*)tpu_global_mem_addr(sorted_index_index_global_addr))[start_idx];
            if ( idx_in_window( idx_out, out_idx_window, window_cur) ){
                ListNode *node = (ListNode*) malloc(sizeof(ListNode));
                node->prv = reserved_list.tail.prv;
                node->next = &reserved_list.tail;
                node->index = start_idx;
                reserved_list.tail.prv = node;
                do{
                    start_idx++;
                    //printf("----start_idx: %d \n", start_idx);
                }while (start_idx < NUM_index && 
                        ((int*)tpu_global_mem_addr(sorted_index_global_addr))[start_idx] == idx_out);
            }else{
                in_idx_window[window_cur] = idx_in;
                out_idx_window[window_cur] = idx_out;
                window_cur++;
                start_idx++;
            }
        }else{
            int idx_out = ((int*)tpu_global_mem_addr(sorted_index_global_addr))[ptr_reserved_idx->index];
            int idx_in  = ((int*)tpu_global_mem_addr(sorted_index_index_global_addr))[ptr_reserved_idx->index];
            if (idx_in_window(idx_out, out_idx_window, window_cur)){
                ptr_reserved_idx = ptr_reserved_idx->next;
            }else{ // find useful idx in list
                in_idx_window[window_cur] = idx_in;
                out_idx_window[window_cur] = idx_out;
                window_cur++;
                // delete node in list
                if (ptr_reserved_idx->index + 1 < NUM_index){
                    int next_idx_out = ((int*)tpu_global_mem_addr(sorted_index_global_addr))[ptr_reserved_idx->index + 1];
                    if (next_idx_out == idx_out) // have another value in list
                    {
                        ptr_reserved_idx->index++;
                        ptr_reserved_idx = ptr_reserved_idx->next;
                    }
                    else // delete node
                    {
                        ptr_reserved_idx = delete_nodel(ptr_reserved_idx);
                    }
                } else{
                    ptr_reserved_idx = delete_nodel(ptr_reserved_idx);
                }
            }
        }
        if (window_cur == window_size || start_idx == NUM_index) { // enough to caculate
            ptr_reserved_idx = reserved_list.head.next;
            //do caculate here
            memcpy(tpu_global_mem_addr(from_index_global_addr), in_idx_window, sizeof(int) * window_cur);
            memcpy(tpu_global_mem_addr(to_index_global_addr), out_idx_window, sizeof(int) * window_cur);
            tpu_flush_cache(from_index_global_addr, ALIGN(sizeof(int) * window_cur, 64));
            tpu_flush_cache(to_index_global_addr, ALIGN(sizeof(int) * window_cur, 64));

            dim4 add_shape = {.n = 1, .c = window_cur, .h = 1, .w=o_shape.w};
            local_addr_t from_local_addr = 0;
            local_addr_t to_local_addr = from_local_addr + add_shape.n * DIV_UP(add_shape.c , NPU_NUM) * tpu_aligned_feature_size(add_shape.h, add_shape.w, grad_dtype);
            local_addr_t res_local_addr = to_local_addr + add_shape.n * DIV_UP(add_shape.c , NPU_NUM) * tpu_aligned_feature_size(add_shape.h, add_shape.w, grad_dtype);
            TPUKERNEL_ASSERT(( 3 * add_shape.n * DIV_UP(add_shape.c , NPU_NUM) * tpu_aligned_feature_size(add_shape.h, add_shape.w, grad_dtype))
                              < LOCAL_MEM_SIZE);
            dim4 from_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
            tpu_gdma_h_gather_S2L(from_local_addr, gradout_global_addr, from_index_global_addr, false, zero_, &add_shape, go_shape.h, NULL, &from_stride, NULL, grad_dtype);
            dim4 to_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
            tpu_gdma_h_gather_S2L(to_local_addr, out_global_addr, to_index_global_addr, false, zero_, &add_shape, o_shape.h, NULL, &to_stride, NULL, grad_dtype);
            
            tpu_poll();
            tpu_bdc_fp_add(res_local_addr, from_local_addr, to_local_addr, &add_shape, NULL, NULL, NULL, grad_dtype);
            tpu_poll();

            // tpu_gdma_general_cpy_L2S(to_buffer_global_addr,res_local_addr, &add_shape, &add_shape, NULL, NULL, grad_dtype);
            // tpu_gdma_h_scatter_S2S(out_global_addr, to_buffer_global_addr, to_index_global_addr, false, &o_shape, window_cur, NULL, NULL, NULL, grad_dtype);
            //dim4 scatter_out_stride = {.n = 0 , .c = add_shape.w, .h = 0, .w = 1};
            dim4 scatter_out_stride = {.n = 0 , .c = 0, .h = add_shape.w, .w = 1};
            dim4 scatter_oshape = {.n = o_shape.n, .c = window_cur, .h = o_shape.h, .w = o_shape.w};
            tpu_gdma_h_scatter_L2S(out_global_addr, res_local_addr, to_index_global_addr, false, &scatter_oshape, 1, &scatter_out_stride, NULL, NULL, grad_dtype);
            tpu_poll();
            memset(in_idx_window, 0, sizeof(int) * window_size);
            memset(out_idx_window, 0, sizeof(int) * window_size);
            window_cur = 0;
        }
    }

    // list have reserved data
    ptr_reserved_idx = reserved_list.head.next;
    while(ptr_reserved_idx != &reserved_list.tail){
        // do calculate
        int idx_out = ((int*)tpu_global_mem_addr(sorted_index_global_addr))[ptr_reserved_idx->index];
        int len = 0;
        for (int tmp = ptr_reserved_idx->index;tmp < NUM_index; tmp++){
            in_idx_window[len] = ((int*)tpu_global_mem_addr(sorted_index_index_global_addr))[tmp];
            len++;
            if(((int*)tpu_global_mem_addr(sorted_index_global_addr))[tmp] != idx_out){
                break;
            }
        }
        // local is enough
        int start_idx = 0;
        while (start_idx < len){
            int cur_len = MIN(len, window_size);
            local_addr_t res_local_addr = 0;
            local_addr_t in_local_addr = res_local_addr + tpu_aligned_feature_size(1, out_shape[out_dim-1], grad_dtype);
            TPUKERNEL_ASSERT(( tpu_aligned_feature_size(1, out_shape[out_dim-1], grad_dtype) +
                               DIV_UP(cur_len, NPU_NUM) * tpu_aligned_feature_size(1, out_shape[out_dim-1], grad_dtype))
                               < LOCAL_MEM_SIZE);

            dim4 one_out_shape = {.n = 1, .c = 1, .h = 1, .w = out_shape[out_dim-1]};
            dim4 one_in_shape = {.n = 1, .c = cur_len, .h = 1, .w = one_out_shape.w};
            memcpy(tpu_global_mem_addr(from_index_global_addr), in_idx_window + start_idx, sizeof(int) * cur_len);
            tpu_flush_cache(from_index_global_addr, ALIGN(sizeof(int) * cur_len, 64));
            dim4 global_buffer_shape= {.n = 1, .c = 1, .h = cur_len, .w = out_shape[out_dim-1]};
            tpu_gdma_h_gather_S2S(from_buffer_global_addr, gradout_global_addr, from_index_global_addr,
                                false, zero_, &global_buffer_shape, go_shape.h, NULL, NULL, NULL, grad_dtype);
            tpu_gdma_cpy_S2L(in_local_addr, from_buffer_global_addr, &one_in_shape,NULL, NULL, grad_dtype);
            // cpy out -> local mem
            if (start_idx == 0)
                tpu_gdma_cpy_S2L(res_local_addr, out_global_addr + idx_out * one_out_shape.w * tpu_data_type_size(grad_dtype),
                                &one_out_shape, NULL, NULL, grad_dtype);

            scalar_t one_ = {.f32 = 1.0};
            if (grad_dtype == DT_FP16) one_ = tpu_fp_cast(one_, DT_FP32, DT_FP16, RM_HALF_TO_EVEN);
            padding_t conv_pad = {.bottom=0, .top = 0, .left = 0, .right = 0};
            dim2 conv_kernel = {.h = 1, .w = 1};
            dim2 conv_stride = {.h = 1, .w = 1};
            dim2 conv_dilation = {.h = 1, .w = 1};
            tpu_bdc_fp_conv2d_kernel_const(res_local_addr, in_local_addr, 0, one_, &one_in_shape, NULL, 1,  &conv_kernel,
                                        &conv_pad, &conv_stride, &conv_dilation, grad_dtype,grad_dtype, false, false);
            start_idx += cur_len;
            if (start_idx >= len)
                tpu_gdma_general_cpy_L2S(out_global_addr + idx_out * one_out_shape.w * tpu_data_type_size(grad_dtype),res_local_addr,
                                        &one_out_shape, &one_out_shape, NULL, NULL, grad_dtype);
        }
        ptr_reserved_idx = delete_nodel(ptr_reserved_idx);
    }
    
    free(in_idx_window);
    free(out_idx_window);
#endif
}

void tpu_kernel_api_emb_backward(const void* args) {
    sg_api_emb_backward_t *api = (sg_api_emb_backward_t *)args;
    tpu_initialize();
    nodechip_embedding_backward(
        api->gradout_global_addr,
        api->index_global_addr,
        api->output_global_addr,
        api->sorted_index_global_addr,
        api->sorted_index_index_global_addr,
        api->from_index_global_addr,
        api->to_index_global_addr,
        api->from_buffer_global_addr,
        api->to_buffer_global_addr,
        api->window_size,
        api->gradout_shape,
        api->idx_shape,
        api->out_shape,
        api->gradout_dim,
        api->idx_dim,
        api->out_dim,
        tpu_type_convert(api->grad_dtype),
        tpu_type_convert(api->idx_dtype));
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_emb_backward);
