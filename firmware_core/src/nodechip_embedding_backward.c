#include "sg_api_struct.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"
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
    scalar_t one_ = {.f32 = 1.0};
    if (grad_dtype == DT_FP16) one_ = tpu_fp_cast(one_, DT_FP32, DT_FP16, RM_HALF_TO_EVEN);

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

    tpu_invalidate_cache(sorted_index_global_addr, ALIGN( NUM_index * sizeof(int), 64));
    tpu_invalidate_cache(sorted_index_index_global_addr, ALIGN(NUM_index * sizeof(int), 64));

    int* sorted_index      = (int*)malloc(NUM_index * sizeof(int));
    int* sorted_index_indx = (int*)malloc(NUM_index * sizeof(int));
    int* tmp1 = NULL, *tmp2 = NULL;
    tmp1 = (int*)tpu_global_mem_addr(sorted_index_global_addr);
    tmp2 = (int*)tpu_global_mem_addr(sorted_index_index_global_addr);
    for(int i = 0; i < NUM_index; i++){
        sorted_index[i] = tmp1[i];
        sorted_index_indx[i] = tmp2[i];
    }
    local_addr_t from_local_addr = 0;
    local_addr_t to_local_addr = from_local_addr + DIV_UP(window_size , NPU_NUM) * tpu_aligned_feature_size(1, o_shape.w, grad_dtype);
    local_addr_t res_local_addr = to_local_addr +  DIV_UP(window_size , NPU_NUM) * tpu_aligned_feature_size(1, o_shape.w, grad_dtype);
    TPUKERNEL_ASSERT(( 3 * DIV_UP(window_size, NPU_NUM) * tpu_aligned_feature_size(1, o_shape.w, grad_dtype))
                        < LOCAL_MEM_SIZE);
    int window_cur = 0;
    int *in_idx_window =  (int*)malloc( window_size * sizeof(int) );
    int *out_idx_window = (int*)malloc( window_size * sizeof(int) );
    List reserved_list;
    reserved_list.head.next = &reserved_list.tail; reserved_list.tail.prv = &reserved_list.head;
    reserved_list.head.prv = NULL; reserved_list.tail.next = NULL;
    ListNode* ptr_reserved_idx = reserved_list.head.next; 
    int start_idx = 0;
    while ( start_idx < NUM_index )
    {
        //printf("start_idx: %d \n", start_idx);
        if (ptr_reserved_idx == &reserved_list.tail){
            int idx_out = sorted_index[start_idx];
            int idx_in  = sorted_index_indx[start_idx];
            if ( idx_in_window( idx_out, out_idx_window, window_cur) ){
                ListNode *node = (ListNode*) malloc(sizeof(ListNode));
                node->prv = reserved_list.tail.prv;
                node->next = &reserved_list.tail;
                node->index = start_idx;
                reserved_list.tail.prv->next = node;
                reserved_list.tail.prv = node;
                do{
                    start_idx++;
                    //printf("----start_idx: %d \n", start_idx);
                }while (start_idx < NUM_index && 
                        sorted_index[start_idx] == idx_out);
            }else{
                in_idx_window[window_cur] = idx_in;
                out_idx_window[window_cur] = idx_out;
                window_cur++;
                start_idx++;
            }
        }else{
            int idx_out = sorted_index[ptr_reserved_idx->index];
            int idx_in  = sorted_index_indx[ptr_reserved_idx->index];
            if (idx_in_window(idx_out, out_idx_window, window_cur)){
                ptr_reserved_idx = ptr_reserved_idx->next;
            }else{ // find useful idx in list
                in_idx_window[window_cur] = idx_in;
                out_idx_window[window_cur] = idx_out;
                window_cur++;
                // delete node in list
                if (ptr_reserved_idx->index + 1 < NUM_index){
                    int next_idx_out = sorted_index[ptr_reserved_idx->index + 1];
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
        if ((window_cur == window_size || start_idx == NUM_index) && window_cur > 0 && window_cur <= window_size) { // enough to caculate
            ptr_reserved_idx = reserved_list.head.next;
            //do caculate here
            tpu_poll(); // necessary, but harm for performance 
            tmp1 = (int*)tpu_global_mem_addr(from_index_global_addr);
            tmp2 = (int*)tpu_global_mem_addr(to_index_global_addr);
            for (int i = 0; i < window_cur; i++){
                tmp1[i] = in_idx_window[i];
                tmp2[i] = out_idx_window[i];
            }
            tpu_flush_cache(from_index_global_addr, ALIGN(sizeof(int) * window_cur, 64));
            tpu_flush_cache(to_index_global_addr, ALIGN(sizeof(int) * window_cur, 64));
            dim4 add_shape = {.n = 1, .c = window_cur, .h = 1, .w=o_shape.w};
            dim4 from_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
            tpu_gdma_h_gather_S2L(from_local_addr, gradout_global_addr, from_index_global_addr, false, zero_, &add_shape, go_shape.h, NULL, &from_stride, NULL, grad_dtype);
            dim4 to_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
            tpu_gdma_h_gather_S2L(to_local_addr, out_global_addr, to_index_global_addr, false, zero_, &add_shape, o_shape.h, NULL, &to_stride, NULL, grad_dtype);

            tpu_bdc_fp_add(res_local_addr, from_local_addr, to_local_addr, &add_shape, NULL, NULL, NULL, grad_dtype);

            dim4 scatter_out_stride = {.n = 0 , .c = 0, .h = add_shape.w, .w = 1};
            dim4 scatter_oshape = {.n = o_shape.n, .c = window_cur, .h = o_shape.h, .w = o_shape.w};
            tpu_gdma_h_scatter_L2S(out_global_addr, res_local_addr, to_index_global_addr, false, &scatter_oshape, 1, &scatter_out_stride, NULL, NULL, grad_dtype);
            window_cur = 0;
        }
    }
#if 1
    // list have reserved data
    while(reserved_list.head.next != &reserved_list.tail){
        // 1. all 
        window_cur = 0;
        ptr_reserved_idx = reserved_list.head.next;
        while(ptr_reserved_idx != &reserved_list.tail){
            // do calculate
            int idx_out = sorted_index[ptr_reserved_idx->index];
            int idx_in  = sorted_index_indx[ptr_reserved_idx->index];
            in_idx_window[window_cur] = idx_in;
            out_idx_window[window_cur] = idx_out;
            window_cur++;
            if (ptr_reserved_idx->index + 1 < NUM_index){
                int next_idx_out = sorted_index[ptr_reserved_idx->index + 1];
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
        // 2. do
        tpu_poll(); // necessary, but harm for performance 
        tmp1 = (int*) tpu_global_mem_addr(from_index_global_addr);
        tmp2 = (int*) tpu_global_mem_addr(to_index_global_addr);
        for(int i = 0; i < window_cur; i++){
            tmp1[i] = in_idx_window[i];
            tmp2[i] = out_idx_window[i];
        }
        tpu_flush_cache(from_index_global_addr, ALIGN(sizeof(int) * window_cur, 64));
        tpu_flush_cache(to_index_global_addr, ALIGN(sizeof(int) * window_cur, 64));

        dim4 add_shape = {.n = 1, .c = window_cur, .h = 1, .w=o_shape.w};
        dim4 from_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
        tpu_gdma_h_gather_S2L(from_local_addr, gradout_global_addr, from_index_global_addr, false, zero_, &add_shape, go_shape.h, NULL, &from_stride, NULL, grad_dtype);
        dim4 to_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
        tpu_gdma_h_gather_S2L(to_local_addr, out_global_addr, to_index_global_addr, false, zero_, &add_shape, o_shape.h, NULL, &to_stride, NULL, grad_dtype);

        tpu_bdc_fp_add(res_local_addr, from_local_addr, to_local_addr, &add_shape, NULL, NULL, NULL, grad_dtype);

        dim4 scatter_out_stride = {.n = 0 , .c = 0, .h = add_shape.w, .w = 1};
        dim4 scatter_oshape = {.n = o_shape.n, .c = window_cur, .h = o_shape.h, .w = o_shape.w};
        tpu_gdma_h_scatter_L2S(out_global_addr, res_local_addr, to_index_global_addr, false, &scatter_oshape, 1, &scatter_out_stride, NULL, NULL, grad_dtype);
    }
#endif
    free(in_idx_window);
    free(out_idx_window);
    free(sorted_index);
    free(sorted_index_indx);
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
