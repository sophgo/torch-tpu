#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include <stdlib.h>
#include <string.h>

//[Error] will lead you to some improvements.
//CONTROL_CODE
#define SG2260_GLOBAL_MEM 0x100000000
#define SG2260_LOCAL_MEM  LOCAL_MEM_SIZE

//Slicing Pattern
#define TOP_CONCURRENCY_H    1 //only slice H
#define TOP_CONCURRENCY_H_BS 2 //slice or loop H & NS,
#define TOP_CONCURRENCY_V_H  3 //slice or loop V & H, not implemented

//SubPattern for TOP_CONCURRENCY_H_BS
#define SUBPATTERN_CODE_LOOP_H  0 //loop H, not implemented
#define SUBPATTERN_CODE_LOOP_BS 1 //loop BS, default

//NPU Computing Pattern
#define NPU_LOOP_V 0                     // {v0,v1..}
#define NPU_RECOLLECT_V_WITH_SAME_SUM 1  // {g0={v0,v1; |v0|==|v1|}}

//Reduce Compute Pattern
#define SUPER_SET 0  //only support for grad_Y==ones
#define L2_PATTERN 1
#define AVG_SIM 2    //default
typedef struct List_Node {
  int         index;
  struct List_Node    *prv;
  struct List_Node    *next;
} ListNode;
typedef struct _List {
  ListNode    head;
  ListNode    tail;
} List;

bool idx_in_window ( int idx,  int* window, int pos ) {
  if ( pos == 0 ) return false;
  if ( window[pos - 1] == idx ) return true;
  return false;
}

ListNode* delete_nodel ( ListNode* pnode ) {
  pnode->next->prv = pnode->prv;
  pnode->prv->next = pnode->next;
  ListNode *to_deleted = pnode;
  pnode = pnode->next;
  free ( to_deleted );
  return pnode;
}

void nodechip_embedding_backward (
global_addr_t gradout_global_addr,
global_addr_t index_global_addr,
global_addr_t out_global_addr,
global_addr_t sorted_index_global_addr,
global_addr_t sorted_index_index_global_addr,
global_addr_t from_index_global_addr,
global_addr_t to_index_global_addr,
int           window_size,
const int     *gradout_shape,
const int     *index_shape,
const int     *out_shape,
int           gradout_dim,
int           idx_dim,
int           out_dim,
data_type_t   grad_dtype,
bool is_index_int64 ) {
  TPUKERNEL_ASSERT ( is_index_int64 == false );
  // 1. fill out_global with zero
  scalar_t zero_ = {.u32 = 0};
  scalar_t one_ = {.f32 = 1.0};
  if ( grad_dtype == DT_FP16 ) one_ = tpu_fp_cast ( one_, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
  dim4 o_shape = {.n = 1, .c = 1, .h = 1, .w = out_shape[out_dim - 1]};
  for ( int i = 0; i < out_dim - 1; i++ ) {
    o_shape.h *= out_shape[i];
  }
  dim4 go_shape = {.n = 1, .c = 1, .h = 1, .w = gradout_shape[out_dim - 1]};
  for ( int i = 0; i < gradout_dim - 1; i++ ) {
    go_shape.h *= gradout_shape[i];
  }
  tpu_gdma_set_C_system (
  out_global_addr,
  zero_,
  &o_shape,
  NULL,
  grad_dtype );
  int NUM_index = 1;
  //printf("index_shape: ");
  for ( int i = 0; i < idx_dim; i++ ) {
    NUM_index *= index_shape[i];
    //printf("  %d,", index_shape[i]);
  }
  //printf("\n");
  // 2. sort index
  tpu_hau_sort_natural_index ( sorted_index_global_addr,
                               sorted_index_index_global_addr,
                               index_global_addr,
                               NUM_index, NUM_index, false, DT_INT32 );
  tpu_hau_poll();
  tpu_invalidate_cache ( sorted_index_global_addr, ALIGN ( NUM_index * sizeof ( int ), 64 ) );
  tpu_invalidate_cache ( sorted_index_index_global_addr, ALIGN ( NUM_index * sizeof ( int ), 64 ) );
  int* sorted_index      = ( int* ) malloc ( NUM_index * sizeof ( int ) );
  int* sorted_index_indx = ( int* ) malloc ( NUM_index * sizeof ( int ) );
  int* tmp1 = NULL, *tmp2 = NULL;
  tmp1 = ( int* ) tpu_global_mem_addr ( sorted_index_global_addr );
  tmp2 = ( int* ) tpu_global_mem_addr ( sorted_index_index_global_addr );
  for ( int i = 0; i < NUM_index; i++ ) {
    sorted_index[i] = tmp1[i];
    sorted_index_indx[i] = tmp2[i];
  }
  local_addr_t from_local_addr = 0;
  local_addr_t to_local_addr = from_local_addr + DIV_UP ( window_size, NPU_NUM ) * tpu_aligned_feature_size ( 1, o_shape.w, grad_dtype );
  local_addr_t res_local_addr = to_local_addr +  DIV_UP ( window_size, NPU_NUM ) * tpu_aligned_feature_size ( 1, o_shape.w, grad_dtype );
  TPUKERNEL_ASSERT ( ( 3 * DIV_UP ( window_size, NPU_NUM ) * tpu_aligned_feature_size ( 1, o_shape.w, grad_dtype ) )
                     < LOCAL_MEM_SIZE );
  int window_cur = 0;
  int *in_idx_window =  ( int* ) malloc ( window_size * sizeof ( int ) );
  int *out_idx_window = ( int* ) malloc ( window_size * sizeof ( int ) );
  List reserved_list;
  reserved_list.head.next = &reserved_list.tail; reserved_list.tail.prv = &reserved_list.head;
  reserved_list.head.prv = NULL; reserved_list.tail.next = NULL;
  ListNode* ptr_reserved_idx = reserved_list.head.next;
  int start_idx = 0;
  while ( start_idx < NUM_index )
  {
    //printf("start_idx: %d \n", start_idx);
    if ( ptr_reserved_idx == &reserved_list.tail ) {
      int idx_out = sorted_index[start_idx];
      int idx_in  = sorted_index_indx[start_idx];
      if ( idx_in_window ( idx_out, out_idx_window, window_cur ) ) {
        ListNode *node = ( ListNode* ) malloc ( sizeof ( ListNode ) );
        node->prv = reserved_list.tail.prv;
        node->next = &reserved_list.tail;
        node->index = start_idx;
        reserved_list.tail.prv->next = node;
        reserved_list.tail.prv = node;
        do {
          start_idx++;
          //printf("----start_idx: %d \n", start_idx);
        } while ( start_idx < NUM_index &&
                  sorted_index[start_idx] == idx_out );
      } else {
        in_idx_window[window_cur] = idx_in;
        out_idx_window[window_cur] = idx_out;
        window_cur++;
        start_idx++;
      }
    } else {
      int idx_out = sorted_index[ptr_reserved_idx->index];
      int idx_in  = sorted_index_indx[ptr_reserved_idx->index];
      if ( idx_in_window ( idx_out, out_idx_window, window_cur ) ) {
        ptr_reserved_idx = ptr_reserved_idx->next;
      } else { // find useful idx in list
        in_idx_window[window_cur] = idx_in;
        out_idx_window[window_cur] = idx_out;
        window_cur++;
        // delete node in list
        if ( ptr_reserved_idx->index + 1 < NUM_index ) {
          int next_idx_out = sorted_index[ptr_reserved_idx->index + 1];
          if ( next_idx_out == idx_out ) // have another value in list
          {
            ptr_reserved_idx->index++;
            ptr_reserved_idx = ptr_reserved_idx->next;
          }
          else // delete node
          {
            ptr_reserved_idx = delete_nodel ( ptr_reserved_idx );
          }
        } else {
          ptr_reserved_idx = delete_nodel ( ptr_reserved_idx );
        }
      }
    }
    if ( ( window_cur == window_size || start_idx == NUM_index ) && window_cur > 0 && window_cur <= window_size ) { // enough to caculate
      ptr_reserved_idx = reserved_list.head.next;
      //do caculate here
      tpu_poll(); // necessary, but harm for performance
      tmp1 = ( int* ) tpu_global_mem_addr ( from_index_global_addr );
      tmp2 = ( int* ) tpu_global_mem_addr ( to_index_global_addr );
      for ( int i = 0; i < window_cur; i++ ) {
        tmp1[i] = in_idx_window[i];
        tmp2[i] = out_idx_window[i];
      }
      tpu_flush_cache ( from_index_global_addr, ALIGN ( sizeof ( int ) * window_cur, 64 ) );
      tpu_flush_cache ( to_index_global_addr, ALIGN ( sizeof ( int ) * window_cur, 64 ) );
      dim4 add_shape = {.n = 1, .c = window_cur, .h = 1, .w = o_shape.w};
      dim4 from_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
      tpu_gdma_h_gather_S2L ( from_local_addr, gradout_global_addr, from_index_global_addr, false, zero_, &add_shape, go_shape.h, NULL, &from_stride, NULL, grad_dtype );
      dim4 to_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
      tpu_gdma_h_gather_S2L ( to_local_addr, out_global_addr, to_index_global_addr, false, zero_, &add_shape, o_shape.h, NULL, &to_stride, NULL, grad_dtype );
      tpu_bdc_fp_add ( res_local_addr, from_local_addr, to_local_addr, &add_shape, NULL, NULL, NULL, grad_dtype );
      dim4 scatter_out_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
      dim4 scatter_oshape = {.n = o_shape.n, .c = window_cur, .h = o_shape.h, .w = o_shape.w};
      tpu_gdma_h_scatter_L2S ( out_global_addr, res_local_addr, to_index_global_addr, false, &scatter_oshape, 1, &scatter_out_stride, NULL, NULL, grad_dtype );
      window_cur = 0;
    }
  }
#if 1
  // list have reserved data
  while ( reserved_list.head.next != &reserved_list.tail ) {
    // 1. all
    window_cur = 0;
    ptr_reserved_idx = reserved_list.head.next;
    while ( ptr_reserved_idx != &reserved_list.tail ) {
      // do calculate
      int idx_out = sorted_index[ptr_reserved_idx->index];
      int idx_in  = sorted_index_indx[ptr_reserved_idx->index];
      in_idx_window[window_cur] = idx_in;
      out_idx_window[window_cur] = idx_out;
      window_cur++;
      if ( ptr_reserved_idx->index + 1 < NUM_index ) {
        int next_idx_out = sorted_index[ptr_reserved_idx->index + 1];
        if ( next_idx_out == idx_out ) // have another value in list
        {
          ptr_reserved_idx->index++;
          ptr_reserved_idx = ptr_reserved_idx->next;
        }
        else // delete node
        {
          ptr_reserved_idx = delete_nodel ( ptr_reserved_idx );
        }
      } else {
        ptr_reserved_idx = delete_nodel ( ptr_reserved_idx );
      }
    }
    // 2. do
    tpu_poll(); // necessary, but harm for performance
    tmp1 = ( int* ) tpu_global_mem_addr ( from_index_global_addr );
    tmp2 = ( int* ) tpu_global_mem_addr ( to_index_global_addr );
    for ( int i = 0; i < window_cur; i++ ) {
      tmp1[i] = in_idx_window[i];
      tmp2[i] = out_idx_window[i];
    }
    tpu_flush_cache ( from_index_global_addr, ALIGN ( sizeof ( int ) * window_cur, 64 ) );
    tpu_flush_cache ( to_index_global_addr, ALIGN ( sizeof ( int ) * window_cur, 64 ) );
    dim4 add_shape = {.n = 1, .c = window_cur, .h = 1, .w = o_shape.w};
    dim4 from_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
    tpu_gdma_h_gather_S2L ( from_local_addr, gradout_global_addr, from_index_global_addr, false, zero_, &add_shape, go_shape.h, NULL, &from_stride, NULL, grad_dtype );
    dim4 to_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
    tpu_gdma_h_gather_S2L ( to_local_addr, out_global_addr, to_index_global_addr, false, zero_, &add_shape, o_shape.h, NULL, &to_stride, NULL, grad_dtype );
    tpu_bdc_fp_add ( res_local_addr, from_local_addr, to_local_addr, &add_shape, NULL, NULL, NULL, grad_dtype );
    dim4 scatter_out_stride = {.n = 0, .c = 0, .h = add_shape.w, .w = 1};
    dim4 scatter_oshape = {.n = o_shape.n, .c = window_cur, .h = o_shape.h, .w = o_shape.w};
    tpu_gdma_h_scatter_L2S ( out_global_addr, res_local_addr, to_index_global_addr, false, &scatter_oshape, 1, &scatter_out_stride, NULL, NULL, grad_dtype );
  }
#endif
  free ( in_idx_window );
  free ( out_idx_window );
  free ( sorted_index );
  free ( sorted_index_indx );
}

void tpu_kernel_api_embedding_backward ( const void* args ) {
  sg_api_embedding_backward_t *api = ( sg_api_embedding_backward_t * ) args;
  tpu_initialize();
  nodechip_embedding_backward (
  api->grad_output_global_addr,
  api->index_global_addr,
  api->grad_input_global_addr,
  api->sorted_index_global_addr,
  api->sorted_index_index_global_addr,
  api->from_index_global_addr,
  api->to_index_global_addr,
  api->window_size,
  api->grad_output_shape,
  api->index_shape,
  api->grad_input_shape,
  api->grad_output_dim,
  api->index_dim,
  api->grad_input_dim,
  ( data_type_t ) api->grad_output_dtype,
  api->is_index_int64 );
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_embedding_backward );

// Counting total elemnet of a given shape
// Assume MEM is densely contiguous
static inline int count_numel_dim4(const dim4 shape) {
  return shape.n* shape.c * shape.h * shape.w;
}

//Find a min 64k length apporaching to given length
static inline int length_align_64bit_apporach(int origin_length, data_type_t Index_dtype){
  TPUKERNEL_ASSERT_INFO(Index_dtype == DT_INT32, "You are trying to align index_addr to 64bits, but Only INT32 is supported!");
  return DIV_UP(origin_length * (tpu_data_type_size(Index_dtype)), 64)*64;
}

//Function: [D] -> [D_sliced]
//Stratgey: Max_Core_Utilization
//          Given:       total_num, current_support_core_num, current_core_id
//          Source       Free       Free                      Free
//          Target       expected_current_slice, expected_avg_slice, expected_secs
//          Compute:     total_num = expected_avg_slice * expected_secs + expected_left_slice
//          Constraints: expected_secs <= current_support_core_num
//          where expected_current_slice = expected_avg_slice   (current_core_id <  expected_secs-1)
//                                     or= expected_left_slice  (current_core_id == expected_secs-1)
static inline void compute_current_slice_info_multi_core_1D_with_given_cores_with_given_id(int total_num, int* expected_current_slice,
                                                                                           int* expected_avg_slice, int* expected_secs,
                                                                                           int current_support_core_num, int current_core_id) {
  const int core_num = current_support_core_num;
  const int core_idx = current_core_id;
  const int avgnum_element_each_core = DIV_UP(total_num, core_num);
  const int num_max_core_needed      = DIV_UP(total_num, avgnum_element_each_core);
  TPUKERNEL_ASSERT_INFO(num_max_core_needed <= core_num, "You are given too many cores, BM1684X has 1 || SG2260 has 8");
  int current_num_for_current_core = avgnum_element_each_core;
  if (core_idx == num_max_core_needed - 1) {
    current_num_for_current_core = total_num - avgnum_element_each_core * (num_max_core_needed - 1);
  }
  *expected_current_slice = current_num_for_current_core;
  *expected_avg_slice     = avgnum_element_each_core;
  *expected_secs          = num_max_core_needed;
}


//Function: [D] -> [D_sliced]
//Stratgey: Max_Core_Utilization
//          Given:       total_num, current_support_core_num, current_core_id
//          Source       Free       Free                      TPU
//          Target       expected_current_slice, expected_avg_slice, expected_secs
//          Compute:     total_num = expected_avg_slice * expected_secs + expected_left_slice
//          Constraints: expected_secs <= current_support_core_num
//          where expected_current_slice = expected_avg_slice   (current_core_id <  expected_secs-1)
//                                     or= expected_left_slice  (current_core_id == expected_secs-1)
static inline void compute_current_slice_info_multi_core_1D_with_given_cores(int total_num, int* expected_current_slice,
                                                         int* expected_avg_slice, int* expected_secs, int current_support_core_num) {
  const int core_num = current_support_core_num;
  const int core_idx = tpu_core_index();
  compute_current_slice_info_multi_core_1D_with_given_cores_with_given_id(total_num, expected_current_slice,
                                                         expected_avg_slice,  expected_secs, core_num, core_idx);
}

//order: (X,Y)->(X, Y_slice, Y_secs)
//Stratgey: Max_Core_Utilization
//Function: [D] -> [D_sliced]
//          Given:       total_num, current_support_core_num, current_core_id
//          Source       Free       TPU                       TPU
//          Target       expected_current_slice, expected_avg_slice, expected_secs
//          Compute:     total_num = expected_avg_slice * expected_secs + expected_left_slice
//          Constraints: expected_secs <= current_support_core_num
//          where expected_current_slice = expected_avg_slice   (current_core_id <  expected_secs-1)
//                                     or= expected_left_slice  (current_core_id == expected_secs-1)
static inline void compute_current_slice_info_multi_core_1D(int total_num, int* expected_current_slice,
                                                         int* expected_avg_slice, int* expected_secs) {
  const int max_core_num = tpu_core_num();
  compute_current_slice_info_multi_core_1D_with_given_cores(total_num, expected_current_slice, expected_avg_slice, expected_secs, max_core_num);
}

//Order: (X,Y)->(X, Y_loop, loop_num),
//Function: (X,Y)->(X, Y_loop, loop_num)
//          Given:       total_num, num_loop_upper, current_loop_id
//          Source       Free       Free            Free
//          Target       expected_current_slice, expected_avg_slice
//          Compute:     total_num = expected_avg_slice * (loop_num - 1) + expected_left_slice
//          Constraints: expected_secs <= num_loop_upper
//          where expected_current_slice = expected_avg_slice   (current_loop_id <  loop_num-1)
//                                     or= expected_left_slice  (current_loop_id == loop_num-1)
static inline void compute_current_loop_info_each_core_1D_with_loop_idx(int total_num, int* expected_current_slice,
                                                         int* expected_avg_slice, int num_loop_upper, int current_loop_id) {
  const int avgnum_element_each_loop = DIV_UP(total_num, num_loop_upper);

  int current_num_for_current_loop = avgnum_element_each_loop;
  if (current_loop_id == num_loop_upper - 1) {
    current_num_for_current_loop = total_num - avgnum_element_each_loop * (num_loop_upper - 1);
  }
  TPUKERNEL_ASSERT_INFO(current_num_for_current_loop > 0, "Split Loop Error!");
  *expected_current_slice = current_num_for_current_loop;
  *expected_avg_slice     = avgnum_element_each_loop;
}


//Function: Tensor (X,Y)->(min_cores, X_slice, Y_slice)
//          Given:       total_num_x, total_num_y, current_support_core_num, current_core_id
//          Source       Free       Free           TPU                       TPU
//          Target       others wih *
//Compute:
//          X*Y = S_easy + S_other
//          S_easy = X_slice_avg* Y_slice_avg* (min_cores_x-1)*(min_cores_x-1)
//          S_other = X_slice_avg* Y_slice_real* (min_cores_x-1) + X_slice_real* Y_slice_avg*(min_cores_y-1) + Y_slice_real* X_slice_real
// Constraints:
//          C1: min_cores_x + min_cores_y == min_cores <= tpu_core_num()
//          C2: (XY)_size <= L2_MEM
// Strategy:
//0) Assign priority to max(X,Y)
//1) assign <current_core_num><-tpu_core_num()
//2) slice 1st_priority dim to min_cores_1  based on  <current_core_num>
//3) slice 2nd_priority dim to min_cores_2  based on  <current_core_num>
//4.1）  if C1, exit
//4.2.1.1） if ~C1 && C2: L2 boldly holds 2nd_priority, keep 2nd_priority;
//4.2.1.2）    Revert  min_cores = min_cores_1
//4.2.2.1) if ~C1 && ~C2: <current_core_num> -=1 (because 1st_priority uses too many cores)
//4.2.2.2）    Return  step 2)
static inline void compute_current_slice_info_multi_core_2D_correlated(int total_num_x, int total_num_y,
                                                         int* expected_current_slice_x, int* expected_current_slice_y,
                                                         int* expected_avg_slice_x, int* expected_avg_slice_y,
                                                         int * min_cores_total,
                                                         int * min_cores_x,
                                                         int * min_cores_y,
                                                         data_type_t dtype) {
  const int core_num = tpu_core_num();
  //Assuming operate only 1 global_slice. In such case, just adjust some slices only if they are too long.
  TPUKERNEL_ASSERT(total_num_x * total_num_y * tpu_data_type_size(dtype) < SG2260_GLOBAL_MEM);
  //step 0
  int max_dim = total_num_x > total_num_y ? 0 : 1;
  // int prority_num_1st = total_num_x > total_num_y ? total_num_x : total_num_y;
  int prority_num_2nd = total_num_x <= total_num_y ? total_num_x : total_num_y;
  //step 1
  int current_core_num =  tpu_core_num();
  int current_core_num_x =  tpu_core_num();
  int current_core_num_y =  tpu_core_num();

  while(current_core_num_x>0 && current_core_num_y >0) {
    //step 2
    if (max_dim==0)
      compute_current_slice_info_multi_core_1D_with_given_cores(total_num_x,expected_current_slice_x,expected_avg_slice_x, min_cores_x, current_core_num_x);
    else
      compute_current_slice_info_multi_core_1D_with_given_cores(total_num_y,expected_current_slice_y,expected_avg_slice_y, min_cores_y, current_core_num_y);
    //step 3
    if (max_dim==1)
      compute_current_slice_info_multi_core_1D_with_given_cores(total_num_x,expected_current_slice_x,expected_avg_slice_x, min_cores_x, current_core_num_x);
    else
      compute_current_slice_info_multi_core_1D_with_given_cores(total_num_y,expected_current_slice_y,expected_avg_slice_y, min_cores_y, current_core_num_y);
    //Use avg_slice rather than current_slice to ensure single-solution of this while-loop.
    int prority_1st_slice = total_num_x > total_num_y ? *expected_avg_slice_x : *expected_avg_slice_y;
      *min_cores_total = *min_cores_x * *min_cores_y;
    //step 4.1
    if (* min_cores_total <= current_core_num) break;
    //step 4.2.1
    if (* min_cores_total >= current_core_num && (prority_1st_slice * prority_num_2nd *  tpu_data_type_size(dtype) < SG2260_GLOBAL_MEM)) {
    //Assume priority_1st(Y) sliced but priority_2nd(X) is not
    // if (* min_cores_total >= current_core_num && (prority_1st_slice * prority_num_2nd < 50 * 192)) {
      *min_cores_x = total_num_x > total_num_y ? *min_cores_x : 1;
      *min_cores_y = total_num_x <= total_num_y ? *min_cores_y : 1;
      *min_cores_total = total_num_x > total_num_y ? *min_cores_x : *min_cores_y;
      //if some-dims not sliced, then force to corresponding total_num
      *expected_avg_slice_x =  total_num_x > total_num_y ?  *expected_avg_slice_x : total_num_x;
      *expected_avg_slice_y =  total_num_x <= total_num_y ? *expected_avg_slice_y : total_num_y;
      *expected_current_slice_x = total_num_x > total_num_y ? *expected_current_slice_x : total_num_x;
      *expected_current_slice_y = total_num_x <= total_num_y ? *expected_current_slice_y : total_num_y;
      TPUKERNEL_ASSERT(* min_cores_total == *min_cores_x * *min_cores_y );
      //Trick
      //1) if avg == current, then this dim is not sliced or its the last slice.
      break;
    }
    //step 4.2.2
    // if num_x*num_y > cores and size(x,y)<=MEM_L2,
    //     select first half groups data at max_dims-axis
    //Ex.  [7,3]->[3,3]->[3,2]
    if (total_num_x > total_num_y) {
        current_core_num_x -= current_core_num_x%2;
        current_core_num_x = current_core_num_x / 2;
        current_core_num_y = current_core_num/current_core_num_x;
    } else {
        current_core_num_y -= current_core_num_y%2;
        current_core_num_y = current_core_num_y / 2;
        current_core_num_x = current_core_num/current_core_num_y;
    }
  }

  //Get non-prime min_cores,
  //          As always slicing such dim with max_num
  //Step 5:  get (x,y) current value considering each {core_idx}
  //         Assuming idx-scanning is x-order prority.
  TPUKERNEL_ASSERT(* min_cores_total <= core_num );
  const int core_idx = tpu_core_index();
  //Notice core_idx starts from 0
  if (core_idx==0) {
      expected_current_slice_x = expected_avg_slice_x;
      expected_current_slice_y = expected_avg_slice_y;
  } else if (core_idx% *min_cores_y==*min_cores_y-1 && core_idx% *min_cores_x==*min_cores_x-1) {

  } else if (core_idx% *min_cores_y==*min_cores_y-1 && core_idx% *min_cores_x!=*min_cores_x-1) {
      expected_current_slice_x = expected_avg_slice_x;
  } else if (core_idx% *min_cores_y!=*min_cores_y-1 && core_idx% *min_cores_x==*min_cores_x-1) {
      expected_current_slice_y = expected_avg_slice_y;

  }  else  {
      expected_current_slice_x = expected_avg_slice_x;
      expected_current_slice_y = expected_avg_slice_y;
  }
}

//Function: if search_index_value is in data[:current_length]
int check_duplicate_value(int search_index_value, int * data,int current_length) {
  if (current_length==0) {
    return 0;
  }
  for(int i=0; i <current_length; i++) {
    if (data[i] == search_index_value)
      return 1;
  }
  return 0;
}

//Function: return id where data[id]==search_index_value && id <= current_length - 1
int search_duplicate_id(int search_index_value, int * data,int current_length) {
  for(int i=0; i <current_length; i++) {
      if (data[i] == search_index_value)
        return i;
  }
  return -1;
}

//Function: check is Tensor T is larger than L1_MEM
//          If not ,T can be fully hold on L1/L2/GMEM
void TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(dim4 object, data_type_t  dtype) {
    int l1_mem_sg2260 = SG2260_LOCAL_MEM;
    TPUKERNEL_ASSERT(object.n*object.h*object.w*object.c * tpu_data_type_size(dtype) <=l1_mem_sg2260); //only 256KB (8Mb) local_bitwidth is 18
}

static inline int gen_bank_4_size() {
   int bank_size =  tpu_local_mem_size_per_npu() / tpu_bank_num();
   int bank_4 = 4* bank_size;//SG2260_LOCAL_MEM;
   return  bank_4;
}
//Function:  grad_W[v,h]=sum(W[X[i',j'],h]; grad_Y=W
//HAU_sorted [i',j']->[i,j]: grad_W[v,h]=sum(W[X_sorted[i,j],h]
//X_sorted_duplicate  resorted_if_has_same_duplicate_num
//    grad_Y:W  [ij_0_v_0,ij_1_v_1,ij_2_v_1, ij_3_v_2] -> grad_Y[{ij_0_v_0,ij_3_v_2},{ij_1_v_1,ij_2_v_1}]
//num_repeated  [1,       2,                 1]        ->       [{1,1}              , 2]
//value_repeated[v_0,     v_1                v_2]      ->       [v_0,     ,v_2,       v_1]
static inline void Function_Recollect_Index(
  global_addr_t grad_Y_gathered_global_addr,
  global_addr_t recollected_global_flush_64b_aligned_addr,
  global_addr_t scatter_index_global_flush_64b_aligned_addr,
  int * NumArray_Duplicate_Index_Value,
  int * value_duplicate_index_value_recorder,
  int NUM_full_Index,
  int NUM_V_used,
  data_type_t grad_dtype)
{
  int * tmp_collected = NULL;
  tmp_collected = (int* ) tpu_global_mem_addr ( grad_Y_gathered_global_addr );
  int *tmp_collected_copy = ( int* ) malloc ( NUM_full_Index * sizeof ( int ) );
  memset( tmp_collected_copy, 0, NUM_full_Index*sizeof(int) );
  int flag_collect = 0;
  for (int k_null=0;k_null<NUM_full_Index;k_null++)
  for (int i_num_recollect=0;i_num_recollect<NUM_full_Index;i_num_recollect++) {
    int j_num_recollect = i_num_recollect+1;
    while(j_num_recollect<NUM_V_used) {
        int duplicate_value_i = NumArray_Duplicate_Index_Value[i_num_recollect];
        int duplicate_value_j = NumArray_Duplicate_Index_Value[j_num_recollect];
        //TODO we can further reduce by ++j_num_recollect if NumArray_Duplicate_Index_Value[j_num_recollect]==NumArray_Duplicate_Index_Value[j_num_recollect+1]
        if (duplicate_value_i > duplicate_value_j) {
              flag_collect = 1;
              //reorder index
              //seq_0, tmp_collected[i_start:i_end), seq_1,  tmp_collected[j_start:j_end), seq_2 ->
              //seq_0, tmp_collected[j_start:j_end), seq_1,  tmp_collected[i_start:i_end), seq_2
              //[start, end)
              int i_start = 0;
              for (int i_tmp=0; i_tmp<i_num_recollect;i_tmp++)
               i_start += NumArray_Duplicate_Index_Value[i_tmp];
              int i_end = i_start+duplicate_value_i;

              int j_start = 0;
              for (int i_tmp=0; i_tmp<j_num_recollect;i_tmp++)
               j_start += NumArray_Duplicate_Index_Value[i_tmp];
              int j_end = j_start+duplicate_value_j;

              int size = i_start       * sizeof(int);
              int size_history =0;
              if(size)
                for (int pp=0;pp<(int)(size/sizeof(int));pp++) {
                  tmp_collected_copy[pp+(int)(size_history/sizeof(int))] = tmp_collected[pp];
                }
                // memcpy(tmp_collected_copy, tmp_collected, size);
              size_history += size;

              size = duplicate_value_j  * sizeof(int);
              if(size)
                for (int pp=0;pp<(int)(size/sizeof(int));pp++) {
                  tmp_collected_copy[pp+(int)(size_history/sizeof(int))] = tmp_collected[j_start+pp];
                }
                // memcpy(tmp_collected_copy + size_history, tmp_collected + j_start *sizeof(int), size);
              size_history += size;

              size = (j_start-i_end)   * sizeof(int);
              if(size)
                for (int pp=0;pp<(int)(size/sizeof(int));pp++) {
                  tmp_collected_copy[pp+(int)(size_history/sizeof(int))] = tmp_collected[i_end+pp];
                }
                // memcpy(tmp_collected_copy + size_history, tmp_collected + i_end *sizeof(int), size);
              size_history += size;

              size = duplicate_value_i   * sizeof(int);
              if(size)
                for (int pp=0;pp<(int)(size/sizeof(int));pp++) {
                  tmp_collected_copy[pp+(int)(size_history/sizeof(int))] = tmp_collected[i_start+pp];
                }
                // memcpy(tmp_collected_copy + size_history, tmp_collected + i_start *sizeof(int), size);
              size_history += size;

              size = (NUM_full_Index-j_end) * sizeof(int);
              if(size)
              for (int pp=0;pp<(int)(size/sizeof(int));pp++) {
                  tmp_collected_copy[pp+(int)(size_history/sizeof(int))] = tmp_collected[j_end+pp];
              }
                // memcpy(tmp_collected_copy + size_history, tmp_collected + j_end *sizeof(int), size);
              size_history += size;

              for (int pp=0;pp<NUM_full_Index;pp++) {
                  tmp_collected[pp] = tmp_collected_copy[pp];
              }

              //memcpy(tmp_collected, tmp_collected_copy, NUM_full_Index*sizeof(int));
              //array_count from ancient, so duplicate_num latter
              int temp_duplicate_num = NumArray_Duplicate_Index_Value[i_num_recollect] ;
              NumArray_Duplicate_Index_Value[i_num_recollect] = NumArray_Duplicate_Index_Value[j_num_recollect] ;
              NumArray_Duplicate_Index_Value[j_num_recollect]= temp_duplicate_num;

              temp_duplicate_num = value_duplicate_index_value_recorder[i_num_recollect] ;
              value_duplicate_index_value_recorder[i_num_recollect] = value_duplicate_index_value_recorder[j_num_recollect] ;
              value_duplicate_index_value_recorder[j_num_recollect]= temp_duplicate_num;

              j_num_recollect +=1;
              i_num_recollect +=1;
        } else {
            j_num_recollect += 1;
        }
    }
  }
  free(tmp_collected_copy);
  if (flag_collect) {
    int* tmp2_3 = NULL;
    tmp2_3 = ( int* ) tpu_global_mem_addr ( recollected_global_flush_64b_aligned_addr );
    for ( int i = 0; i < NUM_full_Index; i++ ) {
      tmp2_3[i] = tmp_collected[i];
    }
    //Donot Reuse scatter_index_global_flush_64b_aligned_addr
    tpu_flush_cache ( scatter_index_global_flush_64b_aligned_addr, ALIGN ( sizeof ( int ) * NUM_full_Index, 64 ) );
    const dim4 shape_tmp_recollected = {.n=1,.c=1,.h=1,.w=NUM_full_Index};
    tpu_gdma_cpy_S2S(grad_Y_gathered_global_addr,recollected_global_flush_64b_aligned_addr, &shape_tmp_recollected, NULL,NULL, grad_dtype);
  }
}

//**********************************************************//
// Embedding Operation:
// 1)Forward_compute:
// Input: X~[B, S]    0<=x in X<=num_embeddings -1
//         X is not singularity, may duplicate
//         W~[num_embeddings, embedding_dim]~[V ,H]
//            V is sparse, H is locally dense
// Output Y~[B*S,  embedding_dim]
// i - B
// j - S
// h - H(embedding_dim)
// Y[i*m +j, h] = W[X[i, j]  ,h]
// 2)Backward_compute:
// gradW [num_embeddings, embedding_dim]~[V ,H]
// grad_X ~[B,S]
// i - B
// j - S
// v - V
// h - H
// grad_W[v, h] = SUM_{i,j}[grad_Y[X[i,j], h]
//**********************************************************//

//Chip Arch
//Concurrency                                     Shared               Shared
//                  Near-NPU     on-chip          Near-Cores           Off-chip
//device            Local        sDMA             L2                   GDMA
//Arch_solution_1   V              X                X                    V
//Arch_solution_2   V              V                X                    V
//Arch_solution_2   V              V                V                    V

void nodechip_embedding_backward_multi_core_basic_cell_patttern_one(
  global_addr_t grad_Y_global_addr,
  global_addr_t X_global_addr,
  global_addr_t grad_Weight_global_addr,
  global_addr_t sorted_index_value_global_addr,
  global_addr_t sorted_index_index_global_addr,
  global_addr_t scatter_index_global_flush_64b_aligned_addr,
  global_addr_t recollected_global_flush_64b_aligned_addr,
  const dim4    grad_Y_shape, //[B*S, H_sliced]
  const dim4    X_shape,      //[B, S]
  const dim4    grad_W_shape, //[V, H_sliced]
  const int     H_native,
  const int     first_loop_flag,
  data_type_t   grad_dtype) {
  //Step 0: check and initalize
  // TPUKERNEL_ASSERT(USING_L2);
  // TPUKERNEL_ASSERT(IN_L2_SRAM(grad_Weight_global_addr));
  scalar_t zero_ = {.u32 = 0};
  const int len_grad_dtype = tpu_data_type_size(grad_dtype);
  const int NUM_full_Index = count_numel_dim4(X_shape);
  const int H_sliced = grad_Y_shape.w;
  const int V = grad_W_shape.h; //different from NUM_V_used
  const int BS = grad_Y_shape.h;
  const int local_offset_1 = BS * H_sliced * len_grad_dtype; //tpu_aligned_feature_size (1, H_sliced, grad_dtype );
  const int local_offset_2 = V  * H_sliced * len_grad_dtype; // tpu_aligned_feature_size (1, H_sliced, grad_dtype ); //sum_num_per_v<=V
  int bank_4_size = gen_bank_4_size();
  local_addr_t grad_Y_gathered_local_addr    = 0;
  local_addr_t reduce_no_parallel_local_addr = bank_4_size;
  local_addr_t grad_Y_scatter_local_addr     = 2 * bank_4_size;
  local_addr_t partial_sum_local_addr        = 3 * bank_4_size;
  TPUKERNEL_ASSERT_INFO(local_offset_1 < bank_4_size, "BS is too large!");
  TPUKERNEL_ASSERT_INFO(local_offset_2 < bank_4_size, "V is too large or not sparse enough!");
  TPUKERNEL_ASSERT_INFO(partial_sum_local_addr+local_offset_2 < (u32)tpu_local_mem_size_per_npu(), "BS_Loop is too large");
  TPUKERNEL_ASSERT_INFO(partial_sum_local_addr+local_offset_1 < (u32)tpu_local_mem_size_per_npu(), "BS_Loop is too large");

  // const dim4 grad_W_stride = {.n = 0, .c = H_native, .h = H_native, .w = 1};
  // if (first_loop_flag) {
    // tpu_gdma_set_C_system (
    //   grad_Weight_global_addr,
    //   zero_,
    //   &grad_W_shape,
    //   &grad_W_stride,
    //   grad_dtype );
  // }
  //Shape Checker
  TPUKERNEL_ASSERT(grad_Y_shape.w == grad_W_shape.w);
  TPUKERNEL_ASSERT(grad_Y_shape.h == NUM_full_Index);
  // Step 1.0 Sorting full-Index(X)
  // Step_Arch: HAU-<GDMA>-GMEM
  tpu_hau_sort_natural_index ( sorted_index_value_global_addr,
                               sorted_index_index_global_addr,
                               X_global_addr,
                               NUM_full_Index, NUM_full_Index, false, DT_INT32 );
  tpu_hau_poll();
  tpu_invalidate_cache ( sorted_index_value_global_addr, ALIGN ( NUM_full_Index * sizeof ( int ), 64 ) );
  tpu_invalidate_cache ( sorted_index_index_global_addr, ALIGN ( NUM_full_Index * sizeof ( int ), 64 ) );

  //Step 1.1: Gather
  //Step_Arch: GMEM-<GDMA>-L1-<GDMA>-GMEM
  //  Function: z =  grad_Y[X[i,j], h]; z = grad_W[v, h]; v is free for every |v|==|x[x,j]|
  //  Input: Index_sorted_by_Index_value~[B*S]
  //  Output: Output[index_sorted]~[B*S,H_sliced]
  dim4 shape_gather_L2_to_L1 = {.n = 1, .c = 1, .h = BS, .w = H_sliced};
  TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(shape_gather_L2_to_L1, grad_dtype);
  //Note: H(.w-dim) is sliced, so each core operates sliced shape but strides in native shape.
  dim4 stride_gather_L2_to_L1 = {.n = 0, .c = 0, .h = H_native, .w = 1};
  //Question: which is faster ? tpu_gdma_h_gather_S2L or tpu_gdma_h_gather_S2S
  tpu_gdma_h_gather_S2L(grad_Y_gathered_local_addr, grad_Y_global_addr, sorted_index_index_global_addr,  false, zero_, &shape_gather_L2_to_L1, BS, NULL, &stride_gather_L2_to_L1, NULL, grad_dtype );
  //Step 1.2: Prepare All_reduce
  //[TODO] can be l2(gdma) over written when its input? (In BM1684X it's forbidden)
  //Note: grad_Y_gathered_global_addr is core-indepedent, so H_native is not considered on dst-stride.h
  global_addr_t grad_Y_gathered_global_addr = grad_Y_global_addr;
  tpu_gdma_cpy_L2S(grad_Y_gathered_global_addr, grad_Y_gathered_local_addr, &shape_gather_L2_to_L1, &stride_gather_L2_to_L1, NULL, grad_dtype);

  //Step 2: AVG_POOL to sim All_reduce
  //Step 2.1 Prepare duplicate Index for SUM-base
  //Step_Arch: GMEM-<AXI>-CPU
  //  Function: temp = SUM_{i,j}[ grad_Y[X[i,j], h] for {(i,j); |v|==|X[i,j]|}
  int* tmp1 = NULL;
  int* value_duplicate_index_value_recorder        = (int*)malloc(NUM_full_Index * sizeof(int));
  int* NumArray_Duplicate_Index_Value              = (int*)malloc(NUM_full_Index * sizeof(int));

  //[D2S]Send duplicate from  Sys to CPU
  tmp1 = ( int* ) tpu_global_mem_addr ( sorted_index_value_global_addr );
  // NUM_V_used : <=|V|. represents nums of V used
  int NUM_V_used = 0;
  //[CPU]Searching duplicate index from sorted Index
  for (int i = 0; i < NUM_full_Index; i++) {
    if (check_duplicate_value(tmp1[i], value_duplicate_index_value_recorder, NUM_V_used)) {
      int current_duplicate_id = search_duplicate_id(tmp1[i], value_duplicate_index_value_recorder, NUM_V_used);
      NumArray_Duplicate_Index_Value[current_duplicate_id] += 1;
    } else {
      value_duplicate_index_value_recorder[NUM_V_used] = tmp1[i]; //v
      NumArray_Duplicate_Index_Value[NUM_V_used] = 1; //v_num
      NUM_V_used += 1;
    }
  }

  //Search Max duplicate Index num
  //Function: sum(W[X==v]): Flops = #v * #{|x|==|v|}
  //Assuming O(reduce_full(sim)) is nearly const for any large-enough shape only if NPU_NUM is max-utlized
  //NPU Strategy 1: unrollong #{|x|==|v|}
  //               Flops_1 = #v; for-loop:#v ,i.e., NUM_V_used
  //NPU Strategy 2: unrollong #v
  //               Flops_2 = #{|x|==|v|}; for-loop, i.e., max_duplicate_num
  //Above all,     unrolling #{|x|==|v|} but partial-unrolling #v shall be ideal.
  //     i.e., sum (W[x==v]), for those group_k {|x|==|v_k|}, with same num #{|x|==|v_k|}
  //     Ex.  {1,3,1,2,1,2,3} ->{1,1,1,2,2,3,3}-> {{1,1,1},{2,2},{3,3}} = {g1,g2,g2}
  //           Then reduce in two loops: 1st g1, 2nd{g2,g3}
  //However, NUM_V_used is index_value_sorted rather than index_repeat_num_sorted
  //we need to reorder and gather additionally

  int max_duplicate_num = 0;
  for (int i = 0; i<  NUM_V_used; i++) {
    max_duplicate_num = max_duplicate_num > value_duplicate_index_value_recorder[i] ? max_duplicate_num : value_duplicate_index_value_recorder[i];
  }
  int NPU_Utils_Pattern = NPU_LOOP_V;
  if  (max_duplicate_num > NUM_V_used)
    NPU_Utils_Pattern = NPU_RECOLLECT_V_WITH_SAME_SUM; //NPU Compute 2
  else
    NPU_Utils_Pattern = NPU_LOOP_V; //NPU Compute 1

  //Step 2.2 Full_Reduce
  //Step_Arch: GMEM-<GDMA>-L1-<SDMA>-L2-<SDMA>-GMEM
  //Function:  sum(W[v]) for |v_j|==|v_i|
  //Input:     W_sorted~[B*S ,H_sliced]
  //output:    Operation ~ [NUM_V_used, H_sliced]

  global_addr_t reduce_global_addr = grad_Y_gathered_global_addr;
  //CONTROL_CODE ERRPR
  NPU_Utils_Pattern = NPU_RECOLLECT_V_WITH_SAME_SUM;
  if (NPU_Utils_Pattern==NPU_RECOLLECT_V_WITH_SAME_SUM) {
    Function_Recollect_Index(
      grad_Y_gathered_global_addr,
      recollected_global_flush_64b_aligned_addr,
      scatter_index_global_flush_64b_aligned_addr,
      NumArray_Duplicate_Index_Value,
      value_duplicate_index_value_recorder,
      NUM_full_Index,
      NUM_V_used,
      grad_dtype);
  }
  int counter_history = 0;
  int old_v_idx = 0, last_v_idx = 0;
  //as index_num is sorted . NumArray_Duplicate_Index_Value must be instances as [1,1,2,2,2,3,3,3...] in order
  //thus we can speed up by regroup [1,1],[2,2,2,2],...

  //NPU_parallel 2: group_size_index_with_same_duplicate_num ensure {1,1,2,1,1}->{{1,1},2,{1,1}} but not {{1,1,1,1},2}
  while(old_v_idx < NUM_V_used) {
      int sum_num_per_v = NumArray_Duplicate_Index_Value[old_v_idx];
      int group_size_index_with_same_duplicate_num = 1;
      last_v_idx = old_v_idx + 1;
      while (last_v_idx < NUM_V_used) {
        if (sum_num_per_v == NumArray_Duplicate_Index_Value[last_v_idx]) {
          group_size_index_with_same_duplicate_num += 1;
          last_v_idx +=1;
        } else {
          break;
        }
      }

      global_addr_t reduce_global_addr_input    = grad_Y_gathered_global_addr +  counter_history * H_native * len_grad_dtype;
      global_addr_t reduce_global_addr_finished = reduce_global_addr          +  old_v_idx       * H_native * len_grad_dtype;
      dim4 shape_temp       = {.n = 1, .c = group_size_index_with_same_duplicate_num, .h = sum_num_per_v, .w = H_sliced };
      TPUKERNEL_ASSERT_INFO(shape_temp.h < 65535, "sum_num_per_v is too large!");
      dim4 stride_tmp       = {.n = 0, .c = H_native * sum_num_per_v, .h = H_native, .w = 1};
      dim4 shape_collected  = {.n = 1, .c = group_size_index_with_same_duplicate_num, .h = 1, .w = H_sliced};
      dim4 stride_collected = {.n = 0, .c = H_native, .h = H_native, .w = 1};

      if (sum_num_per_v > 1) {
        TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(shape_temp, grad_dtype);
        tpu_gdma_cpy_S2L(reduce_no_parallel_local_addr, reduce_global_addr_input, &shape_temp, NULL, &stride_tmp, grad_dtype);
        ////////////////////////////////
        int using_L2_nodechip =NUM_V_used * H_native * len_grad_dtype + tpu_core_num() *  H_native * NUM_V_used * len_grad_dtype<L2_SRAM_SIZE;
        #if  USING_L2
          TPUKERNEL_ASSERT(USING_L2);
          using_L2_nodechip = NUM_V_used * H_native * len_grad_dtype + tpu_core_num() *  H_native * NUM_V_used * len_grad_dtype<L2_SRAM_SIZE;
        #endif
        int reduce_pattern = AVG_SIM; //ERROR
        //Reduce Full Pattern:
        //1)SUPER_SET: ONly work when grad_out==ones
        //2)L2_PATTERN: Using L2
        //3)AVG_SIM
        if(reduce_pattern==SUPER_SET) {
          //This fast using C_set test only valid when grad_out is all_ones;
          TPUKERNEL_ASSERT(0);
          scalar_t one_ = {.f32 = sum_num_per_v};
          if ( grad_dtype == DT_FP16 ) one_ = tpu_fp_cast ( one_, DT_FP32, DT_FP16, RM_HALF_TO_EVEN );
          tpu_gdma_set_C_system(reduce_global_addr_finished,one_, &shape_collected, &stride_collected, grad_dtype);
        } else if (using_L2_nodechip&&reduce_pattern== L2_PATTERN) {
          //Local_Input_Shape~[1, group_size_index_with_same_duplicate_num,sum_num_per_v,H_slice]
          const int core_idx = tpu_core_index();
          //group_size_index_with_same_duplicate_num <= NUM_V_used
          dim4 shape_local_to_L2_addr  = {.n = 1, .c = group_size_index_with_same_duplicate_num, .h = 1, .w = H_sliced};
          dim4 stride_local_to_L2_addr = {.n = 0, .c = H_sliced, .h = H_sliced, .w = 1};

          int sdma_offset = core_idx *  H_native * V * len_grad_dtype;
          TPUKERNEL_ASSERT(count_numel_dim4(shape_local_to_L2_addr) <  H_native * V);
          global_addr_t reduce_sdma_addr = L2_SRAM_START_ADDR + sdma_offset;
          TPUKERNEL_ASSERT( H_native * V < L2_SRAM_SIZE);
          int psum_op = 1; //GDMA_ARE_PSUM_RW
          int op_code = 1; //GDMA_ARE_ADD
          tpu_sdma_set_C_system(reduce_sdma_addr, zero_, &shape_collected, NULL, grad_dtype);
          tpu_poll();
          for (int id_sdma =0; id_sdma<sum_num_per_v ; id_sdma++) {
            local_addr_t local_to_L2_addr =  reduce_no_parallel_local_addr + id_sdma * H_sliced * len_grad_dtype;
            tpu_gdma_cpy_reduce_L12L2(reduce_sdma_addr, local_to_L2_addr, &shape_local_to_L2_addr, NULL, &stride_local_to_L2_addr, grad_dtype, psum_op, op_code );
          }
          tpu_poll();
          dim4 shape_collected2  = {.n = 1, .c = group_size_index_with_same_duplicate_num, .h = 1, .w = H_sliced};
          dim4 stride_collected2 = {.n = 0, .c = H_native, .h = H_native, .w = 1};
          tpu_sdma_cpy_S2S(reduce_global_addr_finished,reduce_sdma_addr,&shape_collected2, &stride_collected2, NULL, grad_dtype);
          tpu_poll();
        } else if (using_L2_nodechip&&reduce_pattern== AVG_SIM)  {
          dim2 avg_kernel = {sum_num_per_v, 1};
          padding_t avg_pad_local_zero = {0, 0};
          dim2 avg_str_local_one = {1, 1};
          dim2 avg_dil_local_one = {1, 1};
          scalar_t avg_scale;
          avg_scale.f32 = 1.0f;
          // printf("compute  %d ;loop: %d ;core: %d; fvg: %d %d\n", lhistory_id, loop, core, group_size_index_with_same_duplicate_num, sum_num_per_v); lhistory_id+=1;
          tpu_bdc_fp_avg_pool2d(reduce_no_parallel_local_addr,
                                reduce_no_parallel_local_addr,
                                &shape_temp,
                                &avg_kernel,
                                &avg_pad_local_zero,
                                &avg_str_local_one,
                                &avg_dil_local_one,
                                grad_dtype,
                                tpu_cast(avg_scale, grad_dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO));
          tpu_gdma_cpy_L2S(reduce_global_addr_finished, reduce_no_parallel_local_addr, &shape_collected, &stride_collected, NULL, grad_dtype);
        }
      } else {
        tpu_gdma_cpy_S2L(reduce_no_parallel_local_addr, reduce_global_addr_input, &shape_collected, NULL, &stride_collected, grad_dtype);
        tpu_gdma_cpy_L2S(reduce_global_addr_finished, reduce_no_parallel_local_addr, &shape_collected, &stride_collected, NULL, grad_dtype);
      }
      counter_history += sum_num_per_v * group_size_index_with_same_duplicate_num;
      old_v_idx = last_v_idx;
  }

  //NPU_Pattern End; As We asuming  scatter_index_global_flush_64b_aligned_addr can hold all indexes;

  //Step 3: Scatter
  //Function: grad_W [v, h] = temp_reduce_value
  // Input: [v,h] ~[V,H_sliced]; temp_reduce_value ~[NUM_V_used]
  // Output: grad_W~[V',H_sliced] V'~NUM_V_used<=V
  dim4 shape_scatter_L2_to_L1 = {.n = 1, .c = 1, .h = V, .w = H_sliced};
  TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(shape_scatter_L2_to_L1, grad_dtype);

  //Step 3.1 get scatter_index
  //Step_Arch: ARM-<AXI>-GDMA
  //Note: H is sliced, so each core operates sliced shape but strides in native shape.
  //      However,  grad_Y_gathered_global_addr is core-indepedent, so H_native is not considered on its coressponding stride
  dim4 stride_scatter_L2_to_L1 = {.n = 0, .c =0, .h = H_native, .w = 1};
  // Send duplicate_index from CPU to GMEM
  int* tmp2 = NULL;
  tmp2 = (int*) tpu_global_mem_addr(scatter_index_global_flush_64b_aligned_addr);
  for ( int i = 0; i < NUM_V_used; i++ ) {
    tmp2[i] = value_duplicate_index_value_recorder[i];
  }
  tpu_flush_cache(scatter_index_global_flush_64b_aligned_addr, ALIGN(sizeof(int) * NUM_V_used, 64));


  //Step_3.2 Scatter
  //Step_Arch: GMEM-<GDMA>-L1-<GDMA>-GDMA

  //Question: which is faster ? tpu_gdma_h_scatter_S2L or tpu_gdma_h_scatter_S2S
  //\mathsf{output(0, c, index(0, c, h, 0), w) = param(0, c, h, w)} param_h < shape.h !

  //necessary
  tpu_gdma_set_C_local(grad_Y_scatter_local_addr, zero_, &shape_scatter_L2_to_L1, NULL, grad_dtype);
  tpu_gdma_set_C_local(partial_sum_local_addr,    zero_, &shape_scatter_L2_to_L1, NULL, grad_dtype);
  tpu_gdma_h_scatter_S2L ( grad_Y_scatter_local_addr, reduce_global_addr, scatter_index_global_flush_64b_aligned_addr, false, &shape_scatter_L2_to_L1,
      NUM_V_used, NULL, &stride_scatter_L2_to_L1, NULL, grad_dtype );
  tpu_gdma_cpy_S2L(partial_sum_local_addr, grad_Weight_global_addr, &shape_scatter_L2_to_L1,NULL, &stride_scatter_L2_to_L1, grad_dtype);
  tpu_bdc_fp_add(grad_Y_scatter_local_addr, partial_sum_local_addr, grad_Y_scatter_local_addr, &shape_scatter_L2_to_L1, NULL,NULL,NULL, grad_dtype);
  //Note: H(.w-dim) is sliced, so each core operates sliced shape but strides in native shape.
  tpu_gdma_cpy_L2S(grad_Weight_global_addr, grad_Y_scatter_local_addr, &shape_scatter_L2_to_L1, &stride_scatter_L2_to_L1, NULL, grad_dtype);
}


//****************QUICK GUIDE********************//
// Pattern 2
//   2.1  slice H intracores,   BS for-loop parallize,  default
//        [BS, V, H] -> [BS_loop, V, H_sliced]
//   2.2  slice BS intracores,  H  for-loop parallize,  not supported
//   2.3  slice H and BS, nasty not supported
//***********************************************//
//SubPattern 2.1 is default, BS-loop avoid async race
// Function:
//    Input: Index={x{i,j}; |x|<V; i <B; j<S}, grad_W[v, h]=sum(W[X[i,j]]); Every |x[i,j]|==|v|
//    Projection represents one |v| correspondings to Groups of |x[i,j]|
// Key Q: how to reduce x(i,j,loop) from different G_loop to same |v|?
// Solution: BS_loop <-> Groups_loop, |v| <-> {|x[i_0,j_0]|}, ...,{|x[i_p,j_q]|} = {G1,G2,..G'}
// Notice:   BS-loop not applied Index_sorted
// Constraints: V*H_slice* dtype_size <= SG2260_LOCAL_MEM
void nodechip_embedding_backward_multi_core_basic_cell_pattern_two (
  global_addr_t gradout_global_addr,
  global_addr_t index_global_addr,
  global_addr_t grad_input_global_addr,
  global_addr_t sorted_index_value_global_addr,
  global_addr_t sorted_index_index_global_addr,
  global_addr_t scatter_index_global_flush_64b_aligned_addr,
  global_addr_t recollected_global_flush_64b_aligned_addr,
  const dim4    grad_Y_shape, //[B*S, H_sliced]
  const dim4    X_shape,      //[B, S]
  const dim4    grad_W_shape, //[V, H_sliced]
  data_type_t   grad_dtype,
  data_type_t   Index_dtype
) {
   int core_idx = tpu_core_index();
   const int len_grad_dtype = tpu_data_type_size(grad_dtype);
   int H_sliced_real        = 1, H_sliced_avg = 1, H_native = grad_W_shape.w;
   int min_cores_needed     = 1;
   const int V = grad_W_shape.h;

    TPUKERNEL_ASSERT_INFO(grad_Y_shape.w == grad_W_shape.w, "check H of [BS, H] <--> [V ,H] failed");
    int BS_native = count_numel_dim4(X_shape);
    int BS_sliced_real = 1, BS_sliced_avg = 1;
    int excute_pattern_level_sub =  BS_native > H_native ? SUBPATTERN_CODE_LOOP_BS :  SUBPATTERN_CODE_LOOP_H;
    excute_pattern_level_sub = SUBPATTERN_CODE_LOOP_BS;
    TPUKERNEL_ASSERT_INFO(excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS, "SUBPATTERN_CODE_LOOP_H is not supported now!");
    int PartRange_loop = excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS ? BS_native :  H_native;
    int FullRange_loop = PartRange_loop;
    int num_loop_upper     = 0;
    compute_current_slice_info_multi_core_1D(H_native, &H_sliced_real, &H_sliced_avg, &min_cores_needed);

    //When core_num is small&& H is large, [BS,H]->[BS,H_sliced]-> [BS_loop, H_sliced_loop]
    // int poor_core_num_flag = 0;
    // [TO-DO]if V * H_sliced_real * len_grad_dtype >= gen_bank_4_size() H_sliced_real_loop
    while(PartRange_loop * H_sliced_real * len_grad_dtype >= gen_bank_4_size()) { //SG2260_LOCAL_MEM;
      if (( PartRange_loop==1 && V== 1 && H_sliced_real * len_grad_dtype >= gen_bank_4_size()) ||
         2 * H_sliced_real * len_grad_dtype >=  gen_bank_4_size()) {
        // poor_core_num_flag = 1;
        break;
      }
      num_loop_upper += 1;
      PartRange_loop   = DIV_UP(FullRange_loop, num_loop_upper);
    }
    num_loop_upper = num_loop_upper > 0 ? num_loop_upper : 1;

    if (core_idx < min_cores_needed) {
        //[Error] when [B,S,V,H]=[6, 4096,2,12288] num_loop_upper is 2458
        // loop efficiency is low and Function_Recollect_Index cannot utilize same index among loops.

        for (int loop_idx = 0 ; loop_idx < num_loop_upper; loop_idx++) {
          compute_current_loop_info_each_core_1D_with_loop_idx(BS_native, &BS_sliced_real, &BS_sliced_avg, num_loop_upper, loop_idx);

          int  core_dim_sliced_avg =  excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS ? H_sliced_avg   : BS_sliced_avg;
          int  loop_dim_sliced_avg =  excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS ? BS_sliced_avg  : H_sliced_avg;
          int  core_dim_sliced_real = excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS ? H_sliced_real  : BS_sliced_real;
          int  loop_dim_sliced_real = excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS ? BS_sliced_real : H_sliced_real;
          int  core_dim_native =      excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS ? H_native       : BS_native;
          //int  loop_dim_native =      excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS ? BS_native      : H_native;

          global_addr_t grad_Y_addr_sliced                    = gradout_global_addr;
          global_addr_t X_addr_sliced                         = index_global_addr;
          global_addr_t grad_Weight_addr_sliced               = grad_input_global_addr;
          global_addr_t sorted_index_value_global_addr_sliced = sorted_index_value_global_addr;
          global_addr_t sorted_index_index_global_addr_sliced = sorted_index_index_global_addr;
          global_addr_t scatter_index_global_addr_sliced      =  scatter_index_global_flush_64b_aligned_addr;
          global_addr_t recollected_global_addr_sliced        = recollected_global_flush_64b_aligned_addr;
          if (excute_pattern_level_sub == SUBPATTERN_CODE_LOOP_BS) {
            int core_offset          = core_idx * core_dim_sliced_avg * 1 * len_grad_dtype;
            //Grad_W [V,H]->[V, H_sliced]
            //[NO-CONFLICT] core-free, loop-sequential
            grad_Weight_addr_sliced += core_offset;
            //Grad_Y [BS,H]->[BS_loop, H_sliced]
            //[NO-CONFLICT] loop-core independent
            int loop_offset          = loop_idx * loop_dim_sliced_avg * 1 * len_grad_dtype;
            grad_Y_addr_sliced      += loop_offset * core_dim_native +  core_offset;
            //X [B,S]->[BS]->[BS_loop]
            //[Error] [CONFLICT-INFO] loop=0,core=1; loop=0,core =2  if not enough or aligned correctly overlap area on these 4 index buffers will be flushed by mutli-thread.
            X_addr_sliced                         += loop_offset;
            int align_offset_core =  core_idx * length_align_64bit_apporach(BS_native, Index_dtype);
            int align_offset =   loop_idx * length_align_64bit_apporach(loop_dim_sliced_avg, Index_dtype);
            tpu_hau_poll();
            sorted_index_value_global_addr_sliced += align_offset + align_offset_core;
            sorted_index_index_global_addr_sliced += align_offset + align_offset_core;
            scatter_index_global_addr_sliced      += align_offset + align_offset_core;
            recollected_global_addr_sliced        += align_offset + align_offset_core;
            TPUKERNEL_ASSERT_INFO(sorted_index_value_global_addr_sliced % 64==0, "HAU addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
            TPUKERNEL_ASSERT_INFO(sorted_index_index_global_addr_sliced % 64==0, "HAU addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
            TPUKERNEL_ASSERT_INFO(scatter_index_global_addr_sliced % 64==0, "flush addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
            TPUKERNEL_ASSERT_INFO(recollected_global_addr_sliced   % 64==0, "flush addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
          } else { TPUKERNEL_ASSERT_INFO(0, "SUBPATTERN_CODE_LOOP_BS is not supported"); }
          dim4 grad_Y_shape_sliced = {.n=1,.c=1,.h=1,.w= 1};
          dim4 grad_W_shape_sliced = {.n=1,.c=1,.h=1,.w= 1};
          dim4 X_shape_sliced   = {.n=1,.c=1,.h=1,.w= 1};
          //Grad_Y [BS,H]->[BS_loop, H_slice]
          grad_Y_shape_sliced.w =  core_dim_sliced_real;
          grad_Y_shape_sliced.h =  loop_dim_sliced_real;
          //Grad_W [V,H]->[V, H_slice]
          grad_W_shape_sliced.w =  core_dim_sliced_real;
          grad_W_shape_sliced.h =  V;
          //X [B,S]->[BS]->[BS_loop]
          X_shape_sliced.w      = loop_dim_sliced_real;
          TPUKERNEL_ASSERT(X_shape_sliced.h==1);
          //All slices are done before entering into pattern_one
          TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(grad_Y_shape_sliced, grad_dtype);
          TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(grad_W_shape_sliced, grad_dtype);
          nodechip_embedding_backward_multi_core_basic_cell_patttern_one(
            grad_Y_addr_sliced,
            X_addr_sliced,
            grad_Weight_addr_sliced,
            sorted_index_value_global_addr_sliced,
            sorted_index_index_global_addr_sliced,
            scatter_index_global_addr_sliced,
            recollected_global_addr_sliced,
            grad_Y_shape_sliced, //[BS_loop, H_sliced]
            X_shape_sliced,      //[BS_loop]
            grad_W_shape_sliced, //[V, H_sliced]
            core_dim_native,
            loop_idx==0 ? 1: 0,  //first_loop, grad_W is preset to 0 & no fp_add applied
            grad_dtype
            );
        }
      }
}

//Decision Level-1: Top Pattern: in case V is large
//Two Stage Decision Tree: Top-SubPattern
//          Stage 1: In case V*H_sliced*len_grad_dtype >> L2_MEM || GMEM
//                   If so assert, havn't support.
//          Stage 2: In case (V*H_sliced + BS*H_sliced)*len_grad_dtype >> L2_MEM || L1_MEM
//                   If so TOP_CONCURRENCY_H_BS else TOP_CONCURRENCY_H
//****************QUICK GUIDE********************//
//pattern_one->Pattern 1   (V,BS,H) -> (V,BS,H_sliced)
//Pattern_two->Pattern 2.1 (V,BS,H) -> (V,BS_loop, H_sliced)
//Other patterns may be lowering to them
static inline int Decision_Adviosr_Top(
  dim4 grad_W_shape,
  int BS,
  int V,
  data_type_t grad_dtype
) {
  const int len_grad_dtype = tpu_data_type_size(grad_dtype);
  //Decision Stage-1:
  int H_sliced_real = 1, H_sliced_avg = 1, H_native = grad_W_shape.w;
  int V_sliced_real = 1, V_sliced_avg = 1, V_native = grad_W_shape.h;
  int min_cores_needed = 1, min_cores_x =1, min_cores_y= 1;
  //(x,y)  -> (V,H)
  compute_current_slice_info_multi_core_2D_correlated(V_native, H_native,
                                           &V_sliced_real, &H_sliced_real,
                                          &V_sliced_avg, &H_sliced_avg,
                                          &min_cores_needed, &min_cores_x, &min_cores_y, grad_dtype);
  int excute_pattern_level_top = min_cores_x >= 1 ? TOP_CONCURRENCY_H : TOP_CONCURRENCY_V_H;
  compute_current_slice_info_multi_core_1D(H_native, &H_sliced_real, &H_sliced_avg, &min_cores_needed);

  //Basic_Cell: [BS], [BS,H_slice], [V, H_slice]
  int judger_mem = (H_sliced_avg * BS + BS +  V * H_sliced_avg) *  len_grad_dtype;
  //Constraints:
  //           (1) L2 must hold [V,H]+[BS]+[BS,H]
  //           (2) L1 might be not enough
  if (judger_mem < SG2260_GLOBAL_MEM) {
    if (judger_mem >= SG2260_LOCAL_MEM || judger_mem >=  gen_bank_4_size()) {
        excute_pattern_level_top = TOP_CONCURRENCY_H_BS;
    }
  }
  TPUKERNEL_ASSERT_INFO(excute_pattern_level_top==TOP_CONCURRENCY_H_BS || excute_pattern_level_top==TOP_CONCURRENCY_H, "such V is too large on GMEM/L2/L1!");
  return excute_pattern_level_top;
}

void nodechip_embedding_backward_multi_core (
  global_addr_t gradout_global_addr,
  global_addr_t index_global_addr,
  global_addr_t grad_input_global_addr,
  global_addr_t sorted_index_value_global_addr,
  global_addr_t sorted_index_index_global_addr,
  global_addr_t scatter_index_global_flush_64b_aligned_addr,
  global_addr_t recollected_global_flush_64b_aligned_addr,
  const int     *gradout_shape,      //[B*S, H]
  const int     *index_shape,        //[B, S]
  const int     *grad_input_shape,   //[V, H]
  int           grad_output_dim,
  int           idx_dim,
  int           grad_input_dim,
  data_type_t   grad_dtype,
  bool is_index_int64 ) {
  data_type_t Index_dtype = DT_INT32;

  //Prepare multi_core Info:
  const int core_idx = tpu_core_index();
  const int len_grad_dtype = tpu_data_type_size(grad_dtype);

  dim4 X_shape = {.n = 1, .c = 1, .h = 1, .w = index_shape[idx_dim - 1]};
  for ( int i = 0; i < idx_dim - 1; i++ ) {
    X_shape.h *= index_shape[i];
  }

  //In case [B, S, H] - > [B*S, H]
  dim4 grad_W_shape = {.n = 1, .c = 1, .h = 1, .w = grad_input_shape[grad_input_dim - 1]};
  for ( int i = 0; i < grad_input_dim - 1; i++ ) {
    grad_W_shape.h *= grad_input_shape[i];
  }

  //In case [v, H] - > [V, H]
  dim4 grad_Y_shape = {.n = 1, .c = 1, .h = 1, .w = gradout_shape[grad_output_dim - 1]};
  for ( int i = 0; i < grad_output_dim - 1; i++ ) {
    grad_Y_shape.h *= gradout_shape[i];
  }
  const int V = grad_Y_shape.h;
  const int BS = X_shape.h;
  //******************************************************//
  // Global Strategy:  Parallel Computing Against Data Race
  //                   1)Slice intracores is indepedent && never causes data race. ->slice H
  //                   2)Loop-without-unrolling ensures sequence operation order   -> loop BS -> ensure data access conflicts among <BS_loop, H_slice>
  //                   3)Full-V, assuming V * H * len_grad_dtype < L2
  //******************************************************//
  // Check marco above for controller params.
  //******************************************************//
  //Pattern 1： [BS, H, V] -> [BS, H_sliced, V]

  //******************************************************//
  //Pattern 2:  [BS, H, V] -> [BS_cut, H_cut, V]
  //Assumption: For X =  {x{i,j}| |x|<V; i <B; j<S},  grad_W[v, h]=sum(W[X[i,j]]);  every |x[i,j]|==|v|
  //            Projection many x|i,j| -> one |v|, so it's a spare project which V << B*S
  //SubPattern:
  //   2.1  [BS, H]  -> [BS_loop, H_sliced],    default, additional adder-tree is applied to recollect loops of BS
  //   2.2  [BS, H]  -> [BS_sliced, H_loop],    <tpu_all_async> need, not implemented
  //   2.3  [BS, H]  -> [BS_sliced, H_sliced],  <tpu_all_async> need, related slicing, not implemented

  //******************************************************//
  //Pattern 3 [BS, H, V] -> [BS_cut, H_cut, V_cut]
  //Assumption:  (BS+V)_size >> L2 || L2
  //SubPattern:
  //      3.1: Regroup Index by v-projection, i.e., [{x(i,j);|x[i,j]|==|v0|},...,{x(i,j);|x[i,j]|==|v{m}|}],
  //           And ensure each_core sums one compeleted group: {x(i,j);|x[i,j]|==|v0|}
  //           Cons 3.1:  Full_Index_Sorting contrary to motivation Assumption (BS<->L2).
  //           Note: 3,1 EQ.to Pattern 2.1 + regroup + reduce_full_cdma
  //******************************************************//

  //Decision Level-1: Top Pattern: in case V is large

  int excute_pattern_level_top = Decision_Adviosr_Top(grad_W_shape, BS, V, grad_dtype);

  TPUKERNEL_ASSERT(excute_pattern_level_top==TOP_CONCURRENCY_H_BS || excute_pattern_level_top==TOP_CONCURRENCY_H);
  if (excute_pattern_level_top==TOP_CONCURRENCY_H) {
    //Pattern 1: Slice H, easy for slicing 2d-shape on multi-cores
    TPUKERNEL_ASSERT_INFO(grad_Y_shape.w == grad_W_shape.w, "check H of [BS,H]-[V,H]" );
    int H_sliced_real    = 1, H_sliced_avg = 1, H_native = grad_W_shape.w;
    int min_cores_needed = 1;
    compute_current_slice_info_multi_core_1D(H_native, &H_sliced_real, &H_sliced_avg, &min_cores_needed);
    if (core_idx < min_cores_needed) {
      //Grad_Y:(BS,H)->(BS,H_sliced)
      global_addr_t grad_Y_addr_sliced = gradout_global_addr + core_idx * H_sliced_avg * 1 * len_grad_dtype;
      //In pattren 1,（Index: BS)->(BS)
      global_addr_t X_addr_sliced = index_global_addr;

      global_addr_t grad_Weight_addr_sliced = grad_input_global_addr + core_idx * H_sliced_avg * 1 * len_grad_dtype;
      dim4 grad_Y_shape_sliced = grad_Y_shape;
      grad_Y_shape_sliced.w    =  H_sliced_real;
      dim4 grad_W_shape_sliced = grad_W_shape;
      grad_W_shape_sliced.w    =  H_sliced_real;
      //All slices are done before entering into pattern_one
      TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(grad_Y_shape_sliced, grad_dtype);
      TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(grad_W_shape_sliced, grad_dtype);
      nodechip_embedding_backward_multi_core_basic_cell_patttern_one (
        grad_Y_addr_sliced,
        X_addr_sliced,
        grad_Weight_addr_sliced,
        sorted_index_value_global_addr,
        sorted_index_index_global_addr,
        scatter_index_global_flush_64b_aligned_addr,
        recollected_global_flush_64b_aligned_addr,
        grad_Y_shape_sliced, //[B*S, H_sliced]
        X_shape,             //[B, S]
        grad_W_shape_sliced, //[V, H_sliced]
        H_native,
        1, //set first_loop_flag =1 for pattern 2.1
        grad_dtype
        );
    }
  }
  //Pattern 2: LOOP BS
  else if (excute_pattern_level_top== TOP_CONCURRENCY_H_BS) {
    nodechip_embedding_backward_multi_core_basic_cell_pattern_two (
        gradout_global_addr,
        index_global_addr,
        grad_input_global_addr,
        sorted_index_value_global_addr,
        sorted_index_index_global_addr,
        scatter_index_global_flush_64b_aligned_addr,
        recollected_global_flush_64b_aligned_addr,
        grad_Y_shape, //[B*S, H]
        X_shape,      //[B, S]
        grad_W_shape, //[V, H]
        grad_dtype,
        Index_dtype
    );
  }  else {
    //Never slice V;
    TPUKERNEL_ASSERT_INFO(0, "V is too large for GMEM/L2/L1");
  }
}

void tpu_kernel_api_embedding_backward_multi_core ( const void* args ) {
  sg_api_embedding_backward_t *api = ( sg_api_embedding_backward_t * ) args;
  tpu_initialize();
  nodechip_embedding_backward_multi_core (
  api->grad_output_global_addr,
  api->index_global_addr,
  api->grad_input_global_addr,
  api->sorted_index_global_addr,
  api->sorted_index_index_global_addr,
  api->from_index_global_addr,
  api->to_index_global_addr,
  api->grad_output_shape,//grad_Y
  api->index_shape,      //X
  api->grad_input_shape, //Grad_W
  api->grad_output_dim,
  api->index_dim,
  api->grad_input_dim,
  ( data_type_t ) api->grad_output_dtype,
  api->is_index_int64 );
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_embedding_backward_multi_core );
