#include "sg_api_struct.h"
#include "tpu_kernel.h"
#include <stdlib.h>
#include <string.h>


//[Error] will lead you to some improvements.
//CONTROL_CODE
#define SG2260_GLOBAL_MEM 0x100000000

//Slicing Pattern
#define TOP_CONCURRENCY_H    1 //only slice H
#define TOP_CONCURRENCY_H_BS 2 //slice or loop H & NS,
#define TOP_CONCURRENCY_V_H  3 //slice or loop V & H, not implemented

//NPU Computing Pattern
#define NPU_LOOP_V 0                     // {v0,v1..}
#define NPU_RECOLLECT_V_WITH_SAME_SUM 1  // {g0={v0,v1; |v0|==|v1|}}

//Reduce Compute Pattern
#define SUPER_SET 0  //only support for grad_Y==ones
#define L2_PATTERN 1
#define AVG_SIM 2    //default

//Whether using MCU
#define IS_USING_MCU  1

//whether using inplace_scatter on GMEM (if not addtional add is adopted and LMEM split into two)
#define USING_SCATTER_INPLACE_ON_GMEM 1

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

int tpu_kernel_api_embedding_backward ( const void* args ) {
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
  return 0;
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_embedding_backward );


#if 0
static inline int get_L1_TOTAL_MEM() {
  int a1 = tpu_npu_num();
  int a2 = tpu_local_mem_size_per_npu();
  return a1 * a2;
}
// Counting total elemnet of a given shape
// Assume MEM is densely contiguous
static inline int count_numel_dim4(const dim4 shape) {
  return shape.n* shape.c * shape.h * shape.w;
}

//Find a min 64k length apporaching to given length
static inline int length_align_64bit_apporach(int origin_length,const  data_type_t Index_dtype){
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
  TPUKERNEL_ASSERT_INFO(*expected_current_slice <= *expected_avg_slice, "Split  Error!");
  TPUKERNEL_ASSERT_INFO(*expected_current_slice > 0, "Split  Error!");
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
  TPUKERNEL_ASSERT_INFO(current_num_for_current_loop <= avgnum_element_each_loop, "Split Loop Error!");
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
  const int max_dim = total_num_x > total_num_y ? 0 : 1;
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
int check_duplicate_value(const int search_index_value,const  int * data,const int current_length) {
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
int search_duplicate_id(const int search_index_value,const int * data,const int current_length) {
  for(int i=0; i <current_length; i++) {
      if (data[i] == search_index_value)
        return i;
  }
  return -1;
}

//Function: check is Tensor T is larger than L1_MEM
//          If not ,T can be fully hold on L1/L2/GMEM
void TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(const dim4 object,const  data_type_t  dtype) {
    const int l1_mem_sg2260 = get_L1_TOTAL_MEM();
    TPUKERNEL_ASSERT(count_numel_dim4(object) * tpu_data_type_size(dtype) <=l1_mem_sg2260); //only 256KB (8Mb) local_bitwidth is 18
}

//Function: get 4* banksize
//Strategy: If bank_size is not enough, DIV_DOWN
//If scatter_inplace_GMEM, only 1 buffer.
//If scatter_inplace_LMEM, 2 buffers ,as inplace_LMEM is invalid, addtional add_ is adopted
static inline int gen_bank_sizes_split_2_per_npu() {
   const int bank_num = tpu_bank_num();
   const int bank_size_per =  tpu_local_mem_size_per_npu() / tpu_bank_num();
   const int num_used_local_addr = IS_USING_MCU ? 2 : USING_SCATTER_INPLACE_ON_GMEM ? 1: 2;
   //Expect: 16/4= 4; 16/5=3;
   int num_bank_per = DIV_UP(bank_num, num_used_local_addr);
   while (num_bank_per * num_used_local_addr > bank_num) {
     num_bank_per -= 1;
   }
   const int bank_2 = num_bank_per * bank_size_per;
   const int local_mem = tpu_local_mem_size_per_npu();
   TPUKERNEL_ASSERT_INFO(local_mem >= bank_size_per * num_used_local_addr, "[ERROR] Bank_size!");
   return  bank_2;
}

static inline int gen_bank_sizes_split_2_all_npus() {
  const int bank_2 = gen_bank_sizes_split_2_per_npu();
  return bank_2 * tpu_npu_num();
}

static inline int quick_for_loop(int * object, const int start ,const int end) {
  int temp = 0;
  for (int i =start; i <end; i++)
     temp += object[i];
  return temp;
}
//Function:  grad_W[v,h]=sum(W[X[i',j'],h]; grad_Y=W
//HAU_sorted [i',j']->[i,j]: grad_W[v,h]=sum(W[X_sorted[i,j],h]
//X_sorted_duplicate  resorted_if_has_same_duplicate_num
//    grad_Y:W  [ij_0_v_0,ij_1_v_1,ij_2_v_1, ij_3_v_2] -> grad_Y[{ij_0_v_0,ij_3_v_2},{ij_1_v_1,ij_2_v_1}]
//num_repeated  [1,       2,                 1]        ->       [{1,1}              , 2]
//value_repeated[v_0,     v_1                v_2]      ->       [v_0,     ,v_2,       v_1]
static inline void Function_Recollect_Index(
  global_addr_t grad_Y_gathered_global_addr, //[BS_loop， H_sliced_loop]
  global_addr_t recollected_global_flush_64b_aligned_addr,   //[BS]
  int * NumArray_Duplicate_Index_Value,                     //[NUM_V_used]
  int * value_duplicate_index_value_recorder,               //[NUM_V_used]
  int BS_loop,
  int NUM_V_used,
  int H_sliced_loop,
  int H_native,
  data_type_t grad_dtype)
  {
  return;//ERROR
  //[Warning] if BS_Loop is too small, regroup has no use, but H is greatly parallel this time;
  const int len_grad_dtype = tpu_data_type_size(grad_dtype);
  int *tmp_collected = (int*) malloc(BS_loop * H_sliced_loop * sizeof(int));
  int *tmp_collected_native = NULL;
  tmp_collected_native = (int*)tpu_global_mem_addr(grad_Y_gathered_global_addr);
  for(int   i=0; i < BS_loop;      i++)
    for(int j=0; j < H_sliced_loop;j++)  {
     tmp_collected[i * H_sliced_loop + j] = tmp_collected_native[i * H_native + j] ;
  }
  int *tmp_collected_copy = (int*) malloc(BS_loop*H_sliced_loop*sizeof(int));
  const int min_while_controller = BS_loop < NUM_V_used ? BS_loop : NUM_V_used;
  int flag_collect = 1;
  if(0){
  for (int k_null=0;k_null<BS_loop;k_null++)
  for (int i_num_recollect=0;i_num_recollect<BS_loop;i_num_recollect++) {
    int j_num_recollect = i_num_recollect+1;
    while(j_num_recollect<min_while_controller) {
        int duplicate_value_i = NumArray_Duplicate_Index_Value[i_num_recollect];
        int duplicate_value_j = NumArray_Duplicate_Index_Value[j_num_recollect];
        //TODO we can further reduce by ++j_num_recollect if NumArray_Duplicate_Index_Value[j_num_recollect]==NumArray_Duplicate_Index_Value[j_num_recollect+1]
        if (duplicate_value_i > duplicate_value_j) {
              memset(tmp_collected_copy,0, BS_loop * H_sliced_loop *sizeof(int) );
              flag_collect = 1;
              //reorder index
              //[FROM]seq_0, tmp_collected[i_start:i_end;H), seq_1,  tmp_collected[j_start:j_end;H), seq_2 ->
              //[TO]  seq_0, tmp_collected[j_start:j_end;H), seq_1,  tmp_collected[i_start:i_end;H), seq_2
              //[start, end)
              int i_start = quick_for_loop(NumArray_Duplicate_Index_Value, 0 , i_num_recollect);
              int i_end = i_start+duplicate_value_i;

              int j_start = quick_for_loop(NumArray_Duplicate_Index_Value, 0 , j_num_recollect);
              int j_end = j_start+duplicate_value_j;
              //[FROM]seq_0:tmp_collected[0, i_start;H)
              //[TO]  seq_0:tmp_collected[0, i_start;H)
              int size = i_start;
              int size_history =0;
              if(size)
                for (int pp=0;pp<size;pp++) { for (int k_H = 0; k_H < H_sliced_loop; k_H++)
                  tmp_collected_copy[pp * H_sliced_loop + k_H] = tmp_collected[pp * H_sliced_loop + k_H];
                }
                // memcpy(tmp_collected_copy, tmp_collected, size);
              size_history += size;

              //[FROM]tmp_collected[i_start:i_end;H) ->
              //[TO]  tmp_collected[j_start:j_end;H)
              size = duplicate_value_j;
              if(size)
                for (int pp=0;pp<size;pp++) { for (int k_H = 0; k_H < H_sliced_loop; k_H++)
                  tmp_collected_copy[(pp+ size_history)*H_sliced_loop+ k_H] = tmp_collected[(j_start+pp)*H_sliced_loop+ k_H];
                }
                // memcpy(tmp_collected_copy + size_history, tmp_collected + j_start *sizeof(int), size);
              size_history += size;

              //[FROM]seq_1:tmp_collected[i_end:j_start;H) ->
              //[TO]  seq_1:tmp_collected[j_end:i_start;H)
              size = j_start-i_end;
              if(size)
                for (int pp=0;pp<size;pp++) { for (int k_H = 0; k_H < H_sliced_loop; k_H++)
                  tmp_collected_copy[(pp+size_history)*H_sliced_loop+ k_H] = tmp_collected[(i_end+pp)*H_sliced_loop+ k_H];
                }
                // memcpy(tmp_collected_copy + size_history, tmp_collected + i_end *sizeof(int), size);
              size_history += size;


              //[FROM]tmp_collected[j_start:j_end;H) ->
              //[TO]  tmp_collected[i_start:i_end;H)
              size = duplicate_value_i;
              if(size)
                for (int pp=0;pp<size;pp++) { for (int k_H = 0; k_H < H_sliced_loop; k_H++)
                  tmp_collected_copy[(pp+size_history)*H_sliced_loop+ k_H] = tmp_collected[(i_start+pp)*H_sliced_loop+ k_H];
                }
                // memcpy(tmp_collected_copy + size_history, tmp_collected + i_start *sizeof(int), size);
              size_history += size;

              //[FROM]seq_2:tmp_collected[j_end:BS_loop;H) ->
              //[TO]  seq_2:tmp_collected[i_end:BS_loop;H)
              size = (BS_loop-j_end);
              if(size)
                for (int pp=0;pp<size;pp++) { for (int k_H = 0; k_H < H_sliced_loop; k_H++)
                  tmp_collected_copy[(pp+size_history)*H_sliced_loop+ k_H] = tmp_collected[(j_end+pp)*H_sliced_loop+ k_H];
              }
                // memcpy(tmp_collected_copy + size_history, tmp_collected + j_end *sizeof(int), size);
              size_history += size;
              for (int k_H = 0; k_H < H_sliced_loop; k_H++)
              for (int pp=0;pp<BS_loop;pp++) {
                  tmp_collected[pp*H_sliced_loop+ k_H] = tmp_collected_copy[pp*H_sliced_loop+ k_H];
              }
              // memcpy(tmp_collected, tmp_collected_copy, H_sliced_loop * BS_loop *sizeof(int));
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
  }
  if (flag_collect) {
    for ( int i = 0; i < H_sliced_loop; i++ ) {
      int* tmp2_3 = NULL;
      tmp2_3 = (int*)tpu_global_mem_addr(recollected_global_flush_64b_aligned_addr);//[BS_LOOP]
      for(int j = 0; j < BS_loop; j++)  {
          tmp2_3[j] =  tmp_collected[j*H_sliced_loop +i];
      }
      TPUKERNEL_ASSERT_INFO(recollected_global_flush_64b_aligned_addr%64==0, "NEED aligned to 64bit");
      tpu_flush_cache ( recollected_global_flush_64b_aligned_addr, ALIGN ( sizeof ( int ) * BS_loop, 64 ) );
      const dim4 shape_tmp_recollected = {.n=1,.c=1,.h=BS_loop,.w=1};
      const dim4 stride_tmp_recollected = {.n=1,.c=1,.h=H_native,.w=1};
      tpu_gdma_cpy_S2S(grad_Y_gathered_global_addr+ i * len_grad_dtype, recollected_global_flush_64b_aligned_addr, &shape_tmp_recollected, &stride_tmp_recollected, NULL, grad_dtype);
    }
  }
  free(tmp_collected_copy);
  free(tmp_collected);
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
// grad_W[v, h] = SUM_{i,j}[grad_Y[(i,j)|X[i,j]==v, h]
//**********************************************************//

//Chip Arch
//Concurrency                                     Shared               Shared
//                  Near-NPU     on-chip          Near-Cores           Off-chip
//device            Local        sDMA             L2                   GDMA
//Arch_solution_1   V              X                X                    V
//Arch_solution_2   V              V                X                    V
//Arch_solution_2   V              V                V                    V

void nodechip_embedding_backward_multi_core_basic_cell_patttern_one_intercore(
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
  const data_type_t   grad_dtype) {
  //Step 0: check and initalize
  // TPUKERNEL_ASSERT(USING_L2);
  // TPUKERNEL_ASSERT(IN_L2_SRAM(grad_Weight_global_addr));
  const scalar_t zero_ = {.u32 = 0};
  const int len_grad_dtype = tpu_data_type_size(grad_dtype);
  const int NUM_full_Index = count_numel_dim4(X_shape);
  const int H_sliced = grad_Y_shape.w;
  const int V = grad_W_shape.h; //different from NUM_V_used
  const int BS = grad_Y_shape.h;
  TPUKERNEL_ASSERT_INFO(BS == NUM_full_Index, "[ERROR] BS CHECK FAILED");
  const int local_offset_1 = BS * H_sliced * len_grad_dtype; //tpu_aligned_feature_size (1, H_sliced, grad_dtype );
  const int local_offset_2 = V  * H_sliced * len_grad_dtype; // tpu_aligned_feature_size (1, H_sliced, grad_dtype ); //sum_num_per_v<=V
  const int bank_2_size = gen_bank_sizes_split_2_per_npu();
  const local_addr_t grad_Y_gathered_local_addr    = 0;
  const local_addr_t reduce_no_parallel_local_addr = 0;
  const local_addr_t grad_Y_scatter_local_addr     = 1 * bank_2_size;
  const local_addr_t partial_sum_local_addr        = 0;
  TPUKERNEL_ASSERT_INFO(local_offset_1 <= bank_2_size, "BS is too large!");
  TPUKERNEL_ASSERT_INFO(local_offset_2 <= bank_2_size, "V is too large or not sparse enough!");
  TPUKERNEL_ASSERT_INFO(partial_sum_local_addr+local_offset_2 <= (u32)tpu_local_mem_size_per_npu(), "BS_Loop is too large");
  TPUKERNEL_ASSERT_INFO(grad_Y_scatter_local_addr+local_offset_1 <= (u32)tpu_local_mem_size_per_npu(), "BS_Loop is too large");
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
  // Step 1.0 Sorting full-Index(X)
  // Step_Arch: HAU-<GDMA>-GMEM
  tpu_hau_sort_natural_index ( sorted_index_value_global_addr,
                               sorted_index_index_global_addr,
                               X_global_addr,
                               BS, BS, false, DT_INT32 );
  tpu_hau_poll();
  tpu_invalidate_cache ( sorted_index_value_global_addr, ALIGN ( BS * sizeof ( int ), 64 ) );
  tpu_invalidate_cache ( sorted_index_index_global_addr, ALIGN ( BS * sizeof ( int ), 64 ) );

  //Step 1.1: Gather
  //Step_Arch: GMEM-<GDMA>-L1-<GDMA>-GMEM
  //  Function: z =  grad_Y[X[i,j], h]; z = grad_W[v, h]; v is free for every |v|==|x[x,j]|
  //  Input: Index_sorted_by_Index_value~[B*S]
  //  Output: Output[index_sorted]~[B*S,H_sliced]
  const dim4 shape_gather_L2_to_L1 = {.n = 1, .c = 1, .h = BS, .w = H_sliced};
  TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(shape_gather_L2_to_L1, grad_dtype);
  //Note: H(.w-dim) is sliced, so each core operates sliced shape but strides in native shape.
  const dim4 stride_gather_L2_to_L1 = {.n = 0, .c = 0, .h = H_native, .w = 1};
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
  int* value_duplicate_index_value_recorder        = (int*)malloc(BS * sizeof(int));
  int* NumArray_Duplicate_Index_Value              = (int*)malloc(BS * sizeof(int));

  //[D2S]Send duplicate from  Sys to CPU
  tmp1 = ( int* ) tpu_global_mem_addr ( sorted_index_value_global_addr );
  // NUM_V_used : <=|V|. represents nums of V used
  int NUM_V_used = 0;
  //[CPU]Searching duplicate index from sorted Index
  for (int i = 0; i < BS; i++) {
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
  //CONTROL_CODE ERROR
  NPU_Utils_Pattern = NPU_RECOLLECT_V_WITH_SAME_SUM;
  if (NPU_Utils_Pattern==NPU_RECOLLECT_V_WITH_SAME_SUM) {
    Function_Recollect_Index(
      grad_Y_gathered_global_addr,
      recollected_global_flush_64b_aligned_addr,
      NumArray_Duplicate_Index_Value,
      value_duplicate_index_value_recorder,
      BS,
      NUM_V_used,
      H_sliced,
      H_native,
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
      const dim4 shape_temp       = {.n = 1, .c = group_size_index_with_same_duplicate_num, .h = sum_num_per_v, .w = H_sliced };
      TPUKERNEL_ASSERT_INFO(shape_temp.h < 65535, "sum_num_per_v is too large!");
      const dim4 stride_tmp       = {.n = 0, .c = H_native * sum_num_per_v, .h = H_native, .w = 1};
      const dim4 shape_collected  = {.n = 1, .c = group_size_index_with_same_duplicate_num, .h = 1, .w = H_sliced};
      const dim4 stride_collected = {.n = 0, .c = H_native, .h = H_native, .w = 1};

      if (sum_num_per_v > 1) {
        TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(shape_temp, grad_dtype);
        tpu_gdma_cpy_S2L(reduce_no_parallel_local_addr, reduce_global_addr_input, &shape_temp, NULL, &stride_tmp, grad_dtype);
        ////////////////////////////////
        int using_L2_nodechip =NUM_V_used * H_native * len_grad_dtype + tpu_core_num() *  H_native * NUM_V_used * len_grad_dtype<L2_SRAM_SIZE;
        #if  USING_L2
          TPUKERNEL_ASSERT(USING_L2);
          using_L2_nodechip = NUM_V_used * H_native * len_grad_dtype + tpu_core_num() *  H_native * NUM_V_used * len_grad_dtype<L2_SRAM_SIZE;
        #endif
        const int reduce_pattern = AVG_SIM; //ERROR
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
          const dim4 shape_local_to_L2_addr  = {.n = 1, .c = group_size_index_with_same_duplicate_num, .h = 1, .w = H_sliced};
          const dim4 stride_local_to_L2_addr = {.n = 0, .c = H_sliced, .h = H_sliced, .w = 1};

          const int sdma_offset = core_idx *  H_native * V * len_grad_dtype;
          TPUKERNEL_ASSERT(count_numel_dim4(shape_local_to_L2_addr) <  H_native * V);
          global_addr_t reduce_sdma_addr = L2_SRAM_START_ADDR + sdma_offset;
          TPUKERNEL_ASSERT( H_native * V < L2_SRAM_SIZE);
          const int psum_op = 1; //GDMA_ARE_PSUM_RW
          const int op_code = 1; //GDMA_ARE_ADD
          tpu_sdma_set_C_system(reduce_sdma_addr, zero_, &shape_collected, NULL, grad_dtype);
          tpu_poll();
          for (int id_sdma =0; id_sdma<sum_num_per_v ; id_sdma++) {
            local_addr_t local_to_L2_addr =  reduce_no_parallel_local_addr + id_sdma * H_sliced * len_grad_dtype;
            tpu_gdma_cpy_reduce_L12L2(reduce_sdma_addr, local_to_L2_addr, &shape_local_to_L2_addr, NULL, &stride_local_to_L2_addr, grad_dtype, psum_op, op_code );
          }
          tpu_poll();
          const dim4 shape_collected2  = {.n = 1, .c = group_size_index_with_same_duplicate_num, .h = 1, .w = H_sliced};
          const dim4 stride_collected2 = {.n = 0, .c = H_native, .h = H_native, .w = 1};
          tpu_sdma_cpy_S2S(reduce_global_addr_finished,reduce_sdma_addr,&shape_collected2, &stride_collected2, NULL, grad_dtype);
          tpu_poll();
        } else if (using_L2_nodechip&&reduce_pattern== AVG_SIM)  {
          const dim2 avg_kernel = {sum_num_per_v, 1};
          const padding_t avg_pad_local_zero = {0, 0};
          const dim2 avg_str_local_one = {1, 1};
          const dim2 avg_dil_local_one = {1, 1};
          const scalar_t avg_scale = {.f32 = 1.0f};
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
  const dim4 shape_scatter_L2_to_L1 = {.n = 1, .c = 1, .h = V, .w = H_sliced};
  TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(shape_scatter_L2_to_L1, grad_dtype);

  //Step 3.1 get scatter_index
  //Step_Arch: ARM-<AXI>-GDMA
  //Note: H is sliced, so each core operates sliced shape but strides in native shape.
  //      However,  grad_Y_gathered_global_addr is core-indepedent, so H_native is not considered on its coressponding stride
  const dim4 stride_scatter_L2_to_L1 = {.n = 0, .c =0, .h = H_native, .w = 1};
  // Send duplicate_index from CPU to GMEM
  int* tmp2 = NULL;
  tmp2 = (int*) tpu_global_mem_addr(scatter_index_global_flush_64b_aligned_addr);
  for ( int i = 0; i < NUM_V_used; i++ ) {
    tmp2[i] = value_duplicate_index_value_recorder[i];
  }
  // TPUKERNEL_ASSERT_INFO(NUM_V_used <= BS_native, "[ERROR]NUM_V_used>BS, as  scatter_index_global_flush_64b_aligned_addr created by BS");
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

//[Notice] This FUnction is purely using TPU inst and no slicing operation.
void nodechip_embedding_backward_multi_core_basic_cell_patttern_one_intercore_no_MCU(
  global_addr_t grad_Y_global_addr,
  global_addr_t X_global_addr,
  global_addr_t grad_Weight_global_addr,
  const dim4    grad_Y_shape, //[B*S, H_sliced]
  const dim4    X_shape,      //[B, S], each value in [0,V)
  const dim4    grad_W_shape, //[V, H_sliced]
  const int     H_native,
  const int     first_loop_flag,
  const data_type_t   grad_dtype) {
  // //Step 0: check and initalize
  // // TPUKERNEL_ASSERT(USING_L2);
  // // TPUKERNEL_ASSERT(IN_L2_SRAM(grad_Weight_global_addr));
  // const scalar_t zero_ = {.u32 = 0};
  // const int len_grad_dtype = tpu_data_type_size(grad_dtype);
  // const int NUM_full_Index = count_numel_dim4(X_shape);
  // const int H_sliced = grad_Y_shape.w;
  // const int V = grad_W_shape.h; //different from NUM_V_used
  // const int BS = grad_Y_shape.h;
  // TPUKERNEL_ASSERT_INFO(BS == NUM_full_Index, "[ERROR] BS CHECK FAILED");
  // const int local_offset_1 = BS * H_sliced * len_grad_dtype; //tpu_aligned_feature_size (1, H_sliced, grad_dtype );
  // //Shape Checker
  // TPUKERNEL_ASSERT(grad_Y_shape.w == grad_W_shape.w);
  // const dim4 shape_gather_L2_to_L1 = {.n = 1, .c = 1, .h = BS, .w = H_sliced};
  // TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(shape_gather_L2_to_L1, grad_dtype);
  // //[Funny Simplified]
  // //Target: grad_W[v, h] = SUM_{i,j}[grad_Y[(i,j)|X[i,j]==v, h]
  // //Without MCU, simplified procedures as follows:
  // //0) regroup x[i,j]==v  => HAU(3) =>regroup(2.1) =>Reduce(4) =>scatter(2.2)
  // //1) MEM simplify 4 global buffer (hau_index; hau_value; scatter_index; index_for_duplicated_v_projection)
  // //2) Simplify 2 MCU; 1st NPU_Utils_Pattern; 2nd scatter_index_global_flush_64b_aligned_addr
  // //                2.1)NPU_Utils_Pattern: regroup v into #k|x[i,j]==v_k, i.e., NUM_V_used, so that each group computes 1,rather than #(i,j)
  // //                2.2)scatter_index_global_flush_64b_aligned_addr exists because sort-trans; 2.1)regroup need to sort 3)Index and so thus v_sorted need to reflush as x[i,j] <-> v.
  // //3) simplify 1 HAU
  // //4) simplify Reduce because scatter_inplace_add
  // //5) TransLine Pattern changed:
  // //   5.1)With MCU : 3)HAU-Gather: {GMEM-L1-GMEM} - 2.1:{GMEM-MCU-GMEM} - 4)Reduce:{GMEM-AVG-GMEM} - 2.2:{GMEM-MCU-GMEM} -  Scatter - Add_BS_loop
  // //   5.2)No MCU   : Scatter - Add_BS_loop
  // //6) TransLine Pattern changed:
  // //   0) started with [BS_loop,H_sliced_loop, V], as they use same slice-loop pattern; H, is ignored as it's intracore parallel.
  // //   Usually BS >> V, so as BS_loop >> |v| >>#k = NUM_V_used
  // //   6.1)With MCU : T_{others} + T_Scatter{#k times, where {k|x[i,j]==v_k}} + Add_BS_loop(1 time)
  // //   6.2)No MCU   : T_Scatter(BS_loop times) + Add_BS_loop(1 time)
  // //   Expect T_{others} >  (BS_loop - #k) T_scatter(1 times), which means 6.2) is better, ususally it's true

  // //Step 1: Scatter
  // //Function: grad_W [v, h] = temp_reduce_value
  // // Input: [v,h] ~[V,H_sliced];
  // // Output: grad_W~[V',H_sliced] V'~NUM_V_used<=V;  V-NUM_V_used is preset as 0
  // const dim4 stride_scatter_L2_to_L1 = {.n = 0, .c =0, .h = H_native, .w = 1};
  // const local_addr_t grad_Y_scatter_local_addr    = 0;
  // //Step_1.1 Scatter
  // //Step_Arch: GMEM-<GDMA>-L1-<GDMA>-GDMA
  // //necessary, preset V-NUM_V_used as 0
  // if(! USING_SCATTER_INPLACE_ON_GMEM) {
  //   const int local_offset_2 = V  * H_sliced * len_grad_dtype; // tpu_aligned_feature_size (1, H_sliced, grad_dtype ); //sum_num_per_v<=V
  //   const int bank_2_size = gen_bank_sizes_split_2_per_npu(); //using all LMEM of per NPU
  //   const local_addr_t partial_sum_local_addr        = 1 * bank_2_size;
  //   TPUKERNEL_ASSERT_INFO(local_offset_1 <= bank_2_size, "BS is too large!");
  //   TPUKERNEL_ASSERT_INFO(local_offset_2 <= bank_2_size, "V is too large or not sparse enough!");
  //   TPUKERNEL_ASSERT_INFO(partial_sum_local_addr+local_offset_2 <= (u32)tpu_local_mem_size_per_npu(), "BS_Loop is too large");
  //   const dim4 shape_scatter_L2_to_L1 = {.n = 1, .c = 1, .h = V, .w = H_sliced};
  //   TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(shape_scatter_L2_to_L1, grad_dtype);
  //   tpu_gdma_set_C_local(grad_Y_scatter_local_addr, zero_, &shape_scatter_L2_to_L1, NULL, grad_dtype);
  //   tpu_gdma_set_C_local(partial_sum_local_addr,     zero_, &shape_scatter_L2_to_L1, NULL, grad_dtype);
  //   tpu_gdma_h_scatter_S2L_with_inplace_add(grad_Y_scatter_local_addr, grad_Y_global_addr, X_global_addr, false, &shape_scatter_L2_to_L1,
  //       BS, NULL, &stride_scatter_L2_to_L1, NULL, 1, grad_dtype );
  //   // Note: H(.w-dim) is sliced, so each core operates sliced shape but strides in native shape.
  //   //[NOTE]scatter_S2L expects y[x] = W,  but scatter_L2S or scatter_S2S expects y[x] +=W
  //   //inplace_add only valid on GMEM ,i.e., S2L valid &&L2S not valid!
  //   //So that S2L scatter needs additional adder.
  //   tpu_gdma_cpy_S2L(partial_sum_local_addr,  grad_Weight_global_addr, &shape_scatter_L2_to_L1,   NULL, &stride_scatter_L2_to_L1, grad_dtype);
  //   tpu_bdc_fp_add(grad_Y_scatter_local_addr, partial_sum_local_addr, grad_Y_scatter_local_addr, &shape_scatter_L2_to_L1, NULL,NULL,NULL, grad_dtype);
  //   // Note: H(.w-dim) is sliced, so each core operates sliced shape but strides in native shape.
  //   tpu_gdma_cpy_L2S(grad_Weight_global_addr, grad_Y_scatter_local_addr, &shape_scatter_L2_to_L1, &stride_scatter_L2_to_L1, NULL, grad_dtype);
  // } else {
  //   tpu_gdma_set_C_local(grad_Y_scatter_local_addr, zero_, &shape_gather_L2_to_L1, NULL, grad_dtype);
  //   //[NOTE]scatter_S2L expects y[x] = W,  but scatter_L2S or scatter_S2S expects y[x] +=W
  //   //inplace_add only valid on GMEM ,i.e., S2L valid &&L2S not valid!
  //   tpu_gdma_cpy_S2L(grad_Y_scatter_local_addr, grad_Y_global_addr, &shape_gather_L2_to_L1, NULL, &stride_scatter_L2_to_L1, grad_dtype);
  //   tpu_gdma_h_scatter_L2S_with_inplace_add(grad_Weight_global_addr, grad_Y_scatter_local_addr, X_global_addr, false, &shape_gather_L2_to_L1,
  //         BS, &stride_scatter_L2_to_L1, NULL, NULL, 1, grad_dtype );
  // }
}

//Slicing MEM: Total LMEM  ->    LMEM of per NPU
//Slice Info [BS_loop, H_sliced_loop, V] rebase on [NPU_NUM, EU_NUM] style
void nodechip_embedding_backward_multi_core_basic_cell_patttern_one_intracore(
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
  const data_type_t   grad_dtype,
  const data_type_t Index_dtype,
  const int     using_MCU_flag) {
  const int len_grad_dtype = tpu_data_type_size(grad_dtype);
  const int NUM_full_Index = count_numel_dim4(X_shape);
  const int H_sliced = grad_Y_shape.w;
  const int V = grad_W_shape.h; //different from NUM_V_used
  const int BS = grad_Y_shape.h;
  TPUKERNEL_ASSERT_INFO(BS == NUM_full_Index, "[ERROR] BS CHECK FAILED");
  const int bank_2_size = gen_bank_sizes_split_2_per_npu();

  // #Target:BS_secs_real
  // #Function:
  // #BS -> NUM_NPU *BS_slice_sophon + BS_secs_left
  // #BS_secs_real = MIN(NUM_NPU, BS_secs_left)
  // #Fact:  #num_loop_upper==BS_slice_sophon+1
  int num_loop_upper_BS = DIV_UP(BS, NPU_NUM);
  for (int idx_BS_loop=0; idx_BS_loop<num_loop_upper_BS; idx_BS_loop++) {
    int BS_sliced_avg= MIN(BS, NPU_NUM);
    int BS_sliced_real= idx_BS_loop == num_loop_upper_BS - 1 ? MAX(1, BS - idx_BS_loop * BS_sliced_avg) : BS_sliced_avg;
    TPUKERNEL_ASSERT(BS_sliced_avg <= NPU_NUM && BS_sliced_real >= 1);

    // #Target:H_sliced_real
    // #Function: H_sliced -> (num_loop_upper -1) *H_sliced_avg + H_sliced_left
    // #Constraints: If H_sliced_avg satisfied, so as H_sliced_real
    // # bank_2 >= len_grad_dtype * BS_secs_real * H_sliced_avg >= k NPU_NUM* EU_NUM = len_grad_dtype * BS_secs_real * H_sliced_real
    int  aligned_per_npu_mem_H_sliced = ALIGN_DOWN(bank_2_size / DIV_UP(BS_sliced_avg, NPU_NUM), EU_NUM);
    int  H_sliced_avg_sophon = MIN(H_sliced ,aligned_per_npu_mem_H_sliced / BS_sliced_avg / len_grad_dtype);
    int  num_loop_upper_H_sliced = DIV_UP(H_sliced, H_sliced_avg_sophon);
    TPUKERNEL_ASSERT(bank_2_size >= len_grad_dtype * BS_sliced_real * H_sliced_avg_sophon);
    TPUKERNEL_ASSERT(bank_2_size >= len_grad_dtype * BS_sliced_avg  * H_sliced_avg_sophon);
    // #Function if BS_secs_real > V : naturally satisified skip V; else cut H_slice again base on V
    if (V > BS_sliced_avg) {
      aligned_per_npu_mem_H_sliced = ALIGN_DOWN(bank_2_size / DIV_UP(V, NPU_NUM), EU_NUM);
      H_sliced_avg_sophon = MIN(H_sliced ,aligned_per_npu_mem_H_sliced / V / len_grad_dtype);
      num_loop_upper_H_sliced = DIV_UP(H_sliced, H_sliced_avg_sophon);
      TPUKERNEL_ASSERT(bank_2_size >= len_grad_dtype * BS_sliced_real * H_sliced_avg_sophon);
    }
    TPUKERNEL_ASSERT(bank_2_size >= len_grad_dtype * V * H_sliced_avg_sophon);
    for (int idx_H_sliced=0; idx_H_sliced < num_loop_upper_H_sliced; idx_H_sliced++) {
      int H_sliced_avg = H_sliced_avg_sophon;
      int H_sliced_real= idx_H_sliced == num_loop_upper_H_sliced - 1 ? MAX(1, H_sliced - idx_H_sliced * H_sliced_avg) : H_sliced_avg;
      //Parallel Relationship:
      //Sequential-Loop (idx_BS_loop,idx_H_sliced)
      //[Note] core_idx is not used here for H_sliced!
      global_addr_t grad_Y_addr_sliced                    = grad_Y_global_addr;
      global_addr_t X_addr_sliced                         = X_global_addr;
      global_addr_t grad_Weight_addr_sliced               = grad_Weight_global_addr;
      global_addr_t sorted_index_value_global_addr_sliced = sorted_index_value_global_addr;
      global_addr_t sorted_index_index_global_addr_sliced = sorted_index_index_global_addr;
      global_addr_t scatter_index_global_addr_sliced      =  scatter_index_global_flush_64b_aligned_addr;
      global_addr_t recollected_global_addr_sliced        = recollected_global_flush_64b_aligned_addr;
      const int offset_H                = idx_H_sliced * H_sliced_avg * 1 * len_grad_dtype;
      //Grad_W [V,H_sliced]->[V, H_sliced_loop]->[V, H_sliced_loop__]
      grad_Weight_addr_sliced += offset_H;
      //Grad_Y [BS,H]->[BS_loop, H_sliced_loop]->[BS_loop__, H_sliced_loop__]
      const int offset_BS_loop = idx_BS_loop    * BS_sliced_avg * 1 * len_grad_dtype;
      grad_Y_addr_sliced      += offset_BS_loop * H_native +  offset_H;
      //X [B,S]->[BS]->[BS_loop]->[BS_loop_]
      //[Note] Index(X) is free with idx_H_sliced_loop
      //[Error] [CONFLICT-INFO] loop=0,core=1; loop=0,core =2  if not enough or aligned correctly overlap area on these 4 index buffers will be flushed by mutli-thread.
      X_addr_sliced           += offset_BS_loop;
      const int align_offset_loop    =  idx_BS_loop * length_align_64bit_apporach(BS_sliced_avg, Index_dtype);
      sorted_index_value_global_addr_sliced += align_offset_loop;
      sorted_index_index_global_addr_sliced += align_offset_loop;
      scatter_index_global_addr_sliced      += align_offset_loop;
      recollected_global_addr_sliced        += align_offset_loop;
      TPUKERNEL_ASSERT_INFO(sorted_index_value_global_addr_sliced % 64==0, "HAU addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
      TPUKERNEL_ASSERT_INFO(sorted_index_index_global_addr_sliced % 64==0, "HAU addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
      TPUKERNEL_ASSERT_INFO(scatter_index_global_addr_sliced % 64==0, "flush addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
      TPUKERNEL_ASSERT_INFO(recollected_global_addr_sliced   % 64==0, "flush addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
      const dim4 grad_Y_shape_sliced = {.n=1,.c=1,.h=BS_sliced_real,.w= H_sliced_real};
      const dim4 grad_W_shape_sliced = {.n=1,.c=1,.h=V,             .w= H_sliced_real};
      const dim4 X_shape_sliced   = {.n=1,.c=1,.h=1,.w= BS_sliced_real};
      //All slices are done before entering into pattern_one
      TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(grad_Y_shape_sliced, grad_dtype);
      TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(grad_W_shape_sliced, grad_dtype);
      if (!using_MCU_flag) {
        nodechip_embedding_backward_multi_core_basic_cell_patttern_one_intercore_no_MCU(
                grad_Y_addr_sliced,
                X_addr_sliced,
                grad_Weight_addr_sliced,
                grad_Y_shape_sliced, //[BS_loop, H_sliced_loop]
                X_shape_sliced,      //[BS_loop]
                grad_W_shape_sliced, //[V, H_sliced_loop]
                H_native,
                1,
                grad_dtype
                );
      } else {
        nodechip_embedding_backward_multi_core_basic_cell_patttern_one_intercore(
                grad_Y_addr_sliced,
                X_addr_sliced,
                grad_Weight_addr_sliced,
                sorted_index_value_global_addr_sliced,
                sorted_index_index_global_addr_sliced,
                scatter_index_global_addr_sliced,
                recollected_global_addr_sliced,
                grad_Y_shape_sliced, //[BS_loop, H_sliced_loop]
                X_shape_sliced,      //[BS_loop]
                grad_W_shape_sliced, //[V, H_sliced_loop]
                H_native,
                1,
                grad_dtype
                );
      }
    }
   }
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
//Slicing Mem: Total GMEM  ->   Total LMEM
// Key Q: how to reduce x(i,j,loop) from different G_loop to same |v|?
// Solution: BS_loop <-> Groups_loop, |v| <-> {|x[i_0,j_0]|}, ...,{|x[i_p,j_q]|} = {G1,G2,..G'}
//Strategy: AS usually BS > V for LLM,   adopt (a) as (b)-(4) need to reloop B for larger slice-shape
// Notice:   BS-loop not applied Index_sorted && V/BS never intracore sliced
// (a) 1)H_slice; 2)ensue (V,H_Sliced_loop)  < gen_bank_sizes_split_2_all_npus()  3) (only when BS > V) loop B ensure (BS_loop,H_Sliced_loop) < gen_bank_sizes_split_2_all_npus()
// (b) 1)H_slice; 2)ensue (BS_loop,H_Sliced) < gen_bank_sizes_split_2_all_npus()  3) (only when V > BS_loop) loop H ensure (V,H_Sliced_loop) < gen_bank_sizes_split_2_all_npus() 4) (only when V > BS_loop) reloop B find a larger (BS_loop',H_Sliced_loop)
// Thus, Mem Constraints Prioirty:
// 1)[Slice] Ensure grad_W_shape_sliced[V, H_sliced_loop]  < gen_bank_sizes_split_2_all_npus()
// 2)[Slice] Ensure grad_Y_shape_sliced[BS_loop. H_sliced] < gen_bank_sizes_split_2_all_npus()
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
  const data_type_t   grad_dtype,
  const data_type_t   Index_dtype,
  const int using_MCU_flag
) {
   TPUKERNEL_ASSERT_INFO(Index_dtype == DT_INT32, "You are trying to align index_addr to 64bits, but Only INT32 is supported!");
   const int core_idx = tpu_core_index();
   const int len_grad_dtype = tpu_data_type_size(grad_dtype);
   const int V = grad_W_shape.h;
   const int BS_native = count_numel_dim4(X_shape);
   TPUKERNEL_ASSERT_INFO(grad_Y_shape.w == grad_W_shape.w, "check H of [BS, H] <--> [V ,H] failed");
   //[Slice] Step 1: H_sliced
   int H_sliced_real        = 1, H_sliced_avg = 1, H_native = grad_W_shape.w;
   int min_cores_needed     = 1;
   compute_current_slice_info_multi_core_1D(H_native, &H_sliced_real, &H_sliced_avg, &min_cores_needed);
   if (core_idx < min_cores_needed) {
    //[Slice] Step 2: Ensue grad_W_shape_sliced(V,H_Sliced_loop) < gen_bank_sizes_split_2_all_npus()
    //[BS, H ,V] -> [BS, H_sliced, V] ->  [BS_loop, H_sliced_loop]
    int num_loop_upper_H_sliced_loop = 1;
    int H_sliced_loop_tmp = H_sliced_real;
    while (V * H_sliced_loop_tmp * len_grad_dtype >= gen_bank_sizes_split_2_all_npus()) {
        num_loop_upper_H_sliced_loop += 1;
        H_sliced_loop_tmp = DIV_UP(H_sliced_real, num_loop_upper_H_sliced_loop);
    }
    for (int idx_H_sliced_loop = 0; idx_H_sliced_loop < num_loop_upper_H_sliced_loop; idx_H_sliced_loop++) {
        int H_sliced_loop_real = 1, H_sliced_loop_avg  = 1;
        compute_current_loop_info_each_core_1D_with_loop_idx(H_sliced_real, &H_sliced_loop_real, &H_sliced_loop_avg, num_loop_upper_H_sliced_loop, idx_H_sliced_loop);
        //[Slice] Step 3: Ensure grad_Y_shape_sliced[BS_loop, H_sliced_loop] < gen_bank_sizes_split_2_all_npus()
        int BS_sliced_real = 1, BS_sliced_avg = 1;
        int PartRange_loop_tmp =  BS_native;
        int num_loop_upper_BS     = 1;
        TPUKERNEL_ASSERT_INFO(V * H_sliced_loop_real * len_grad_dtype < gen_bank_sizes_split_2_all_npus(), "at least svae one H_slice on L1!");
        while(PartRange_loop_tmp * H_sliced_loop_real * len_grad_dtype >=  gen_bank_sizes_split_2_all_npus()) {
          num_loop_upper_BS += 1;
          PartRange_loop_tmp = DIV_UP(BS_native, num_loop_upper_BS);
        }
        //[Error] when [B,S,V,H]=[6, 4096,2,12288] num_loop_upper is 2458
        // loop efficiency is low and Function_Recollect_Index cannot utilize same index among loops.
        for (int idx_BS_loop = 0 ; idx_BS_loop < num_loop_upper_BS; idx_BS_loop++) {
          compute_current_loop_info_each_core_1D_with_loop_idx(BS_native, &BS_sliced_real, &BS_sliced_avg, num_loop_upper_BS, idx_BS_loop);
          const int  core_dim_sliced_avg  =  H_sliced_loop_avg;
          const int  loop_dim_sliced_avg  =  BS_sliced_avg;
          const int  core_dim_sliced_real =  H_sliced_loop_real;
          const int  loop_dim_sliced_real =  BS_sliced_real;
          TPUKERNEL_ASSERT_INFO(loop_dim_sliced_avg * core_dim_sliced_avg * len_grad_dtype <= gen_bank_sizes_split_2_all_npus(), "[Slice] Step-2 Wrong");
          const int offset_idx_H_slice_loop = idx_H_sliced_loop * core_dim_sliced_avg * 1 * len_grad_dtype;
          const int offset_core_H_slice     = core_idx * H_sliced_avg * 1 * len_grad_dtype;
          const int offset_H                = offset_idx_H_slice_loop + offset_core_H_slice;
          global_addr_t grad_Y_addr_sliced                    = gradout_global_addr;
          global_addr_t X_addr_sliced                         = index_global_addr;
          global_addr_t grad_Weight_addr_sliced               = grad_input_global_addr;
          global_addr_t sorted_index_value_global_addr_sliced = sorted_index_value_global_addr;
          global_addr_t sorted_index_index_global_addr_sliced = sorted_index_index_global_addr;
          global_addr_t scatter_index_global_addr_sliced      =  scatter_index_global_flush_64b_aligned_addr;
          global_addr_t recollected_global_addr_sliced        = recollected_global_flush_64b_aligned_addr;
          //Grad_W [V,H]->[V, H_sliced_loop]
          grad_Weight_addr_sliced += offset_H;
          //Grad_Y [BS,H]->[BS_loop, H_sliced_loop]
          const int offset_BS_loop = idx_BS_loop    * loop_dim_sliced_avg * 1 * len_grad_dtype;
          grad_Y_addr_sliced      += offset_BS_loop * H_native +  offset_H;
          //X [B,S]->[BS]->[BS_loop]
          //[Note] Index(X) is free with idx_H_sliced_loop
          //[Error] [CONFLICT-INFO] loop=0,core=1; loop=0,core =2  if not enough or aligned correctly overlap area on these 4 index buffers will be flushed by mutli-thread.
          X_addr_sliced           += offset_BS_loop;
          const int align_offset_core    =  core_idx    * length_align_64bit_apporach(BS_native, Index_dtype);
          const int align_offset_loop    =  idx_BS_loop * length_align_64bit_apporach(loop_dim_sliced_avg, Index_dtype);
          sorted_index_value_global_addr_sliced += align_offset_loop + align_offset_core;
          sorted_index_index_global_addr_sliced += align_offset_loop + align_offset_core;
          scatter_index_global_addr_sliced      += align_offset_loop + align_offset_core;
          recollected_global_addr_sliced        += align_offset_loop + align_offset_core;
          TPUKERNEL_ASSERT_INFO(sorted_index_value_global_addr_sliced % 64==0, "HAU addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
          TPUKERNEL_ASSERT_INFO(sorted_index_index_global_addr_sliced % 64==0, "HAU addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
          TPUKERNEL_ASSERT_INFO(scatter_index_global_addr_sliced % 64==0, "flush addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
          TPUKERNEL_ASSERT_INFO(recollected_global_addr_sliced   % 64==0, "flush addr mush be aligned to 65bit, using <length_align_64bit_apporach>!");
          //Grad_Y [BS,H]->[BS_loop, H_slice_loop]
          const dim4 grad_Y_shape_sliced = {.n=1,.c=1,.h=loop_dim_sliced_real,.w= core_dim_sliced_real};
          //Grad_W [V,H]->[V, H_slice_loop]
          const dim4 grad_W_shape_sliced = {.n=1,.c=1,.h=V,                   .w= core_dim_sliced_real};
          //X [B,S]->[BS]->[BS_loop]
          const dim4 X_shape_sliced   = {.n=1,.c=1,.h=1,.w= loop_dim_sliced_real};
          //All slices are done before entering into pattern_one
          TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(grad_Y_shape_sliced, grad_dtype);
          TENSOR_MEM_ISOLATED_PRECHECK_L1_dim4(grad_W_shape_sliced, grad_dtype);
          nodechip_embedding_backward_multi_core_basic_cell_patttern_one_intracore(
              grad_Y_addr_sliced,
              X_addr_sliced,
              grad_Weight_addr_sliced,
              sorted_index_value_global_addr_sliced,
              sorted_index_index_global_addr_sliced,
              scatter_index_global_addr_sliced,
              recollected_global_addr_sliced,
              grad_Y_shape_sliced, //[BS_loop, H_sliced_loop]
              X_shape_sliced,      //[BS_loop]
              grad_W_shape_sliced, //[V, H_sliced_loop]
              H_native,
              idx_BS_loop,//==0 ? 1: 0,  //first_loop, grad_W is preset to 0 & no fp_add applied
              grad_dtype,
              Index_dtype,
              using_MCU_flag
              );
          }
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
  const dim4 grad_W_shape,
  const int BS,
  const int V,
  const data_type_t grad_dtype
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
  const int judger_mem = (H_sliced_avg * BS + BS +  V * H_sliced_avg) *  len_grad_dtype;
  //Constraints:
  //           (1) L2 must hold [V,H]+[BS]+[BS,H]
  //           (2) L1 might be not enough
  if (judger_mem < SG2260_GLOBAL_MEM) {
    //juder Rule:
    //  1)                      judger_mem <  gen_bank_sizes_split_2_per_npu()    Pattern_one [BS, V, H_sliced]
    //  1)                      judger_mem <  gen_bank_sizes_split_2_all_npus()    Pattern_one [BS, V, H_sliced]
    //  2) get_L1_TOTAL_MEM() > judger_mem >  gen_bank_sizes_split_2_all_npus()     Pattern_one + H_sliced_loop   => Patterb_Two degraded
    //  3)                      judger_mem >  get_L1_TOTAL_MEM()     Patterb_Two [BS_loop, V, H_sliced]=> enhanced [BS_loop, V, H_sliced_loop]
    if (judger_mem >= get_L1_TOTAL_MEM() || judger_mem >=  gen_bank_sizes_split_2_all_npus()) {
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
  bool is_index_int64,
  const int using_MCU_flag ) {
  const data_type_t Index_dtype = DT_INT32;

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

  const int excute_pattern_level_top = Decision_Adviosr_Top(grad_W_shape, BS, V, grad_dtype);

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
      nodechip_embedding_backward_multi_core_basic_cell_patttern_one_intracore (
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
        grad_dtype,
        Index_dtype,
        using_MCU_flag
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
        Index_dtype,
        using_MCU_flag
    );
  }  else {
    //Never slice V;
    TPUKERNEL_ASSERT_INFO(0, "V is too large for GMEM/L2/L1");
  }
}

int tpu_kernel_api_embedding_backward_multi_core ( const void* args ) {
  sg_api_embedding_backward_t *api = ( sg_api_embedding_backward_t * ) args;
  const int using_MCU_flag = IS_USING_MCU;
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
  api->is_index_int64,
  using_MCU_flag);
  tpu_poll();
  return 0;
}

TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_embedding_backward_multi_core );
#endif