
#include "ppl.h"
#include <vector>

using namespace ppl;
#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void tgi_input_ids_update_kernel_multicore(T *all_input_ids, T *next_ids,
    int input_length0, int input_length1, int input_length2, int input_length3,
    int input_length4, int input_length5, int input_length6, int input_length7,
    int n_accept_ids,int seq_length,int base_batch_idx) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int input_lengths[8] = {input_length0, input_length1, input_length2, input_length3,
                        input_length4, input_length5, input_length6, input_length7};

    for (int i = 0; i < 8; i++){
      if ( i % core_num == core_idx){
        enable_pipeline();
        dim4 shape = {1, 1, 1, n_accept_ids};
        auto in_gt = gtensor<T>(shape, GLOBAL, next_ids);
        auto out_gt = gtensor<T>(shape, GLOBAL, all_input_ids);

        dim4 src_offset = {0, 0, 0, base_batch_idx + i};
        dim4 dst_offset = {0, 0, 0, (base_batch_idx + i) * seq_length + input_lengths[i]};
        auto src_view = in_gt.sub_view(shape, src_offset);
        auto dst_view = out_gt.sub_view(shape, dst_offset);
        dma::move(dst_view, src_view);
      }
    }
}

__KERNEL__ void tgi_input_ids_update_int32(int32 *all_input_ids, int32 *next_ids,
            int input_length0, int input_length1, int input_length2, int input_length3,
            int input_length4, int input_length5, int input_length6, int input_length7,
            int n_accept_ids,int seq_length, int base_batch_idx) {
  tgi_input_ids_update_kernel_multicore<int32>(all_input_ids, next_ids,
      input_length0, input_length1, input_length2, input_length3,
      input_length4, input_length5, input_length6, input_length7,
      n_accept_ids, seq_length,base_batch_idx);
}
