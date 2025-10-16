
#include "ppl.h"
#include "ppl_wrapper_func.h"
#include <vector>

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void arange_kernel(T *ptr_output, int start, int end, int step, const int n) {
  ppl::set_block_num(BLOCK_NUM);
  int block_idx = get_block_index();
  if (block_idx == 0)
  {
    int num = (end - start + step - (step > 0 ? 1 : -1)) / step;
    dim4 output_shape = {1, n, 1, num};

    auto out_g = gtensor<T>(output_shape, GLOBAL, ptr_output);
    auto out_l = make_tensor<T>(output_shape, output_shape);

    dma::load(out_l, out_g);
    if constexpr (std::is_same<T, fp32>::value || std::is_same<T, fp16>::value ||
                  std::is_same<T, bf16>::value) {
        auto out_int = make_tensor<int32>(output_shape, output_shape);
        arange_broadcast(out_int, n, start, step, num);
        tiu::cast(out_l, out_int);
    } else {
        arange_broadcast(out_l, n, start, step, num);
    }

    dma::store(out_g, out_l);
  }
}

__KERNEL__ void arange_int32(int32 *ptr_output, int start, int end, int step, const int output_shape) {
  arange_kernel<int32>(ptr_output, start, end, step, output_shape);
}

__KERNEL__ void arange_fp32(float *ptr_output, int start, int end, int step, const int output_shape) {
  arange_kernel<float>(ptr_output, start, end, step, output_shape);
}

__KERNEL__ void arange_fp16(fp16 *ptr_output, int start, int end, int step, const int output_shape) {
  arange_kernel<fp16>(ptr_output, start, end, step, output_shape);
}

__KERNEL__ void arange_bf16(bf16 *ptr_output, int start, int end, int step, const int output_shape) {
  arange_kernel<bf16>(ptr_output, start, end, step, output_shape);
}