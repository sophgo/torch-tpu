#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"

#ifdef USING_PPL
#include "Tgi_input_ids_update.h"
constexpr int MAX_BATCH_CHUNK_SIZE = 16;
#define AT_DISPATCH_FLOAT_INT_TYPES(scalar_type, name, func)  \
AT_DISPATCH_SWITCH(                   \
scalar_type, name,                    \
AT_DISPATCH_CASE(at::kFloat, func)    \
AT_DISPATCH_CASE(at::kHalf, func)     \
AT_DISPATCH_CASE(at::kBFloat16, func) \
AT_DISPATCH_CASE(at::kInt, func)      \
AT_DISPATCH_CASE(at::kShort, func)    \
AT_DISPATCH_CASE(at::kChar, func)     \
AT_DISPATCH_CASE(at::kByte, func))

template <typename scalar_t>
static void tgi_input_ids_update_impl(
    uint64_t all_input_ids_ptr,
    uint64_t next_ids_ptr,
    int input_length0,int input_length1,int input_length2,int input_length3,
    int input_length4,int input_length5,int input_length6,int input_length7,
    int64_t n_accept_ids,
    int base_batch_idx,
    int seq_length
) {
    scalar_t* input_ids = reinterpret_cast<scalar_t*>(all_input_ids_ptr);
    scalar_t* next_id = reinterpret_cast<scalar_t*>(next_ids_ptr);

    auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
        if constexpr (std::is_same_v<scalar_t, int32_t>) {
            return tgi_input_ids_update_int32(
                stream,
#ifndef BACKEND_SG2260
                ppl_module,
#endif
                reinterpret_cast<uint64_t>(input_ids),
                reinterpret_cast<uint64_t>(next_id),
                input_length0, input_length1, input_length2, input_length3,
                input_length4, input_length5, input_length6, input_length7,
                n_accept_ids, seq_length,base_batch_idx
            );
        }
        return -1;
    };

    auto stream = c10_tpu::getCurrentTPUStream();
    tpuKernelModule_t ppl_module = getPplModule();
    int ret = kernel(stream, ppl_module);
    if (ret == 0) {
        return;
    }
    TORCH_CHECK(false, "tgi_input_ids_update failed !");
}
#endif
namespace at
{
    void TGI_input_ids_update_decode_phase(
        Tensor &all_input_ids,
        Tensor &next_ids,
        IntArrayRef input_lengths,
        int64_t n_accept_ids)
    {
        TIMING_START;
        CHECK_TENSOR_IN_DEVICE(all_input_ids);
        CHECK_TENSOR_IN_DEVICE(all_input_ids);
        TORCH_CHECK((int)input_lengths.size() == (int)all_input_ids.size(0));
        TORCH_CHECK((n_accept_ids == 1) && "only support n_accept_ids == 1 now");

        std::vector<int> input_lengths_vec;
        for (int i = 0; i < (int)input_lengths.size(); i++)
        {
            input_lengths_vec.push_back(input_lengths[i]);
        }
#ifdef USING_PPL
        if (usePPLKernels()) {
            const int batch_size = all_input_ids.size(0);
            const int seq_length = all_input_ids.size(1);

            AT_DISPATCH_FLOAT_INT_TYPES(all_input_ids.scalar_type(), "TGI_input_ids_update", [&] {
                for (int b = 0; b < batch_size; b+=8) {
                    int end_idx = std::min(b + 8, batch_size);
                    tgi_input_ids_update_impl<scalar_t>(
                        reinterpret_cast<uint64_t>(all_input_ids.data_ptr()),
                        reinterpret_cast<uint64_t>(next_ids.data_ptr()),
                        input_lengths[b],
                        end_idx > b+1 ? input_lengths[b+1] : 0,
                        end_idx > b+2 ? input_lengths[b+2] : 0,
                        end_idx > b+3 ? input_lengths[b+3] : 0,
                        end_idx > b+4 ? input_lengths[b+4] : 0,
                        end_idx > b+5 ? input_lengths[b+5] : 0,
                        end_idx > b+6 ? input_lengths[b+6] : 0,
                        end_idx > b+7 ? input_lengths[b+7] : 0,
                        n_accept_ids,
                        b,
                        seq_length
                    );
                }
            });
        } else
#endif
       {
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnTgiInputIdsUpdateAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, all_input_ids),
            tpu::TPUGenerateTpudnnTensor(stream, next_ids),
            (int*)input_lengths_vec.data(),
            input_lengths_vec.size(),
            n_accept_ids);
        TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
        }
        TIMING_END;
        return;
    }

} // at