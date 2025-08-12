#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"

#include "common/config.h"
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
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnTgiInputIdsUpdateAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, all_input_ids),
            tpu::TPUGenerateTpudnnTensor(stream, next_ids),
            (int*)input_lengths_vec.data(),
            input_lengths_vec.size(),
            n_accept_ids);
        TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
    	TIMING_END;
        return;
    }

} // at