#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include "TPUTorchUtils.h"
#include "common/config.h"

#if defined BACKEND_SG2260
#define CORE_NUM 8
#elif defined BACKEND_SG2260E
#define CORE_NUM 4
#endif

#ifdef USING_PPL
#include "MmW8A16Dq_ppl.h"

bool findValid_mn(uint32_t m_slice, uint32_t n_slice, std::function<int(TPUStream, tpuKernelModule_t, uint32_t, uint32_t)> kernel_func) {
    auto stream = c10_tpu::getCurrentTPUStream();
    tpuKernelModule_t ppl_module = getPplModule();
    const int possible_m_slices[] = {128};
	const int possible_n_slices[] = {128};
    const int num_possible_m_slices = sizeof(possible_m_slices) / sizeof(possible_m_slices[0]);
	const int num_possible_n_slices = sizeof(possible_n_slices) / sizeof(possible_n_slices[0]);

	for (int i = 0; i < num_possible_m_slices; i++) {
		m_slice = possible_m_slices[i];
		for (int j = 0; j < num_possible_n_slices; j++) {
            n_slice = possible_n_slices[j];
            int ret = kernel_func(stream, ppl_module, m_slice, n_slice);
            if (ret == 0) {
                return true;
            }
		}
	}
	return false;
}

bool findValid_k(uint32_t k_slice, std::function<int(TPUStream, tpuKernelModule_t, uint32_t)> kernel_func) {
  auto stream = c10_tpu::getCurrentTPUStream();
  tpuKernelModule_t ppl_module = getPplModule();

  const int possible_n_slices[] = {128};
  const int num_possible_slices = sizeof(possible_n_slices) / sizeof(possible_n_slices[0]);

  for (int i = 0; i < num_possible_slices; i++) {
	k_slice = possible_n_slices[i];
	int ret = kernel_func(stream, ppl_module, k_slice);
	if (ret == 0) {
		return true;
	}
  }
  return false;
}

static void mm_w8a16_dq_ppl_impl(
	uint64_t left_addr,
	uint64_t right_addr,
	uint64_t scale_addr,
	uint64_t output_addr,
	uint32_t M,
	uint32_t K,
	uint32_t N,
	uint32_t group_size)

{
	float zeropoint = sqrt((2 * CORE_NUM * M) / N); // minimize y = m * CORE_NUM / x * 2 + n * x; x is CORE_NUM for batch_slice
	int num[8];
	int candidate_count = 0;
	for (int i = 1; i <= CORE_NUM; i *= 2) {
		num[candidate_count++] = i;
	}

	if (candidate_count == 0 || num[candidate_count-1] != CORE_NUM) {
		num[candidate_count++] = CORE_NUM;
	}

	int left_index = 0;
	for (int i = 0; i < candidate_count; ++i) {
		if (zeropoint >= num[i]) {
			left_index = i;
		}
		else {
			break;
		}
	}
	int group_core = num[left_index];

	if (left_index >= 0 && left_index < candidate_count - 1) {
		int left_val = num[left_index];
		int right_val = num[left_index + 1];
		float lval = 2 * CORE_NUM * M / left_val + N * left_val;
		float rval = 2 * CORE_NUM * M / right_val + N * right_val;
		group_core = lval < rval ? left_val : right_val;
	}

	long long slice_m_n = (CORE_NUM / group_core - 1) * 2 * (long long) M * K + (group_core - 1) * N * K;
	long long slice_k = (CORE_NUM - 1) * 2 * (long long) M * N;

	uint32_t m_slice = M;
	uint32_t n_slice = N;
	uint32_t k_slice = K;

	uint32_t n_single = N;
	uint32_t k_single = K;

	std::function<int(TPUStream, tpuKernelModule_t, uint32_t, uint32_t)> kernel_mn;
	std::function<int(TPUStream, tpuKernelModule_t, uint32_t)> kernel_k;

	if(slice_m_n < slice_k){
		uint32_t cores_per_group = CORE_NUM / group_core;
		uint32_t percore_m = ((M + group_core - 1)/ group_core);
		uint32_t percore_n = ((N + cores_per_group * group_size - 1) / (cores_per_group * group_size)) * group_size;
		n_single = percore_n;
		m_slice = percore_m;
		n_slice = percore_n;
		k_slice = K;

		kernel_mn = [&](TPUStream stream, tpuKernelModule_t ppl_module, uint32_t m_slice, uint32_t n_slice) -> int {
				return  matmul_w8a16_mn_mn_bf16(stream,
				#ifndef BACKEND_SG2260
				ppl_module,
				#endif
				output_addr,left_addr, right_addr, scale_addr,
				M, K, N, group_size, m_slice, n_slice, k_slice, group_core);
		};

		kernel_k = [&](TPUStream stream, tpuKernelModule_t ppl_module, uint32_t k_slice) -> int {
				return  matmul_w8a16_mn_k_bf16(stream,
				#ifndef BACKEND_SG2260
				ppl_module,
				#endif
				output_addr,left_addr, right_addr, scale_addr,
				M, K, N, group_size, m_slice, n_slice, k_slice, group_core);
		};
	}
	else{
		uint32_t percore_k = ((K + CORE_NUM - 1) / (CORE_NUM ));
		m_slice = M;
		n_slice = N;
		k_slice = percore_k;
		k_single = percore_k;

		kernel_mn = [&](TPUStream stream, tpuKernelModule_t ppl_module, uint32_t m_slice, uint32_t n_slice) -> int {
				return  matmul_w8a16_k_mn_bf16(stream,
					#ifndef BACKEND_SG2260
					ppl_module,
					#endif
					output_addr,left_addr, right_addr, scale_addr,
					M, K, N, group_size, m_slice, n_slice, k_slice);
		};

		kernel_k = [&](TPUStream stream, tpuKernelModule_t ppl_module, uint32_t k_slice) -> int {
				return  matmul_w8a16_k_k_bf16(stream,
					#ifndef BACKEND_SG2260
					ppl_module,
					#endif
					output_addr,left_addr, right_addr, scale_addr,
					M, K, N, group_size, m_slice, n_slice, k_slice);
		};
	}

	if(k_single < n_single){
		bool ret_k = findValid_k(k_slice, kernel_k);
		if(ret_k) return;
		else{
			bool ret_mn = findValid_mn(m_slice, n_slice, kernel_mn);
			if(ret_mn) return;
			else TORCH_CHECK(false, "Tile size reduction failed after attempts");
		}
	}
	else{
		bool ret_mn = findValid_mn(m_slice, n_slice, kernel_mn);
		if(ret_mn) return;
		else{
			bool ret_k = findValid_k(k_slice, kernel_k);
			if(ret_k) return;
			else  TORCH_CHECK(false, "Tile size reduction failed after attempts");
		}
	}
}
#endif

namespace at
{
	Tensor mm_w8a16_dq_forward(
		Tensor &input,
		Tensor &weight,
		Tensor &scale,
		Tensor &output,
		int64_t blocksize)
	{
		TIMING_START;
		CHECK_TENSOR_IN_DEVICE(input);
		CHECK_TENSOR_IN_DEVICE(weight);
        CHECK_TENSOR_IN_DEVICE(scale);
		CHECK_TENSOR_IN_DEVICE(output);

#if defined BACKEND_SG2260  ||  defined BACKEND_SG2260E
#ifdef USING_PPL
	if (usePPLKernels()){
		mm_w8a16_dq_ppl_impl(
			GetAddrByUnifiedAddr(reinterpret_cast<uint64_t>(input.data_ptr())),
			GetAddrByUnifiedAddr(reinterpret_cast<uint64_t>(weight.data_ptr())),
			GetAddrByUnifiedAddr(reinterpret_cast<uint64_t>(scale.data_ptr())),
			GetAddrByUnifiedAddr(reinterpret_cast<uint64_t>(output.data_ptr())),
			input.size(0),
			input.size(1),
			weight.size(0),
			blocksize);
	}else
#endif
	{
		auto stream = c10_tpu::getCurrentTPUStream();
		auto status = tpudnnMmW8A16DqAsync(
			stream,
			tpu::TPUGenerateTpudnnTensor( stream, input),
			tpu::TPUGenerateTpudnnTensor( stream, weight),
            tpu::TPUGenerateTpudnnTensor( stream, scale),
			tpu::TPUGenerateTpudnnTensor( stream, output),
            blocksize);
		TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
	}
#elif defined BACKEND_1684X
		TORCH_CHECK(false);
#endif
		TIMING_END;
		return output;
	}

}
