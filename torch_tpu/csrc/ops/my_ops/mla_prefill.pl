#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

#ifdef __bm1690__
#define CORE_NUM 8
#elif defined(__sg2260e__)
#define CORE_NUM 4
#endif
#define MAX_BATCH_SIZE 512

// Define the attention modes
// CONTINOUS_KV_CACHE_PREFILL = 0
// CONTINOUS_KV_CACHE_DECODE = 1
// PAGE_KV_CACHE_PREFILL = 2
// PAGE_KV_CACHE_DECODE = 3

#ifdef __bm1690__
const int block_num_heads = 16;
#elif defined(__sg2260e__)
const int block_num_heads = 8;
#endif

const int block_batch = 128;
const int block_dq = 1536;
const int block_dkv = 512;
const int block_dnope = 128;
const int block_dpe = 64;
const int block_dv = 128;
const int block_max_cache_size = 128 * 1024;
const int max_gather_scatter_num = 17000000;
const int block_paged_block_size = 16;
const int block_quant_block_size = 128;

const int block_q_secs = 512;
const int block_kv_secs = 512;
const int block_cache_seq_len = 2048;
const int block_seq_len = 4096;
const int block_max_paged_block_num =
    (block_cache_seq_len + block_seq_len) / block_paged_block_size;

void generate_slot_idx_per_batch(tensor<uint32> &slot_idx_tensor,
                                 gtensor<uint32> &block_table_global_tensor,
                                 int max_paged_block_num, int block_size) {
  ppl::assert(block_size <= NPU_NUM);
  dim4 block_table_shape = {1, NPU_NUM, 1, block_max_paged_block_num};
  dim4 block_table_real_shape = {1, NPU_NUM, 1, max_paged_block_num};
  auto block_table_tensor =
      make_tensor<uint32>(block_table_shape, block_table_real_shape);
  auto work_tensor =
      make_tensor<uint32>(block_table_shape, block_table_real_shape);

  dim4 shape_per_npu = {CORE_NUM, 1, 1, block_max_paged_block_num};
  dim4 real_shape_per_npu = {CORE_NUM, 1, 1, max_paged_block_num};
  int core_idx = ppl::get_core_index();
  dim4 real_shape_per_core_npu = {1, 1, 1, max_paged_block_num};
  dim4 core_offset = {core_idx, 0, 0, 0};
  auto const_l2_tensor =
      make_l2tensor<uint32>(shape_per_npu, L2, real_shape_per_npu)
          .sub_view(real_shape_per_core_npu, core_offset);

  // ① tpu_gdma_channel_bcast_S2L 把 block_table 从 DRAM 广播到每个 NPU，形状
  // (NPU_NUM, max_paged_block_num)
  ppl::dma::load_broadcast(block_table_tensor, block_table_global_tensor);

  // ② 计算每个 block 起始索引 = block_id * block_size
  ppl::tiu::mul(block_table_tensor, block_table_tensor, block_size);

  // ③ 生成一共 block_size 个 block 内偏移, 形状仍为 (NPU_NUM,
  // max_paged_block_num)
  for (int i = 0; i < block_size; ++i) {
    dim4 work_offset = {0, i, 0, 0};
    ppl::dma::fill(const_l2_tensor, i);
    ppl::dma::load(work_tensor.sub_view(real_shape_per_core_npu, work_offset),
                   const_l2_tensor);
  }

  // ④ 起始索引 + 偏移 = slot 索引
  dim4 shape = {1, block_size, 1, max_paged_block_num};
  dim4 offset = {0, 0, 0, 0};
  ppl::tiu::add(slot_idx_tensor, block_table_tensor.sub_view(shape, offset),
                work_tensor.sub_view(shape, offset));
}

void generate_slot_idx(gtensor<uint32> &slot_idx_l2_tensor,
                       gtensor<uint32> &block_table_global_tensor,
                       int max_paged_block_num, int block_size, int batch) {
  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();
  int batch_slice = ppl::div_up(batch, core_num);
  int batch_secs = ppl::div_up(batch, batch_slice);

  if (core_idx < batch_secs) {
    int batch_len = (core_idx == batch_secs - 1)
                        ? batch - core_idx * batch_slice
                        : batch_slice;
    int batch_start = core_idx * batch_slice;
    int batch_end = batch_start + batch_len;

    dim4 slot_idx_shape = {1, block_paged_block_size, 1,
                           block_batch * block_max_paged_block_num};
    dim4 slot_idx_real_shape = {1, block_size, 1,
                                batch_len * max_paged_block_num};
    auto slot_idx_tensor =
        make_tensor<uint32>(slot_idx_shape, slot_idx_real_shape);

    for (int bidx = batch_start; bidx < batch_end; ++bidx) {
      dim4 slot_idx_sub_shape = {1, block_size, 1, max_paged_block_num};
      dim4 slot_idx_sub_offset = {0, 0, 0,
                                  (bidx - batch_start) * max_paged_block_num};
      dim4 block_table_sub_shape = {1, 1, 1, max_paged_block_num};
      dim4 block_table_sub_offset = {bidx, 0, 0, 0};
      generate_slot_idx_per_batch(
          slot_idx_tensor.sub_view(slot_idx_sub_shape, slot_idx_sub_offset),
          block_table_global_tensor.sub_view(block_table_sub_shape,
                                             block_table_sub_offset),
          max_paged_block_num, block_size);
    }

    dim4 slot_idx_cw_trans_shape = {1, block_batch * block_max_paged_block_num,
                                    1, block_paged_block_size};
    dim4 slot_idx_cw_trans_real_shape = {1, batch_len * max_paged_block_num, 1,
                                         block_size};
    auto slot_idx_cw_trans_tensor = make_tensor<uint32>(
        slot_idx_cw_trans_shape, slot_idx_cw_trans_real_shape);
    ppl::tiu::transpose_wc(slot_idx_cw_trans_tensor, slot_idx_tensor);
    dim4 slot_idx_l2_sub_shape = {batch_len, max_paged_block_num, 1,
                                  block_size};
    dim4 slot_idx_l2_sub_offset = {batch_start, 0, 0, 0};
    ppl::dma::store(slot_idx_l2_tensor
                        .sub_view(slot_idx_l2_sub_shape, slot_idx_l2_sub_offset)
                        .view(slot_idx_cw_trans_real_shape),
                    slot_idx_cw_trans_tensor);
  }
  ppl::sync();
}

template <typename TYPE>
void llama_rope_local(tensor<TYPE> &input_tensor, tensor<TYPE> &cos_tensor,
                      tensor<TYPE> &sin_tensor, tensor<TYPE> &work_tensor,
                      tensor<TYPE> &output_tensor, dim4 &shape, bool inplace) {
  dim4 neg_shape = {shape.n, shape.c, shape.h, shape.w / 2};
  dim4 dst_shape = {shape.n, shape.c, shape.h, shape.w};

  dim4 input_stride;
  ppl::aligned_stride_4d(&input_stride, &shape, 0, sizeof(TYPE));
  dim4 stride = {input_stride.n, input_stride.c, input_stride.h, 2};

  dim4 src_offset = {0, 0, 0, 1};
  dim4 dst_offset = {0, 0, 0, shape.w / 2};
  dim4 zero_offset = {0, 0, 0, 0};

  ppl::tiu::move(work_tensor.sub_view(neg_shape, dst_offset),
                 input_tensor.view(neg_shape, stride));
  ppl::tiu::fmul(
      work_tensor.sub_view(neg_shape, zero_offset),
      input_tensor.sub_view(neg_shape, src_offset).view(neg_shape, stride),
      -1.0);
  ppl::tiu::fmul(work_tensor, work_tensor, sin_tensor);

  ppl::tiu::move(output_tensor.sub_view(neg_shape, zero_offset),
                 input_tensor.view(neg_shape, stride));
  ppl::tiu::move(
      output_tensor.sub_view(neg_shape, dst_offset),
      input_tensor.sub_view(neg_shape, src_offset).view(neg_shape, stride));
  ppl::tiu::fmul(output_tensor, output_tensor, cos_tensor);

  if (inplace) {
    ppl::tiu::fadd(input_tensor, work_tensor, output_tensor);
  } else {
    ppl::tiu::fadd(output_tensor, work_tensor, output_tensor);
  }
}

template <typename TYPE>
void ds_attention_core(tensor<TYPE> &Q_tensor, tensor<TYPE> &K_tensor,
                       tensor<TYPE> &V_tensor, tensor<TYPE> &mask_tensor,
                       tensor<TYPE> &Y_tensor, tensor<TYPE> &last_max_tensor,
                       tensor<fp32> &softmax_exp_sum_tensor, int batch,
                       int Qseq, int Qhead, int KVseq, int KVhead, int dimq,
                       int dimv, bool is_first_K_slice, bool is_last_K_slice,
                       bool has_mask, bool Q_is_slice, float C) {
  // 1.S=Q*K
  dim4 QK_shape = {1, block_q_secs, 1, block_kv_secs};
  dim4 QK_real_shape = {1, Qseq, 1, KVseq};
  auto QK_tensor = make_tensor<fp32>(QK_shape, QK_real_shape);
  ppl::tiu::fmm2(QK_tensor, Q_tensor, K_tensor, false, true, false);
  ppl::tiu::fmul(QK_tensor, QK_tensor, C);

  auto work0_tensor = make_tensor<TYPE>(QK_shape, QK_real_shape);
  ppl::tiu::cast(work0_tensor, QK_tensor);

  // 2.Where
  if (has_mask) {
    ppl::tiu::fadd(work0_tensor, work0_tensor, mask_tensor);
  }

  // 3.rowmax(S)
  dim4 max_val_shape = {1, block_q_secs, 1, 1};
  dim4 max_val_real_shape = {1, Qseq, 1, 1};
  auto work1_tensor = make_tensor<TYPE>(max_val_shape, max_val_real_shape);
  auto work2_tensor = make_tensor<TYPE>(max_val_shape, max_val_real_shape);
  auto work3_tensor = make_tensor<fp32>(max_val_shape, max_val_real_shape);
  quick_pooling(work1_tensor, work0_tensor, &QK_shape, &QK_real_shape, -15000,
                0);

  if (!is_first_K_slice) {
    // 4. max_val : m2 = max(m1, rowmax(S2))
    ppl::tiu::fmax(work1_tensor, last_max_tensor, work1_tensor);
    //  bc_param : m1 - m2
    ppl::tiu::fsub(work2_tensor, last_max_tensor, work1_tensor);
    // copy m2 to last_max_val
    ppl::tiu::move(last_max_tensor, work1_tensor);
    // exp(m1 - m2)
    exp_no_overflow(work2_tensor, work2_tensor, &max_val_shape,
                    &max_val_real_shape);
    // update exp_sum
    // exp(m1 - m2) *l1
    ppl::tiu::cast(work3_tensor, work2_tensor);
    ppl::tiu::fmul(softmax_exp_sum_tensor, softmax_exp_sum_tensor,
                   work3_tensor);
    // update output
    // exp(m1 - m2)*O1
    ppl::tiu::fmul(Y_tensor, Y_tensor, work2_tensor);
  } else {
    ppl::tiu::move(last_max_tensor, work1_tensor);
  }

  // S2 - m2
  ppl::tiu::fsub(work0_tensor, work0_tensor, last_max_tensor);
  // exp(S2-m2)
  exp_no_overflow(work0_tensor, work0_tensor, &QK_shape, &QK_real_shape);
  // l = rowsum(exp(S2-m2))
  quick_pooling(work1_tensor, work0_tensor, &QK_shape, &QK_real_shape, 0, 1);

  if (!is_first_K_slice) {
    // l2 = exp(m1 - m2) * l1 + rowsum(exp(S2 - m2))
    ppl::tiu::cast(work3_tensor, work1_tensor);
    ppl::tiu::fadd(softmax_exp_sum_tensor, softmax_exp_sum_tensor,
                   work3_tensor);
  } else {
    ppl::tiu::cast(softmax_exp_sum_tensor, work1_tensor);
  }

  // exp(S2 - m2) * V
  dim4 Y_shape = {1, block_q_secs, 1, block_dv};
  dim4 Y_real_shape = {1, Qseq, 1, dimv};
  auto Y_work_tensor = make_tensor<TYPE>(Y_shape, Y_real_shape);
  ppl::tiu::fmm2(Y_work_tensor, work0_tensor, V_tensor, false, false, false);

  if (!is_first_K_slice) {
    // exp(m1 - m2)*O1 + exp(S2 - m2) * V
    ppl::tiu::fadd(Y_tensor, Y_tensor, Y_work_tensor);
  } else {
    ppl::tiu::move(Y_tensor, Y_work_tensor);
  }

  if (is_last_K_slice) {
    ppl::tiu::fdiv(work3_tensor, 1.0, softmax_exp_sum_tensor);
    ppl::tiu::cast(work1_tensor, work3_tensor);
    ppl::tiu::fmul(Y_tensor, Y_tensor, work1_tensor);
  }
}

template <typename TYPE>
void nodechip_deepseekv3_mla_local_prefill(
    tensor<TYPE> &Q_tensor, tensor<TYPE> &PE_tensor,
    tensor<TYPE> &PE_store_tensor, tensor<TYPE> &RoPE_cos_tensor,
    tensor<TYPE> &RoPE_sin_tensor, tensor<TYPE> &KRoPE_cos_tensor,
    tensor<TYPE> &KRoPE_sin_tensor, tensor<TYPE> &WUQ_tensor,
    tensor<TYPE> &mask_tensor, tensor<TYPE> &last_max_tensor,
    tensor<fp32> &softmax_exp_sum_tensor, tensor<TYPE> &Y_tensor,
    tensor<TYPE> &QU_tensor, tensor<TYPE> &Q_rope_tensor,
    tensor<TYPE> &KVU_tensor, float C, int batch, int q_seq, int kv_seq,
    int n_heads,
    int q_lora_rank,      // 1536
    int kv_lora_rank,     // 512
    int qk_nope_head_dim, // 128
    int qk_rope_head_dim, // 64
    int v_head_dim,       // 128
    int secs, bool K_first, bool K_last, bool has_mask, bool Q_first,
    bool kv_slice_cached

) {
  dim4 Q_u_shape = {batch, q_seq, n_heads, qk_nope_head_dim + qk_rope_head_dim};
  dim4 KV_u_shape = {batch, kv_seq, n_heads, qk_nope_head_dim + v_head_dim};
  dim4 ku_shape = {batch, kv_seq, n_heads, qk_nope_head_dim};
  dim4 vu_shape = {batch, kv_seq, n_heads, v_head_dim};
  dim4 Q_rope_shape = {batch, q_seq, n_heads, qk_rope_head_dim};
  dim4 PE_rope_shape = {batch, kv_seq, 1, qk_rope_head_dim};
  dim4 K_nope_shape = {batch, kv_seq, 1, qk_nope_head_dim};
  dim4 Q_actual_shape = {batch, q_seq, 1, qk_nope_head_dim + qk_rope_head_dim};
  dim4 K_actual_shape = {batch, kv_seq, 1, qk_nope_head_dim + qk_rope_head_dim};
  dim4 V_actual_shape = {batch, kv_seq, 1, v_head_dim};
  dim4 Y_actual_shape = {batch, q_seq, 1, v_head_dim};
  dim4 reduce_shape = {batch * n_heads, q_seq, 1, 1};
  dim4 Y_shape = {batch, q_seq, n_heads, v_head_dim};

  // 计算Q_c * WUQ = Q_u, 由于外循环Q，只在每个K第一次时做解压
  if (K_first) {
    ppl::tiu::fmm2(QU_tensor, Q_tensor, WUQ_tensor, false, true, false);
    dim4 QU_offset = {0, 0, 0, qk_nope_head_dim};
    ppl::tiu::move(Q_rope_tensor.view(Q_rope_shape),
                   QU_tensor.view(Q_u_shape).sub_view(Q_rope_shape, QU_offset));

    dim4 Q_rope_block_shape = {1, block_q_secs, block_num_heads / CORE_NUM,
                               block_dpe};
    auto work0_tensor = make_tensor<TYPE>(Q_rope_block_shape, Q_rope_shape);
    auto work1_tensor = make_tensor<TYPE>(Q_rope_block_shape, Q_rope_shape);

    llama_rope_local(Q_rope_tensor.view(Q_rope_shape), RoPE_cos_tensor,
                     RoPE_sin_tensor, work0_tensor, work1_tensor, Q_rope_shape,
                     true);

    ppl::tiu::move(QU_tensor.view(Q_u_shape).sub_view(Q_rope_shape, QU_offset),
                   Q_rope_tensor.view(Q_rope_shape));
  }

  // 同理，只在第一个Q的时候，计算PE RoPE
  if (Q_first && !kv_slice_cached) {
    // K_rope，每个head均一样，遍历n_heads时concat即可
    dim4 PE_rope_block_shape = {1, block_kv_secs, 1, block_dpe};
    auto work0_tensor = make_tensor<TYPE>(PE_rope_block_shape, PE_rope_shape);
    auto work1_tensor = make_tensor<TYPE>(PE_rope_block_shape, PE_rope_shape);
    llama_rope_local(PE_tensor.view(PE_rope_shape), KRoPE_cos_tensor,
                     KRoPE_sin_tensor, work0_tensor, work1_tensor,
                     PE_rope_shape, true);
    ppl::tiu::move(PE_store_tensor, PE_tensor);
  }

  dim4 Q_actual_block_shape = {1, block_q_secs, 1, block_dnope + block_dpe};
  auto Q_actual_tensor =
      make_tensor<TYPE>(Q_actual_block_shape, Q_actual_shape);

  dim4 K_actual_block_shape = {1, block_kv_secs, 1, block_dnope + block_dpe};
  auto K_actual_tensor =
      make_tensor<TYPE>(K_actual_block_shape, K_actual_shape);

  dim4 V_actual_block_shape = {1, block_kv_secs, 1, block_dv};
  auto V_actual_tensor =
      make_tensor<TYPE>(V_actual_block_shape, V_actual_shape);

  dim4 Y_actual_block_shape = {1, block_q_secs, 1, block_dv};
  auto Y_actual_tensor =
      make_tensor<TYPE>(Y_actual_block_shape, Y_actual_shape);

  for (int head_idx = 0; head_idx < n_heads; ++head_idx) {
    dim4 head_offset = {0, 0, head_idx, 0};
    dim4 K_actual_offset = {0, 0, 0, qk_nope_head_dim};
    dim4 V_actual_offset = {0, 0, head_idx, qk_nope_head_dim};
    dim4 zero_offset = {0, 0, 0, 0};

    ppl::tiu::move(Q_actual_tensor, QU_tensor.view(Q_u_shape).sub_view(
                                        Q_actual_shape, head_offset));
    ppl::tiu::move(K_actual_tensor.sub_view(PE_rope_shape, K_actual_offset),
                   PE_tensor.view(PE_rope_shape));
    ppl::tiu::move(
        K_actual_tensor.sub_view(K_nope_shape, zero_offset),
        KVU_tensor.view(KV_u_shape).sub_view(K_nope_shape, head_offset));
    ppl::tiu::move(
        V_actual_tensor,
        KVU_tensor.view(KV_u_shape).sub_view(V_actual_shape, V_actual_offset));

    ppl::tiu::move(Y_actual_tensor, Y_tensor.view(Y_shape).sub_view(
                                        Y_actual_shape, head_offset));

    dim4 reduce_shape = {1, q_seq, 1, 1};
    dim4 reduce_head_idx = {head_idx, 0, 0, 0};
    ds_attention_core(
        Q_actual_tensor, K_actual_tensor, V_actual_tensor, mask_tensor,
        Y_actual_tensor,
        last_max_tensor.sub_view(reduce_shape, reduce_head_idx),
        softmax_exp_sum_tensor.sub_view(reduce_shape, reduce_head_idx), batch,
        q_seq, 1, kv_seq, 1, qk_nope_head_dim + qk_rope_head_dim, v_head_dim,
        K_first, K_last, has_mask & !kv_slice_cached, secs < q_seq, C);

    ppl::tiu::move(Y_tensor.view(Y_shape).sub_view(Y_actual_shape, head_offset),
                   Y_actual_tensor);
  }
}

template <typename TYPE, typename W_TYPE>
void nodechip_deepseekv3_mla_prefill(
    gtensor<TYPE> &q_global_tensor, gtensor<TYPE> &kv_global_tensor,
    gtensor<TYPE> &pe_global_tensor, gtensor<TYPE> &kvcache_global_tensor,
    gtensor<TYPE> &pecache_global_tensor, gtensor<TYPE> &KVU_global_tensor,
    gtensor<TYPE> &cos_global_tensor, gtensor<TYPE> &sin_global_tensor,
    gtensor<TYPE> &mask_global_tensor, gtensor<W_TYPE> &wuq_global_tensor,
    gtensor<W_TYPE> &wukv_global_tensor, gtensor<TYPE> &wuq_scale_global_tensor,
    gtensor<TYPE> &wuq_scale_l2_tensor, gtensor<TYPE> &wukv_scale_global_tensor,
    gtensor<TYPE> &y_global_tensor, float C, int batch, int bidx,
    int mask_batch, int mask_max, int seq_len, int cache_seq_len,
    int heads_total, int n_heads,
    int q_lora_rank,      // 1536
    int kv_lora_rank,     // 512
    int qk_nope_head_dim, // 128
    int qk_rope_head_dim, // 64
    int v_head_dim,       // 128
    bool has_mask, int core_idx,
    gtensor<uint32> &slot_idx_l2_tensor, int wuq_quant_block_size,
    int wukv_quant_block_size, int quant_block_size, int attention_mode) {
  const int qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  const int kv_head_dim = qk_nope_head_dim + v_head_dim;

  int kv_seq = seq_len + cache_seq_len;
  int q_seq = seq_len;

  bool store_KVU = block_kv_secs < kv_seq && block_q_secs < q_seq;
  bool load_PE = block_kv_secs < kv_seq;

  dim4 wuq_shape = {1, block_num_heads / CORE_NUM * (block_dpe + block_dnope),
                    1, block_dq};
  dim4 wuq_real_shape = {1, n_heads * qk_head_dim, 1, q_lora_rank};
  auto wuq_local_tensor = make_tensor<TYPE>(wuq_shape, wuq_real_shape);

  dim4 wukv_shape = {1, block_num_heads / CORE_NUM * (block_dnope + block_dv),
                     1, block_dkv};
  dim4 wukv_real_shape = {1, n_heads * kv_head_dim, 1, kv_lora_rank};
  auto wukv_local_tensor = make_tensor<TYPE>(wukv_shape, wukv_real_shape);

  int dtype_eu_num = ppl::get_eu_num<TYPE>();

  if (bidx == 0) {
    // 由于不同batch的wuq/wukv大小、位置、内容完全一致，故仅在bidx ==
    // 0时搬运 从system mem拷贝wq/wkv至localmem
    if (std::is_same_v<W_TYPE, fp8e5m2> || std::is_same_v<W_TYPE, fp8e4m3>) {
      // load wuq
      auto wuq_wtype_local_tensor =
          make_tensor<W_TYPE>(wuq_shape, wuq_real_shape);
      ppl::dma::load(wuq_wtype_local_tensor, wuq_global_tensor);

      // cast fp8 -> fp16
      ppl::tiu::cast(wuq_local_tensor, wuq_wtype_local_tensor);

      // load wuq_scale
      dim4 wuq_scale_shape = {
          block_num_heads / CORE_NUM * (block_dpe + block_dnope) /
              block_quant_block_size * 2,
          NPU_NUM, ppl::div_up(block_dq, block_quant_block_size) * 2, 1};
      dim4 wuq_scale_real_shape = {
          n_heads * qk_head_dim / wuq_quant_block_size, NPU_NUM,
          ppl::div_up(q_lora_rank, wuq_quant_block_size), 1};
      auto wuq_scale_local_tensor =
          make_tensor<TYPE>(wuq_scale_shape, wuq_scale_real_shape);
      dim4 wuq_scale_global_shape = {
          n_heads * qk_head_dim / wuq_quant_block_size, 1,
          ppl::div_up(q_lora_rank, wuq_quant_block_size), 1};
      if (wuq_quant_block_size == quant_block_size) {
        ppl::dma::load_broadcast(
            wuq_scale_local_tensor,
            wuq_scale_global_tensor.view(wuq_scale_global_shape));
      } else {
        ppl::dma::load_broadcast(
            wuq_scale_local_tensor,
            wuq_scale_l2_tensor.view(wuq_scale_global_shape));
      }

      // wuq * wuq_scale
      dim4 wuq_mul_shape = {wuq_scale_real_shape.n, wuq_quant_block_size,
                            wuq_scale_real_shape.h, wuq_quant_block_size};
      int wuq_scale_stride_n = ppl::align(wuq_scale_real_shape.h, dtype_eu_num);
      dim4 wuq_scale_stride = {wuq_scale_stride_n, 0, 1, 0};
      ppl::tiu::fmul(
          wuq_local_tensor.view(wuq_mul_shape),
          wuq_local_tensor.view(wuq_mul_shape),
          wuq_scale_local_tensor.view(wuq_scale_real_shape, wuq_scale_stride));

      // load wukv
      auto wukv_wtype_local_tensor =
          make_tensor<W_TYPE>(wukv_shape, wukv_real_shape);
      ppl::dma::load(wukv_wtype_local_tensor, wukv_global_tensor);

      // cast fp8 -> fp16
      ppl::tiu::cast(wukv_local_tensor, wukv_wtype_local_tensor);

      // load wukv_scale
      dim4 wukv_scale_shape = {
          block_num_heads / CORE_NUM * (block_dnope + block_dv) /
              block_quant_block_size,
          NPU_NUM, ppl::div_up(block_dkv, block_quant_block_size), 1};
      dim4 wukv_scale_real_shape = {
          n_heads * kv_head_dim / wukv_quant_block_size, NPU_NUM,
          ppl::div_up(kv_lora_rank, wukv_quant_block_size), 1};
      auto wukv_scale_local_tensor =
          make_tensor<TYPE>(wukv_scale_shape, wukv_scale_real_shape);
      dim4 wukv_scale_global_shape = {
          n_heads * kv_head_dim / wukv_quant_block_size, 1,
          ppl::div_up(kv_lora_rank, wukv_quant_block_size), 1};
      ppl::dma::load_broadcast(
          wukv_scale_local_tensor,
          wukv_scale_global_tensor.view(wukv_scale_global_shape));

      // wukv * wukv_scale
      dim4 wukv_mul_shape = {wukv_scale_real_shape.n, wukv_quant_block_size,
                             wukv_scale_real_shape.h, wukv_quant_block_size};
      int wukv_scale_stride_n =
          ppl::align(wukv_scale_real_shape.h, dtype_eu_num);
      dim4 wukv_scale_stride = {wukv_scale_stride_n, 0, 1, 0};
      ppl::tiu::fmul(wukv_local_tensor.view(wukv_mul_shape),
                     wukv_local_tensor.view(wukv_mul_shape),
                     wukv_scale_local_tensor.view(wukv_scale_real_shape,
                                                  wukv_scale_stride));
    } else {
      ppl::dma::load(wuq_local_tensor, wuq_global_tensor.template view<TYPE>());
      ppl::dma::load(wukv_local_tensor,
                     wukv_global_tensor.template view<TYPE>());
    }
  }

  int q_seq_tail =
      q_seq % block_q_secs == 0 ? block_q_secs : q_seq % block_q_secs;

  dim4 QU_shape = {1, block_q_secs, 1,
                   block_num_heads / CORE_NUM * (block_dpe + block_dnope)};
  dim4 QU_real_shape = {1, q_seq_tail, 1, n_heads * qk_head_dim};
  auto QU_tensor = make_tensor<TYPE>(QU_shape, QU_real_shape);

  dim4 Q_rope_shape = {1, block_q_secs, 1,
                       block_num_heads / CORE_NUM * block_dpe};
  dim4 Q_rope_real_shape = {1, q_seq_tail, 1, n_heads * qk_rope_head_dim};
  auto Q_rope_tensor = make_tensor<TYPE>(Q_rope_shape, Q_rope_real_shape);

  dim4 Y_shape = {1, block_q_secs, 1, block_num_heads / CORE_NUM * block_dv};
  dim4 Y_real_shape = {1, q_seq_tail, 1, n_heads * v_head_dim};
  auto Y_tensor = make_tensor<TYPE>(Y_shape, Y_real_shape);

  dim4 last_max_shape = {block_num_heads / CORE_NUM, block_q_secs, 1, 1};
  dim4 last_max_real_shape = {n_heads, block_q_secs, 1, 1};
  auto last_max_tensor = make_tensor<TYPE>(last_max_shape, last_max_real_shape);
  auto softmax_exp_sum_tensor =
      make_tensor<fp32>(last_max_shape, last_max_real_shape);
  bool K_first, K_last, Q_first;

  for (int q_seq_idx = 0; q_seq_idx < q_seq; q_seq_idx += block_q_secs) {
    int q_secs = ppl::min(block_q_secs, q_seq - q_seq_idx);
    for (int kv_seq_idx = 0; kv_seq_idx < kv_seq; kv_seq_idx += block_kv_secs) {
      ppl::enable_pipeline();
      int kv_secs;
      if (kv_seq_idx < cache_seq_len) {
        kv_secs = ppl::min(block_kv_secs, cache_seq_len - kv_seq_idx);
      } else {
        kv_secs = ppl::min(block_kv_secs, kv_seq - kv_seq_idx);
      }

      // load kv
      dim4 KV_shape = {1, block_kv_secs, 1, block_dkv};
      dim4 KV_real_shape = {1, kv_secs, 1, kv_lora_rank};
      auto KV_tensor = make_tensor<TYPE>(KV_shape, KV_real_shape);
      // load cached kv
      if (q_seq_idx == 0 && kv_seq_idx < cache_seq_len &&
          attention_mode == 2) { // PAGE_KV_CACHE_PREFILL = 2
        dim4 kvcache_shape = {CORE_NUM, 1, block_kv_secs, block_dkv};
        dim4 kvcache_real_shape = {CORE_NUM, 1, kv_secs, kv_lora_rank};
        auto kvcache_l2_tensor =
            make_l2tensor<TYPE>(kvcache_shape, L2, kvcache_real_shape);

        dim4 kvcache_per_core_real_shape = {1, 1, kv_secs, kv_lora_rank};
        dim4 core_idx_offset = {core_idx, 0, 0, 0};

        dim4 slot_idx_sub_view_shape = {1, 1, kv_secs, 1};
        dim4 slot_idx_sub_view_offset = {0, 0, kv_seq_idx, 0};

        ppl::dma::gather_h(
            kvcache_l2_tensor.sub_view(kvcache_per_core_real_shape,
                                       core_idx_offset),
            kvcache_global_tensor,
            slot_idx_l2_tensor.sub_view(slot_idx_sub_view_shape,
                                        slot_idx_sub_view_offset),
            0);
        ppl::dma::load(KV_tensor, kvcache_l2_tensor
                                      .sub_view(kvcache_per_core_real_shape,
                                                core_idx_offset)
                                      .view(KV_real_shape));
      }

      // load kv
      if (q_seq_idx == 0 && kv_seq_idx >= cache_seq_len) {
        dim4 kv_offset = {0, kv_seq_idx - cache_seq_len, 0, 0};
        ppl::dma::load(KV_tensor,
                       kv_global_tensor.sub_view(KV_real_shape, kv_offset));
      }

      dim4 PE_shape = {1, block_kv_secs, 1, block_dpe};
      dim4 PE_real_shape = {1, kv_secs, 1, qk_rope_head_dim};
      auto PE_tensor = make_tensor<TYPE>(PE_shape, PE_real_shape);
      auto PE_store_tensor = make_tensor<TYPE>(PE_shape, PE_real_shape);

      // load cache PE
      if (kv_seq_idx < cache_seq_len &&
          attention_mode == 2) { // PAGE_KV_CACHE_PREFILL = 2
        dim4 pe_shape = {CORE_NUM, 1, block_kv_secs, block_dpe};
        dim4 pe_real_shape = {CORE_NUM, 1, kv_secs, qk_rope_head_dim};
        auto pe_l2_tensor = make_l2tensor<TYPE>(pe_shape, L2, pe_real_shape);

        dim4 pe_per_core_real_shape = {1, 1, kv_secs, qk_rope_head_dim};
        dim4 core_idx_offset = {core_idx, 0, 0, 0};

        dim4 slot_idx_sub_view_shape = {1, 1, kv_secs, 1};
        dim4 slot_idx_sub_view_offset = {0, 0, kv_seq_idx, 0};

        ppl::dma::gather_h(
            pe_l2_tensor.sub_view(pe_per_core_real_shape, core_idx_offset),
            pecache_global_tensor,
            slot_idx_l2_tensor.sub_view(slot_idx_sub_view_shape,
                                        slot_idx_sub_view_offset),
            0);

        dim4 k_rope_shape = {1, kv_secs, 1, qk_rope_head_dim};
        ppl::dma::load(
            PE_tensor,
            pe_l2_tensor.sub_view(pe_per_core_real_shape, core_idx_offset)
                .view(k_rope_shape));
      }

      dim4 k_rope_shape = {1, block_kv_secs, 1, block_dpe};
      dim4 k_rope_real_shape = {1, kv_secs, 1, qk_rope_head_dim};
      auto KRoPE_cos_tensor =
          make_tensor<TYPE>(k_rope_shape, k_rope_real_shape);
      auto KRoPE_sin_tensor =
          make_tensor<TYPE>(k_rope_shape, k_rope_real_shape);
      // load pe & kRoPE
      if (q_seq_idx == 0 && kv_seq_idx >= cache_seq_len) {
        dim4 PE_offset = {0, kv_seq_idx - cache_seq_len, 0, 0};
        ppl::dma::load(PE_tensor,
                       pe_global_tensor.sub_view(k_rope_real_shape, PE_offset));
        ppl::dma::load(KRoPE_cos_tensor, cos_global_tensor.sub_view(
                                             k_rope_real_shape, PE_offset));
        ppl::dma::load(KRoPE_sin_tensor, sin_global_tensor.sub_view(
                                             k_rope_real_shape, PE_offset));
      }

      if (q_seq_idx > 0 && kv_seq_idx >= cache_seq_len && load_PE) {
        if (attention_mode == 2) {
          dim4 pe_shape = {CORE_NUM, 1, block_kv_secs, block_dpe};
          dim4 pe_real_shape = {CORE_NUM, 1, kv_secs, qk_rope_head_dim};
          auto pe_load_l2_tensor =
              make_l2tensor<TYPE>(pe_shape, L2, pe_real_shape);

          dim4 pe_per_core_real_shape = {1, 1, kv_secs, qk_rope_head_dim};
          dim4 core_idx_offset = {core_idx, 0, 0, 0};

          dim4 slot_idx_sub_view_shape = {1, 1, kv_secs, 1};
          dim4 slot_idx_sub_view_offset = {0, 0, kv_seq_idx, 0};

          ppl::dma::gather_h(
              pe_load_l2_tensor.sub_view(pe_per_core_real_shape,
                                         core_idx_offset),
              pecache_global_tensor,
              slot_idx_l2_tensor.sub_view(slot_idx_sub_view_shape,
                                          slot_idx_sub_view_offset),
              0);

          dim4 k_rope_shape = {1, kv_secs, 1, qk_rope_head_dim};
          ppl::dma::load(PE_tensor,
                         pe_load_l2_tensor
                             .sub_view(pe_per_core_real_shape, core_idx_offset)
                             .view(k_rope_shape));
        } else {
          dim4 PE_offset = {0, kv_seq_idx - cache_seq_len, 0, 0};
          ppl::dma::load(PE_tensor, pe_global_tensor.sub_view(k_rope_real_shape,
                                                              PE_offset));
        }
      }

      dim4 q_shape = {1, block_q_secs, 1, block_dq};
      dim4 q_real_shape = {1, q_secs, 1, q_lora_rank};
      auto Q_tensor = make_tensor<TYPE>(q_shape, q_real_shape);
      dim4 q_rope_shape = {1, block_q_secs, 1, block_dpe};
      dim4 q_rope_real_shape = {1, q_secs, 1, qk_rope_head_dim};
      auto RoPE_cos_tensor = make_tensor<TYPE>(q_rope_shape, q_rope_real_shape);
      auto RoPE_sin_tensor = make_tensor<TYPE>(q_rope_shape, q_rope_real_shape);
      // load q & QRoPE
      if (kv_seq_idx == 0) {
        dim4 q_offset = {0, q_seq_idx, 0, 0};
        ppl::dma::load(Q_tensor,
                       q_global_tensor.sub_view(q_real_shape, q_offset));
        ppl::dma::load(RoPE_cos_tensor,
                       cos_global_tensor.sub_view(q_rope_real_shape, q_offset));
        ppl::dma::load(RoPE_sin_tensor,
                       sin_global_tensor.sub_view(q_rope_real_shape, q_offset));
      }

      dim4 mask_shape = {1, block_q_secs, 1, block_kv_secs};
      dim4 mask_real_shape = {1, q_secs, 1, kv_secs};
      auto mask_tensor = make_tensor<TYPE>(mask_shape, mask_real_shape);
      // load_mask
      if (has_mask) {
        dim4 mask_offset = {0, q_seq_idx + cache_seq_len, 0, kv_seq_idx};
        ppl::dma::load(mask_tensor, mask_global_tensor.sub_view(mask_real_shape,
                                                                mask_offset));
      }

      dim4 KVU_shape = {1, block_kv_secs, 1,
                        block_num_heads / CORE_NUM * (block_dnope + block_dv)};
      dim4 KVU_real_shape = {1, kv_secs, 1,
                             n_heads * (qk_nope_head_dim + v_head_dim)};
      auto KVU_tensor = make_tensor<TYPE>(KVU_shape, KVU_real_shape);
      auto KVU_store_tensor = make_tensor<TYPE>(KVU_shape, KVU_real_shape);

      // load KVU
      if (store_KVU && q_seq_idx > 0) {
        dim4 KVU_per_core_real_shape = {1, kv_secs, n_heads,
                                        qk_nope_head_dim + v_head_dim};
        dim4 kvu_global_offset = {0, kv_seq_idx, 0, 0};
        ppl::dma::load(KVU_tensor,
                       KVU_global_tensor
                           .sub_view(KVU_per_core_real_shape, kvu_global_offset)
                           .view(KVU_real_shape));
      }

      if (q_seq_idx == 0) {
        ppl::tiu::fmm2(KVU_tensor, KV_tensor, wukv_local_tensor, false, true,
                       false);
        ppl::tiu::move(KVU_store_tensor, KVU_tensor);
      }

      K_first = (kv_seq_idx == 0) ? true : false;
      K_last = (kv_seq_idx + kv_secs >= kv_seq) ? true : false;
      Q_first = (q_seq_idx == 0) ? true : false;

      nodechip_deepseekv3_mla_local_prefill(
          Q_tensor, PE_tensor, PE_store_tensor, RoPE_cos_tensor,
          RoPE_sin_tensor, KRoPE_cos_tensor, KRoPE_sin_tensor, wuq_local_tensor,
          mask_tensor, last_max_tensor, softmax_exp_sum_tensor, Y_tensor,
          QU_tensor, Q_rope_tensor, KVU_tensor, C, batch, q_secs, kv_secs,
          n_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
          qk_rope_head_dim, v_head_dim, seq_len, K_first, K_last, has_mask,
          Q_first, kv_seq_idx < cache_seq_len);

      if (q_seq_idx == 0 && core_idx == 0 && kv_seq_idx >= cache_seq_len) {
        if (attention_mode == 2) {
          dim4 kv_global_view_shape = {1, 1, seq_len, kv_lora_rank};
          dim4 kv_sub_shape = {1, 1, kv_secs, kv_lora_rank};
          dim4 kv_sub_offset = {0, 0, kv_seq_idx - cache_seq_len, 0};

          dim4 slot_idx_sub_view_shape = {1, 1, kv_secs, 1};
          dim4 slot_idx_sub_view_offset = {0, 0, kv_seq_idx, 0};

          ppl::dma::scatter_h(
              kvcache_global_tensor,
              kv_global_tensor.view(kv_global_view_shape)
                  .sub_view(kv_sub_shape, kv_sub_offset),
              slot_idx_l2_tensor.sub_view(slot_idx_sub_view_shape,
                                          slot_idx_sub_view_offset));

          auto PE_store_l2_tensor =
              make_l2tensor<TYPE>(k_rope_shape, L2, k_rope_real_shape);
          ppl::dma::store(PE_store_l2_tensor, PE_store_tensor);

          dim4 PE_view_shape = {1, 1, kv_secs, qk_rope_head_dim};
          ppl::dma::scatter_h(
              pecache_global_tensor, PE_store_l2_tensor.view(PE_view_shape),
              slot_idx_l2_tensor.sub_view(slot_idx_sub_view_shape,
                                          slot_idx_sub_view_offset));
        } else {
          dim4 kv_sub_shape = {1, 1, kv_secs, kv_lora_rank};
          dim4 kv_sub_offset = {0, 0, kv_seq_idx - cache_seq_len, 0};
          dim4 kv_offset = {0, kv_seq_idx - cache_seq_len, 0, 0};
          ppl::dma::move(
              kvcache_global_tensor.sub_view(kv_sub_shape, kv_sub_offset)
                  .view(KV_real_shape),
              kv_global_tensor.sub_view(KV_real_shape, kv_offset));

          dim4 pe_sub_shape = {1, 1, kv_secs, qk_rope_head_dim};
          dim4 pe_sub_offset = {0, 0, kv_seq_idx - cache_seq_len, 0};
          ppl::dma::store(
              pecache_global_tensor.sub_view(pe_sub_shape, pe_sub_offset)
                  .view(k_rope_real_shape),
              PE_store_tensor);
        }
      }

      // store KVU to global mem
      if (q_seq_idx == 0 && store_KVU) {
        dim4 KVU_per_core_real_shape = {1, kv_secs, n_heads,
                                        qk_nope_head_dim + v_head_dim};
        dim4 kvu_global_offset = {0, kv_seq_idx, 0, 0};
        ppl::dma::store(
            KVU_global_tensor
                .sub_view(KVU_per_core_real_shape, kvu_global_offset)
                .view(KVU_real_shape),
            KVU_store_tensor);
      }
    }
      // store K_Last时，存储Y
      dim4 q_offset = {0, q_seq_idx, 0, 0};
      ppl::dma::store(y_global_tensor.sub_view(Y_real_shape, q_offset),
                        Y_tensor);
  }
}

template <typename TYPE, typename W_TYPE>
void mla_prefill_multi_core(
    TYPE *Q_global_addr, TYPE *KV_global_addr, TYPE *PE_global_addr,
    TYPE *KVcache_global_addr, TYPE *PEcache_global_addr, TYPE *KVU_global_addr,
    TYPE *RoPE_cos_global_addr, TYPE *RoPE_sin_global_addr,
    W_TYPE *WUQ_global_addr, W_TYPE *WUKV_global_addr,
    TYPE *WUQ_scale_global_addr, TYPE *WUKV_scale_global_addr,
    TYPE *Mask_global_addr, TYPE *Y_global_addr,
    uint32 *block_table_global_addr,
    int max_paged_block_num, // paged block num per batch
    int block_size,          // paged block size
    int n_heads, int q_lora_rank, int kv_lora_rank, int qk_nope_head_dim,
    int qk_rope_head_dim, int v_head_dim, float C, int batch, int mask_max,
    int max_cache_size, int attention_mode, bool has_mask, int quant_block_size,
    int data[MAX_BATCH_SIZE]) {
  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();

  int *input_length = data;
  int *cache_length = input_length + batch;

  dim4 slot_idx_shape = {block_batch, block_max_paged_block_num, 1,
                         block_paged_block_size};
  dim4 slot_idx_real_shape = {batch, max_paged_block_num, 1, block_size};
  auto slot_idx_l2_tensor =
      make_l2tensor<uint32>(slot_idx_shape, L2, slot_idx_real_shape);
// ppl::dma::move(slot_idx_l2_tensor, slot_idx_l2_tensor);
  dim4 block_table_global_shape = {batch, 1, 1, max_paged_block_num};
  auto block_table_global_tensor = gtensor<uint32>(
      block_table_global_shape, GLOBAL, block_table_global_addr);

  if (attention_mode == 2) { // PAGE_KV_CACHE_PREFILL = 2
    generate_slot_idx(slot_idx_l2_tensor, block_table_global_tensor,
                      max_paged_block_num, block_size, batch);
  }

  int heads_per_core = ppl::div_up(n_heads, core_num);
  int heads_secs = ppl::div_up(n_heads, heads_per_core);
  int qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  int kv_head_dim = qk_nope_head_dim + v_head_dim;

  // 针对TP16，需将wuq quant_block_size / 2，分解为多个小block
  int wuq_quant_block_size = quant_block_size;
  int wukv_quant_block_size = quant_block_size;
  dim4 wuq_scale_l2_shape = {
      1,
      2 * block_num_heads * (block_dpe + block_dnope) / block_quant_block_size,
      1, 2 * ppl::div_up(block_dq, block_quant_block_size)};
  dim4 wuq_scale_l2_real_shape = {
      1, 2 * n_heads * qk_head_dim / wuq_quant_block_size, 1,
      2 * ppl::div_up(q_lora_rank, wuq_quant_block_size)};
  auto wuq_scale_l2_tensor =
      make_l2tensor<TYPE>(wuq_scale_l2_shape, L2, wuq_scale_l2_real_shape);

  dim4 WUQ_scale_shape = {
      1, block_num_heads * (block_dpe + block_dnope) / block_quant_block_size,
      1, ppl::div_up(block_dq, block_quant_block_size)};
  dim4 WUQ_scale_real_shape = {1, n_heads * qk_head_dim / wuq_quant_block_size,
                               1,
                               ppl::div_up(q_lora_rank, wuq_quant_block_size)};
  auto WUQ_scale_global_tensor =
      gtensor<TYPE>(WUQ_scale_real_shape, GLOBAL, WUQ_scale_global_addr);
  if ((std::is_same_v<W_TYPE, fp8e5m2> || std::is_same_v<W_TYPE, fp8e4m3>) &&
      (heads_per_core * qk_head_dim) % quant_block_size != 0) {
    if(core_idx == 0) {
        auto wuq_scale_tensor =
            make_tensor<TYPE>(WUQ_scale_shape, WUQ_scale_real_shape);
        ppl::dma::load(wuq_scale_tensor, WUQ_scale_global_tensor);
        dim4 scale_stride;
        scale_stride.w = 2;
        scale_stride.h = WUQ_scale_real_shape.w * scale_stride.w;
        scale_stride.c = WUQ_scale_real_shape.h * scale_stride.h * 2;
        scale_stride.n = WUQ_scale_real_shape.c * scale_stride.c;

        ppl::dma::store(
            wuq_scale_l2_tensor.view(WUQ_scale_real_shape, scale_stride),
            wuq_scale_tensor);
        dim4 offset = {0, 0, 0, 1};
        ppl::dma::store(wuq_scale_l2_tensor.sub_view(WUQ_scale_real_shape, offset)
                            .view(WUQ_scale_real_shape, scale_stride),
                        wuq_scale_tensor);
        offset.w = 0;
        offset.h = 1;
        ppl::dma::store(wuq_scale_l2_tensor.sub_view(WUQ_scale_real_shape, offset)
                            .view(WUQ_scale_real_shape, scale_stride),
                        wuq_scale_tensor);
        offset.w = 1;
        ppl::dma::store(wuq_scale_l2_tensor.sub_view(WUQ_scale_real_shape, offset)
                            .view(WUQ_scale_real_shape, scale_stride),
                        wuq_scale_tensor);
    }
    wuq_quant_block_size = ppl::div_up(wuq_quant_block_size, 2);
  }
  ppl::sync();

  // 确定每个core分配的数据
  // 共享：Q, KV, PE, cos/sin cache, mask
  // 切分：WUQ, WUKV
  if (core_idx < heads_secs) {
    int head_idx = core_idx * heads_per_core;

    // wuq shape: [head * qk_head_dim, q_lora_rank]
    // wukv shape: [head * kv_head_dim, kv_lora_rank]
    // 确定多batch共享的wuq、wukv、wuq_scale、wukv_scale在不同core中位置
    dim4 WUQ_global_shape = {1, n_heads * qk_head_dim, 1, q_lora_rank};
    dim4 WUQ_per_core_global_shape = {1, heads_per_core * qk_head_dim, 1,
                                      q_lora_rank};
    dim4 WUQ_per_core_global_offset = {0, head_idx * qk_head_dim, 0, 0};
    auto WUQ_per_core_global_tensor =
        gtensor<W_TYPE>(WUQ_global_shape, GLOBAL, WUQ_global_addr)
            .sub_view(WUQ_per_core_global_shape, WUQ_per_core_global_offset);

    dim4 WUKV_global_shape = {1, n_heads * kv_head_dim, 1, kv_lora_rank};
    dim4 WUKV_per_core_global_shape = {1, heads_per_core * kv_head_dim, 1,
                                       kv_lora_rank};
    dim4 WUKV_per_core_global_offset = {0, head_idx * kv_head_dim, 0, 0};
    auto WUKV_per_core_global_tensor =
        gtensor<W_TYPE>(WUKV_global_shape, GLOBAL, WUKV_global_addr)
            .sub_view(WUKV_per_core_global_shape, WUKV_per_core_global_offset);

    dim4 WUQ_scale_per_core_global_shape = {
        1, heads_per_core * qk_head_dim / wuq_quant_block_size, 1,
        ppl::div_up(q_lora_rank, wuq_quant_block_size)};
    dim4 WUQ_scale_per_core_global_offset = {
        0, head_idx * qk_head_dim / wuq_quant_block_size, 0, 0};

    dim4 WUKV_scale_global_shape = {
        1, n_heads * kv_head_dim / wukv_quant_block_size, 1,
        ppl::div_up(kv_lora_rank, wukv_quant_block_size)};
    dim4 WUKV_scale_per_core_global_shape = {
        1, heads_per_core * kv_head_dim / wukv_quant_block_size, 1,
        ppl::div_up(kv_lora_rank, wukv_quant_block_size)};
    dim4 WUKV_scale_per_core_global_offset = {
        0, head_idx * kv_head_dim / wukv_quant_block_size, 0, 0};
    auto WUKV_scale_per_core_global_tensor =
        gtensor<TYPE>(WUKV_scale_global_shape, GLOBAL, WUKV_scale_global_addr)
            .sub_view(WUKV_scale_per_core_global_shape,
                      WUKV_scale_per_core_global_offset);

    int total_token = 0;
    for (int bidx = 0; bidx < batch; ++bidx) {
      total_token += input_length[bidx];
    }

    dim4 RoPE_cos_sin_global_shape = {1, total_token, 1, qk_rope_head_dim};
    auto RoPE_cos_global_tensor =
        gtensor<TYPE>(RoPE_cos_sin_global_shape, GLOBAL, RoPE_cos_global_addr);
    auto RoPE_sin_global_tensor =
        gtensor<TYPE>(RoPE_cos_sin_global_shape, GLOBAL, RoPE_sin_global_addr);

    dim4 Q_global_shape = {1, total_token, 1, q_lora_rank};
    auto Q_global_tensor = gtensor<TYPE>(Q_global_shape, GLOBAL, Q_global_addr);

    dim4 KV_global_shape = {1, total_token, 1, kv_lora_rank};
    auto KV_global_tensor =
        gtensor<TYPE>(KV_global_shape, GLOBAL, KV_global_addr);

    dim4 PE_global_shape = {1, total_token, 1, qk_rope_head_dim};
    auto PE_global_tensor =
        gtensor<TYPE>(PE_global_shape, GLOBAL, PE_global_addr);

    dim4 Y_global_shape = {1, total_token, 1, n_heads * v_head_dim};
    auto Y_global_tensor = gtensor<TYPE>(Y_global_shape, GLOBAL, Y_global_addr);

    int token_offset = 0;
    for (int bidx = 0; bidx < batch; ++bidx) {
      int seq_len = input_length[bidx];
      int cache_seq_len = attention_mode == 2 ? cache_length[bidx]
                                              : 0; // PAGE_KV_CACHE_PREFILL = 2
      dim4 KVU_global_shape = {n_heads, seq_len + cache_seq_len, 1,
                               qk_nope_head_dim + v_head_dim};
      dim4 KVU_global_shape_per_core = {heads_per_core, seq_len + cache_seq_len,
                                        1, qk_nope_head_dim + v_head_dim};
      dim4 KVU_global_offset = {head_idx, 0, 0, 0};
      dim4 KVU_global_shape_per_core_view = {1, seq_len + cache_seq_len,
                                             heads_per_core,
                                             qk_nope_head_dim + v_head_dim};
      auto KVU_per_core_global_tensor =
          gtensor<TYPE>(KVU_global_shape, GLOBAL, KVU_global_addr)
              .sub_view(KVU_global_shape_per_core, KVU_global_offset)
              .view(KVU_global_shape_per_core_view);

      dim4 per_core_offset = {0, token_offset, 0, 0};

      dim4 cos_sin_per_core_global_shape = {1, seq_len, 1, qk_rope_head_dim};
      auto cos_per_core_global_tensor = RoPE_cos_global_tensor.sub_view(
          cos_sin_per_core_global_shape, per_core_offset);
      auto sin_per_core_global_tensor = RoPE_sin_global_tensor.sub_view(
          cos_sin_per_core_global_shape, per_core_offset);

      dim4 mask_global_shape = {1, mask_max, 1, mask_max};
      auto mask_per_core_global_tensor =
          gtensor<TYPE>(mask_global_shape, GLOBAL, Mask_global_addr);

      dim4 Q_per_core_global_shape = {1, seq_len, 1, q_lora_rank};
      auto Q_per_core_global_tensor =
          Q_global_tensor.sub_view(Q_per_core_global_shape, per_core_offset);

      dim4 KV_per_core_global_shape = {1, seq_len, 1, kv_lora_rank};
      auto KV_per_core_global_tensor =
          KV_global_tensor.sub_view(KV_per_core_global_shape, per_core_offset);

      dim4 PE_per_core_global_shape = {1, seq_len, 1, qk_rope_head_dim};
      auto PE_per_core_global_tensor =
          PE_global_tensor.sub_view(PE_per_core_global_shape, per_core_offset);

      // KVcache PEcache
      int cache_global_shape_n = batch;
      int cache_global_shape_h = max_cache_size;
      int cache_per_core_offset_n = bidx;
      if (attention_mode == 2) {
        cache_global_shape_n = 1;
        cache_global_shape_h = max_gather_scatter_num;
        cache_per_core_offset_n = 0;
      }
      dim4 KVcache_global_shape = {cache_global_shape_n, 1,
                                   cache_global_shape_h, kv_lora_rank};
      dim4 KVcache_per_core_global_shape = {1, 1, cache_global_shape_h,
                                            kv_lora_rank};
      dim4 cache_per_core_offset = {cache_per_core_offset_n, 0, 0, 0};
      auto KVcache_per_core_global_tensor =
          gtensor<TYPE>(KVcache_global_shape, GLOBAL, KVcache_global_addr)
              .sub_view(KVcache_per_core_global_shape, cache_per_core_offset);

      dim4 PEcache_global_shape = {cache_global_shape_n, 1,
                                   cache_global_shape_h, qk_rope_head_dim};
      dim4 PEcache_per_core_global_shape = {1, 1, cache_global_shape_h,
                                            qk_rope_head_dim};
      auto PEcache_per_core_global_tensor =
          gtensor<TYPE>(PEcache_global_shape, GLOBAL, PEcache_global_addr)
              .sub_view(PEcache_per_core_global_shape, cache_per_core_offset);

      dim4 Y_per_core_global_shape = {1, seq_len, 1,
                                      heads_per_core * v_head_dim};
      dim4 Y_per_core_offset = {0, token_offset, 0,
                                core_idx * heads_per_core * v_head_dim};
      auto Y_per_core_global_tensor =
          Y_global_tensor.sub_view(Y_per_core_global_shape, Y_per_core_offset);

      dim4 slots_idx_per_core_shape = {1, max_paged_block_num, 1, block_size};
      dim4 slots_idx_per_core_offset = {bidx, 0, 0, 0};
      dim4 slots_idx_per_core_view_shape = {
          1, 1, max_paged_block_num * block_size, 1};

      nodechip_deepseekv3_mla_prefill(
          Q_per_core_global_tensor, KV_per_core_global_tensor,
          PE_per_core_global_tensor, KVcache_per_core_global_tensor,
          PEcache_per_core_global_tensor, KVU_per_core_global_tensor,
          cos_per_core_global_tensor, sin_per_core_global_tensor,
          mask_per_core_global_tensor, WUQ_per_core_global_tensor,
          WUKV_per_core_global_tensor,
          WUQ_scale_global_tensor.sub_view(WUQ_scale_per_core_global_shape,
                                           WUQ_scale_per_core_global_offset),
          wuq_scale_l2_tensor.sub_view(WUQ_scale_per_core_global_shape,
                                       WUQ_scale_per_core_global_offset),
          WUKV_scale_per_core_global_tensor, Y_per_core_global_tensor, C, 1,
          bidx, 1, mask_max, seq_len, cache_seq_len, n_heads, heads_per_core,
          q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
          v_head_dim, has_mask, core_idx,
          slot_idx_l2_tensor
              .sub_view(slots_idx_per_core_shape, slots_idx_per_core_offset)
              .view(slots_idx_per_core_view_shape),
          wuq_quant_block_size, wukv_quant_block_size, quant_block_size,
          attention_mode);

      token_offset += seq_len;
    }
  }
}

__KERNEL__ void mla_prefill_bf16(
    bf16 *Q_global_addr, bf16 *KV_global_addr, bf16 *PE_global_addr,
    bf16 *KVcache_global_addr, bf16 *PEcache_global_addr, bf16 *KVU_global_addr,
    bf16 *RoPE_cos_global_addr, bf16 *RoPE_sin_global_addr,
    bf16 *WUQ_global_addr, bf16 *WUKV_global_addr, bf16 *Mask_global_addr,
    bf16 *Y_global_addr, uint32 *block_table_global_addr,
    uint32 *save_slots_global_addr, int max_paged_block_num, int block_size,
    int n_heads, int q_lora_rank, int kv_lora_rank, int qk_nope_head_dim,
    int qk_rope_head_dim, int v_head_dim, float C, bool has_mask, int batch,
    int mask_max, int max_cache_size, int attention_mode,
    int data[MAX_BATCH_SIZE])
{
    ppl::set_core_num(CORE_NUM);
    mla_prefill_multi_core<bf16, bf16>(
        Q_global_addr, KV_global_addr, PE_global_addr, KVcache_global_addr,
        PEcache_global_addr, KVU_global_addr, RoPE_cos_global_addr,
        RoPE_sin_global_addr, WUQ_global_addr, WUKV_global_addr, nullptr,
        nullptr, Mask_global_addr, Y_global_addr, block_table_global_addr,
        max_paged_block_num, block_size, n_heads, q_lora_rank, kv_lora_rank,
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim, C, batch, mask_max,
        max_cache_size, attention_mode, has_mask, 0, data);
}

__KERNEL__ void mla_prefill_bf16_fp8e4m3(
    bf16 *Q_addr, bf16 *KV_addr, bf16 *PE_addr, bf16 *KVcache_addr,
    bf16 *PEcache_addr, bf16 *KVU_addr, bf16 *RoPE_cos_addr,
    bf16 *RoPE_sin_addr, fp8e4m3 *WUQ_addr, fp8e4m3 *WUKV_addr, bf16 *Mask_addr,
    bf16 *Y_addr, bf16 *WUQ_scale_addr, bf16 *WUKV_scale_addr,
    uint32 *block_table_addr, uint32 *save_slots_addr, int num_heads,
    int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim, int q_lora_rank,
    int kv_lora_rank, float softmax_scale, int mask_max, int quant_block_size,
    int max_paged_block_num, int paged_cache_block_size, int max_cache_size,
    int attention_mode, bool has_mask, int batch, int seqlen[MAX_BATCH_SIZE])
{
    ppl::set_core_num(CORE_NUM);
    mla_prefill_multi_core<bf16, fp8e4m3>(
        Q_addr, KV_addr, PE_addr, KVcache_addr, PEcache_addr, KVU_addr,
        RoPE_cos_addr, RoPE_sin_addr, WUQ_addr, WUKV_addr, WUQ_scale_addr,
        WUKV_scale_addr, Mask_addr, Y_addr, block_table_addr,
        max_paged_block_num, paged_cache_block_size, num_heads, q_lora_rank,
        kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
        softmax_scale, batch, mask_max, max_cache_size, attention_mode,
        has_mask, quant_block_size, seqlen);
}

__TEST__ void mla_multi_core_bf16_fp8e4m3_test()
{
    int context_len = 192;
    int cache_len = 0;
    const int paged_cache_block_size = 16;
    const int max_paged_block_num =
        ppl::div_up(context_len, paged_cache_block_size);
    const int num_heads = 16;
    const int q_lora_rank = 1536;
    const int kv_lora_rank = 512;
    const int qk_nope_head_dim = 128;
    const int qk_rope_head_dim = 64;
    const int v_head_dim = 128;
    float softmax_scale = 0.07216878364870322;
    const int batch = 1;
    const int mask_max = context_len + cache_len;
    const int max_cache_size = 65536;
    const int quant_block_size = 128;
    int attention_mode = 2;

    //   int seqlen[32] = {context_len, context_len, context_len, context_len,
    //   context_len, context_len, context_len, context_len,
    //                     context_len, context_len, context_len, context_len,
    //                     context_len, context_len, context_len, context_len,
    //                     0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int seqlen[256] = {context_len, cache_len};

    dim4 Q_shape = {batch, 1, context_len, q_lora_rank};
    dim4 KV_shape = {batch, 1, context_len, kv_lora_rank};
    dim4 PE_shape = {batch, 1, context_len, qk_rope_head_dim};
    dim4 KVcache_shape = {batch, 1, max_cache_size, kv_lora_rank};
    dim4 PEcache_shape = {batch, 1, max_cache_size, qk_rope_head_dim};
    dim4 RoPE_cos_sin_shape = {batch, 1, context_len, qk_rope_head_dim};
    dim4 WUQ_shape = {1, num_heads * (qk_nope_head_dim + qk_rope_head_dim), 1,
                      q_lora_rank};
    dim4 WUKV_shape = {1, num_heads * (qk_nope_head_dim + v_head_dim), 1,
                       kv_lora_rank};
    dim4 Y_shape = {batch, context_len, num_heads, v_head_dim};
    dim4 WUQ_scale_shape = {
        1, 1,
        num_heads * (qk_nope_head_dim + qk_rope_head_dim) / quant_block_size,
        ppl::div_up(q_lora_rank, quant_block_size)};
    dim4 WUKV_scale_shape = {
        1, 1, num_heads * (qk_nope_head_dim + v_head_dim) / quant_block_size,
        ppl::div_up(kv_lora_rank, quant_block_size)};
    dim4 block_table_shape = {batch, 1, 1, max_paged_block_num};
    dim4 save_slots_shape = {batch, 1, 1, max_paged_block_num};
    dim4 mask_shape = {1, mask_max, 1, mask_max};
    dim4 KVU_shape = {context_len + cache_len, 1, num_heads,
                      qk_nope_head_dim + v_head_dim};

    auto Q_addr = ppl::malloc<bf16>(&Q_shape);
    auto KV_addr = ppl::malloc<bf16>(&KV_shape);
    auto PE_addr = ppl::malloc<bf16>(&PE_shape);
    auto KVcache_addr = ppl::malloc<bf16>(&KVcache_shape);
    auto PEcache_addr = ppl::malloc<bf16>(&PEcache_shape);
    auto RoPE_cos_addr = ppl::malloc<bf16>(&RoPE_cos_sin_shape);
    auto RoPE_sin_addr = ppl::malloc<bf16>(&RoPE_cos_sin_shape);
    auto WUQ_addr = ppl::malloc<fp8e4m3>(&WUQ_shape);
    auto WUKV_addr = ppl::malloc<fp8e4m3>(&WUKV_shape);
    auto WUQ_scale_addr = ppl::malloc<bf16>(&WUQ_scale_shape);
    auto WUKV_scale_addr = ppl::malloc<bf16>(&WUKV_scale_shape);
    auto Mask_addr = ppl::malloc<bf16>(&mask_shape);
    auto KVU_addr = ppl::malloc<bf16>(&KVU_shape);
    auto block_table_addr = ppl::malloc<uint32>(&block_table_shape);
    auto save_slots_addr = ppl::malloc<uint32>(&save_slots_shape);

    read_npz(Q_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "Q");
    read_npz(KV_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "KV");
    read_npz(PE_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "PE");
    read_npz(KVcache_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "kv_cache");
    read_npz(PEcache_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "pe_cache");
    read_npz(RoPE_cos_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "cos");
    read_npz(RoPE_sin_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "sin");
    read_npz(WUQ_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "WUQ");
    read_npz(WUKV_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "WUKV");
    read_npz(WUQ_scale_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "WUQ_scale");
    read_npz(WUKV_scale_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "WUKV_scale");
    read_npz(Mask_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "mask");
    read_npz(KVU_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "KVU");
    read_npz(block_table_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "block_tables");
    read_npz(save_slots_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "slots");

    bf16 *Y_addr = rand<bf16>(&Y_shape, -1, 1);

    bool has_mask = true;

    mla_prefill_bf16_fp8e4m3(
        Q_addr, KV_addr, PE_addr, KVcache_addr, PEcache_addr, KVU_addr,
        RoPE_cos_addr, RoPE_sin_addr, WUQ_addr, WUKV_addr, Mask_addr, Y_addr,
        WUQ_scale_addr, WUKV_scale_addr, block_table_addr, save_slots_addr,
        num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, q_lora_rank,
        kv_lora_rank, softmax_scale, mask_max, quant_block_size,
        max_paged_block_num, paged_cache_block_size, max_cache_size,
        attention_mode, has_mask, batch, seqlen);
}

__TEST__ void mla_multi_core_bf16_test()
{
    int context_len = 192;
    int cache_len = 0;
    const int paged_cache_block_size = 16;
    const int max_paged_block_num =
        ppl::div_up(context_len, paged_cache_block_size);
    const int num_heads = 16;
    const int q_lora_rank = 1536;
    const int kv_lora_rank = 512;
    const int qk_nope_head_dim = 128;
    const int qk_rope_head_dim = 64;
    const int v_head_dim = 128;
    float softmax_scale = 0.07216878364870322;
    const int batch = 1;
    const int mask_max = context_len + cache_len;
    const int max_cache_size = 65536;
    const int quant_block_size = 128;
    int attention_mode = 2;

    //   int seqlen[32] = {context_len, context_len, context_len, context_len,
    //   context_len, context_len, context_len, context_len,
    //                     context_len, context_len, context_len, context_len,
    //                     context_len, context_len, context_len, context_len,
    //                     0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int seqlen[256] = {context_len, cache_len};

    dim4 Q_shape = {batch, 1, context_len, q_lora_rank};
    dim4 KV_shape = {batch, 1, context_len, kv_lora_rank};
    dim4 PE_shape = {batch, 1, context_len, qk_rope_head_dim};
    dim4 KVcache_shape = {batch, 1, max_cache_size, kv_lora_rank};
    dim4 PEcache_shape = {batch, 1, max_cache_size, qk_rope_head_dim};
    dim4 RoPE_cos_sin_shape = {batch, 1, context_len, qk_rope_head_dim};
    dim4 WUQ_shape = {1, num_heads * (qk_nope_head_dim + qk_rope_head_dim), 1,
                      q_lora_rank};
    dim4 WUKV_shape = {1, num_heads * (qk_nope_head_dim + v_head_dim), 1,
                       kv_lora_rank};
    dim4 Y_shape = {batch, context_len, num_heads, v_head_dim};
    dim4 WUQ_scale_shape = {
        1, 1,
        num_heads * (qk_nope_head_dim + qk_rope_head_dim) / quant_block_size,
        ppl::div_up(q_lora_rank, quant_block_size)};
    dim4 WUKV_scale_shape = {
        1, 1, num_heads * (qk_nope_head_dim + v_head_dim) / quant_block_size,
        ppl::div_up(kv_lora_rank, quant_block_size)};
    dim4 block_table_shape = {batch, 1, 1, max_paged_block_num};
    dim4 save_slots_shape = {batch, 1, 1, max_paged_block_num};
    dim4 mask_shape = {1, mask_max, 1, mask_max};
    dim4 KVU_shape = {context_len + cache_len, 1, num_heads,
                      qk_nope_head_dim + v_head_dim};

    auto Q_addr = ppl::malloc<bf16>(&Q_shape);
    auto KV_addr = ppl::malloc<bf16>(&KV_shape);
    auto PE_addr = ppl::malloc<bf16>(&PE_shape);
    auto KVcache_addr = ppl::malloc<bf16>(&KVcache_shape);
    auto PEcache_addr = ppl::malloc<bf16>(&PEcache_shape);
    auto RoPE_cos_addr = ppl::malloc<bf16>(&RoPE_cos_sin_shape);
    auto RoPE_sin_addr = ppl::malloc<bf16>(&RoPE_cos_sin_shape);
    auto WUQ_addr = ppl::malloc<bf16>(&WUQ_shape);
    auto WUKV_addr = ppl::malloc<bf16>(&WUKV_shape);
    auto Mask_addr = ppl::malloc<bf16>(&mask_shape);
    auto KVU_addr = ppl::malloc<bf16>(&KVU_shape);
    auto block_table_addr = ppl::malloc<uint32>(&block_table_shape);
    auto save_slots_addr = ppl::malloc<uint32>(&save_slots_shape);

    read_npz(Q_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "Q");
    read_npz(KV_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "KV");
    read_npz(PE_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "PE");
    read_npz(KVcache_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "kv_cache");
    read_npz(PEcache_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "pe_cache");
    read_npz(RoPE_cos_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "cos");
    read_npz(RoPE_sin_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "sin");
    read_npz(WUQ_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "WUQ");
    read_npz(WUKV_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "WUKV");
    read_npz(Mask_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "mask");
    read_npz(KVU_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "KVU");
    read_npz(block_table_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "block_tables");
    read_npz(save_slots_addr,
             "/workspace/ppl_dev/ppl/samples/deepseek_mla/mla_prefill_v2.npz",
             "slots");

    bf16 *Y_addr = rand<bf16>(&Y_shape, -1, 1);

    bool has_mask = true;

    mla_prefill_bf16(Q_addr, KV_addr, PE_addr, KVcache_addr, PEcache_addr,
                    KVU_addr, RoPE_cos_addr, RoPE_sin_addr, WUQ_addr, WUKV_addr,
                    Mask_addr, Y_addr, block_table_addr, save_slots_addr,
                    max_paged_block_num, paged_cache_block_size, num_heads,
                    q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                    qk_rope_head_dim, v_head_dim, softmax_scale, has_mask,
                    batch, mask_max, max_cache_size, attention_mode, seqlen);
}