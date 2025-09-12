#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

#ifdef __bm1690__
#define CORE_NUM 8
#elif defined(__sg2260e__)
#define CORE_NUM 4
#endif
#define MAX_BATCH_SIZE 512

using QK_TYPE = fp32;
// Define the attention modes
// CONTINOUS_KV_CACHE_PREFILL = 0
// CONTINOUS_KV_CACHE_DECODE = 1
// PAGE_KV_CACHE_PREFILL = 2
// PAGE_KV_CACHE_DECODE = 3

#ifdef __bm1690__
const int block_batch = 256;
const int block_num_heads = 16;
#elif defined(__sg2260e__)
const int block_batch = 256;
const int block_num_heads = 8;
#endif

const int block_dq = 1536;
const int block_dkv = 512;
const int block_dnope = 128;
const int block_dpe = 64;
const int block_dv = 128;
const int block_tiled_seqlen_k = 2048;
const int max_gather_scatter_num = 17000000;
const int block_paged_block_size = 16;
const int block_quant_block_size = 128;

int get_max_fetch_tokens(int input_length[MAX_BATCH_SIZE], int batch)
{
    int seqlen_k_cache = 0;

    for (int i = 0; i < batch; i++) {
        int fetch_tokens = input_length[i];
        seqlen_k_cache = ppl::max(seqlen_k_cache, fetch_tokens);
    }

    return seqlen_k_cache;
}

template <typename TYPE>
void load_cache(TYPE *cache_gaddr, gtensor<TYPE> &gathered_global_tensor,
                gtensor<uint32> &index_global_tensor, int block_size,
                int block_dim, int seqlen)
{
    int block_num = seqlen / block_size;
    int tail = seqlen % block_size;
    if (block_num > 0) {
        dim4 cache_global_shape = {1, 1, max_gather_scatter_num,
                                   block_size * block_dim};
        auto cache_global_tensor =
            gtensor<TYPE>(cache_global_shape, GLOBAL, cache_gaddr);
        dim4 index_global_shape = {1, 1, block_num, 1};
        dim4 gathered_global_shape = {1, 1, block_num, block_size * block_dim};
        ppl::sdma::gather_h(gathered_global_tensor.view(gathered_global_shape),
                            cache_global_tensor,
                            index_global_tensor.view(index_global_shape), 0);
    }
    if (tail > 0) {
        dim4 cache_global_shape = {1, 1, max_gather_scatter_num, block_dim};
        auto cache_global_tensor =
            gtensor<TYPE>(cache_global_shape, GLOBAL, cache_gaddr);
        dim4 index_shape = {1, 1, 1, block_paged_block_size};
        dim4 index_real_shape = {1, 1, 1, tail};
        auto index_tensor = make_tensor<uint32>(index_shape, index_real_shape);
        arange_broadcast(index_tensor, 1, 0, 1, tail);
        dim4 block_shape = {1, 1, 1, 1};
        auto block_tensor = tensor<uint32>(block_shape);
        dim4 index_global_offset = {0, 0, 0, block_num};
        ppl::dma::load(block_tensor, index_global_tensor.sub_view(
                                         block_shape, index_global_offset));
        ppl::tiu::mul(block_tensor, block_tensor, block_size);
        ppl::tiu::add(index_tensor, index_tensor, block_tensor);
        index_real_shape.h = tail;
        index_real_shape.w = 1;

        dim4 gathered_global_offset = {0, 0, block_num * block_size, 0};
        dim4 gathered_global_sub_shape = {1, 1, tail, block_dim};
        ppl::dma::gather_h(
            gathered_global_tensor.sub_view(gathered_global_sub_shape,
                                            gathered_global_offset),
            cache_global_tensor, index_tensor.view(index_real_shape), 0);
    }
}

template <typename TYPE>
void attn_rope_local(tensor<TYPE> &output_tensor, tensor<TYPE> &input_tensor,
                     tensor<TYPE> &cos_tensor, tensor<TYPE> &sin_tensor,
                     tensor<TYPE> &buffer_tensor,
                     tensor<TYPE> &cos_sin_buffer_tensor, int c, int w)
{
    dim4 input_shape = {1, c, 1, w};
    dim4 input_stride;
    ppl::aligned_stride_4d(&input_stride, &input_shape, 0, sizeof(TYPE));
    dim4 src_stride = {input_stride.n, input_stride.c, input_stride.h, 2};
    dim4 neg_shape = {1, c, 1, w / 2};
    dim4 offset = {0, 0, 0, 0};
    dim4 src_offset = {0, 0, 0, 1};
    dim4 dst_offset = {0, 0, 0, w / 2};
    // chose the even and odd elements, and then multiply the cos value
    ppl::tiu::move(output_tensor.sub_view(neg_shape, offset),
                   input_tensor.view(neg_shape, src_stride));
    ppl::tiu::move(output_tensor.sub_view(neg_shape, dst_offset),
                   input_tensor.sub_view(neg_shape, src_offset)
                       .view(neg_shape, src_stride));
    ppl::tiu::broadcast(cos_sin_buffer_tensor, cos_tensor);
    ppl::tiu::fmul(input_tensor, output_tensor, cos_sin_buffer_tensor);

    // swap the even and odd elements, and then multiply the sin value
    ppl::tiu::fmul(buffer_tensor.sub_view(neg_shape, offset),
                   output_tensor.sub_view(neg_shape, dst_offset), -1.0);
    ppl::tiu::move(buffer_tensor.sub_view(neg_shape, dst_offset),
                   output_tensor.sub_view(neg_shape, offset));
    ppl::tiu::broadcast(cos_sin_buffer_tensor, sin_tensor);
    ppl::tiu::fmul(output_tensor, buffer_tensor, cos_sin_buffer_tensor);

    // add
    ppl::tiu::fadd(output_tensor, input_tensor, output_tensor);
}

template <typename TYPE, typename QK_TYPE>
void attn_softmax_local(tensor<TYPE> &QK_dtype_tensor,
                        tensor<QK_TYPE> &QK_buffer_tensor,
                        tensor<QK_TYPE> &softmax_last_max_val_tensor,
                        tensor<TYPE> &softmax_exp_sum_tensor, int num_heads,
                        int seq_len)
{
    // rowmax(S)
    dim4 QK_shape = {1, block_num_heads, 1, block_tiled_seqlen_k + 1};
    dim4 QK_real_shape = {1, num_heads, 1, seq_len};
    quick_pooling(softmax_last_max_val_tensor, QK_buffer_tensor, &QK_shape,
                  &QK_real_shape, -15000, 0);

    // S2 - m2
    ppl::tiu::fsub(QK_buffer_tensor, QK_buffer_tensor,
                   softmax_last_max_val_tensor);

    // QK_dtype cast to dtype
    ppl::tiu::cast(QK_dtype_tensor, QK_buffer_tensor);

    // exp(S2-m2)
    exp_no_overflow(QK_dtype_tensor, QK_dtype_tensor, &QK_shape,
                    &QK_real_shape);

    // rowsum(exp(S2-m2))
    quick_pooling(softmax_exp_sum_tensor, QK_dtype_tensor, &QK_shape,
                  &QK_real_shape, 0, 1);
}

template <typename TYPE, typename QK_TYPE>
void attn_online_softmax_local(tensor<TYPE> &QK_dtype_tensor,
                               tensor<QK_TYPE> &QK_buffer_tensor,
                               tensor<QK_TYPE> &softmax_last_max_val_tensor,
                               tensor<TYPE> &softmax_cur_max_val_tensor,
                               tensor<TYPE> &softmax_exp_sum_tensor,
                               tensor<TYPE> &QKV_flash_res_tensor,
                               int num_heads, int seq_len)
{
    // rowmax(S)
    dim4 QK_shape = {1, block_num_heads, 1, block_tiled_seqlen_k + 1};
    dim4 QK_real_shape = {1, num_heads, 1, seq_len};
    dim4 softmax_shape = {1, block_num_heads, 1, 1};
    dim4 softmax_real_shape = {1, num_heads, 1, 1};
    auto softmax_max_buffer_tensor =
        make_tensor<QK_TYPE>(softmax_shape, softmax_real_shape);
    quick_pooling(softmax_max_buffer_tensor, QK_buffer_tensor, &QK_shape,
                  &QK_real_shape, -15000, 0);

    // max_val : m2 = max(m1, rowmax(S2))
    ppl::tiu::fmax(softmax_max_buffer_tensor, softmax_last_max_val_tensor,
                   softmax_max_buffer_tensor);

    // m1 - m2
    auto buffer0_tensor =
        make_tensor<QK_TYPE>(softmax_shape, softmax_real_shape);
    ppl::tiu::fsub(buffer0_tensor, softmax_last_max_val_tensor,
                   softmax_max_buffer_tensor);

    // copy m2 to last_max_val
    ppl::tiu::move(softmax_last_max_val_tensor, softmax_max_buffer_tensor);

    // S2 - m2
    ppl::tiu::fsub(QK_buffer_tensor, QK_buffer_tensor,
                   softmax_max_buffer_tensor);

    // QK_dtype cast to dtype
    auto buffer1_tensor = make_tensor<TYPE>(softmax_shape, softmax_real_shape);
    ppl::tiu::cast(buffer1_tensor, buffer0_tensor);

    // exp(m1 - m2)
    exp_no_overflow(softmax_cur_max_val_tensor, buffer1_tensor, &softmax_shape,
                    &softmax_real_shape);

    // cast QK_dtype to dtype
    ppl::tiu::cast(QK_dtype_tensor, QK_buffer_tensor);

    // exp(S2-m2)
    exp_no_overflow(QK_dtype_tensor, QK_dtype_tensor, &QK_shape,
                    &QK_real_shape);

    // l = rowsum(exp(S2-m2))
    auto buffer3_tensor = make_tensor<TYPE>(softmax_shape, softmax_real_shape);
    quick_pooling(buffer3_tensor, QK_dtype_tensor, &QK_shape, &QK_real_shape, 0,
                  1);

    // exp(m1 - m2) *l1
    ppl::tiu::fmul(softmax_exp_sum_tensor, softmax_exp_sum_tensor,
                   softmax_cur_max_val_tensor);

    // l2 = exp(m1 - m2) * l1 + rowsum(exp(S2 - m2))
    ppl::tiu::fadd(softmax_exp_sum_tensor, softmax_exp_sum_tensor,
                   buffer3_tensor);

    // exp(m1 - m2)*O1
    ppl::tiu::fmul(QKV_flash_res_tensor, QKV_flash_res_tensor,
                   softmax_cur_max_val_tensor);
}

template <typename TYPE, typename QK_TYPE>
void attn_online_multicore_softmax_local(
    tensor<QK_TYPE> &qk_multicore_tensor, tensor<TYPE> &QK_dtype_tensor,
    tensor<QK_TYPE> &softmax_last_max_val_tensor,
    tensor<TYPE> &softmax_cur_max_val_tensor,
    tensor<TYPE> &softmax_exp_sum_tensor, tensor<TYPE> &QKV_flash_res_tensor,
    int num_heads, int seq_len)
{
    dim4 softmax_shape = {1, block_num_heads, 1, 1};
    dim4 softmax_real_shape = {1, num_heads, 1, 1};
    auto softmax_max_buffer_tensor =
        make_tensor<QK_TYPE>(softmax_shape, softmax_real_shape);

    // max_val : m2 = max(m1, rowmax(S2))
    ppl::tiu::max(softmax_max_buffer_tensor, softmax_last_max_val_tensor,
                  qk_multicore_tensor);

    // m1 - m2
    ppl::tiu::fsub(qk_multicore_tensor, softmax_last_max_val_tensor,
                   softmax_max_buffer_tensor);

    // copy m2 to last_max_val
    ppl::tiu::move(softmax_last_max_val_tensor, softmax_max_buffer_tensor);

    // cast max value from QK_dtype to dtype
    ppl::tiu::cast(softmax_cur_max_val_tensor, qk_multicore_tensor);

    // exp(m1 - m2)
    exp_no_overflow(softmax_cur_max_val_tensor, softmax_cur_max_val_tensor,
                    &softmax_shape, &softmax_real_shape);

    // exp(m1 - m2) * S1
    ppl::tiu::fmul(QK_dtype_tensor, QK_dtype_tensor,
                   softmax_cur_max_val_tensor);

    // exp(m1 - m2) *l1
    ppl::tiu::fmul(softmax_exp_sum_tensor, softmax_exp_sum_tensor,
                   softmax_cur_max_val_tensor);

    // exp(m1 - m2)*O1
    ppl::tiu::fmul(QKV_flash_res_tensor, QKV_flash_res_tensor,
                   softmax_cur_max_val_tensor);
}

template <typename TYPE, typename W_TYPE>
void load_upper_weight(W_TYPE *WUQ_global_addr, W_TYPE *WUKV_global_addr,
                       TYPE *WUQ_scale_gaddr, TYPE *WUKV_scale_gaddr,
                       TYPE *Q_global_addr,
                       gtensor<TYPE> &Q_rope_batch_l2_tensor,
                       gtensor<TYPE> &Q_WUKV_batch_l2_tensor,
                       tensor<TYPE> &WUQ_tensor, tensor<TYPE> &WUKV_K_tensor,
                       tensor<TYPE> &WUKV_V_tensor, int dq, int dkv, int dqu,
                       int dkvu, int dnope, int dv, int quant_block_size,
                       int cur_num_heads, int num_heads, int WU_head_tile,
                       int core_idx, int batch)
{
    const int core_num = ppl::get_core_num();
    int dtype_eu_num = ppl::get_eu_num<TYPE>();

    if (std::is_same_v<W_TYPE, fp8e5m2> || std::is_same_v<W_TYPE, fp8e4m3>) {
        // gdma load WUKV
        dim4 WUKV_global_shape = {1, num_heads * dkvu, 1, dkv};
        auto WUKV_global_tensor =
            gtensor<W_TYPE>(WUKV_global_shape, GLOBAL, WUKV_global_addr);

        dim4 WUKV_global_offset = {0, core_idx * WU_head_tile * dkvu, 0, 0};
        dim4 WUKV_shape = {
            1, block_num_heads / core_num * (block_dnope + block_dv), 1,
            block_dkv};
        dim4 WUKV_real_shape = {1, cur_num_heads * dkvu, 1, dkv};
        auto WUKV_tensor = make_tensor<W_TYPE>(WUKV_shape, WUKV_real_shape);
        ppl::dma::load(WUKV_tensor, WUKV_global_tensor.sub_view(
                                        WUKV_real_shape, WUKV_global_offset));

        ppl::parallel_start();

        // gdma load Q_lora
        dim4 Q_lora_shape = {1, block_batch, 1, block_dq};
        dim4 Q_lora_real_shape = {1, batch, 1, dq};
        auto Q_lora_global_tensor =
            gtensor<TYPE>(Q_lora_real_shape, GLOBAL, Q_global_addr);
        auto Q_lora_tensor = make_tensor<TYPE>(Q_lora_shape, Q_lora_real_shape);
        ppl::dma::load(Q_lora_tensor, Q_lora_global_tensor);

        dim4 WUKV_cw_shape = {
            1, block_dkv, 1,
            block_num_heads / core_num * (block_dnope + block_dv)};
        dim4 WUKV_cw_real_shape = {1, dkv, 1, cur_num_heads * dkvu};
        auto WUKV_cw_tensor =
            make_tensor<W_TYPE>(WUKV_cw_shape, WUKV_cw_real_shape);
        // (1, cur_num_heads * dkvu, 1, dkv) -> cw_trans (1, dkv, 1,
        // cur_num_heads * dkvu)
        ppl::tiu::transpose_wc(WUKV_cw_tensor, WUKV_tensor);

        auto WUKV_cw_tensor_dtype =
            make_tensor<TYPE>(WUKV_cw_shape, WUKV_cw_real_shape);

        // load WUKV_scale
        ppl::assert(quant_block_size % NPU_NUM == 0 &&
                    WUKV_cw_real_shape.c % quant_block_size == 0 &&
                    WUKV_cw_real_shape.w % quant_block_size == 0);

        dim4 WUKV_scale_shape = {
            1, 1,
            block_num_heads / CORE_NUM * (block_dnope + block_dv) /
                block_quant_block_size,
            ppl::div_up(block_dkv, block_quant_block_size)};
        dim4 WUKV_scale_real_shape = {1, 1,
                                      cur_num_heads * dkvu / quant_block_size,
                                      dkv / quant_block_size};
        auto WUKV_scale_tensor =
            make_tensor<TYPE>(WUKV_scale_shape, WUKV_scale_real_shape);

        dim4 WUKV_scale_global_shape = {
            1, 1, num_heads * dkvu / quant_block_size, dkv / quant_block_size};
        auto WUKV_scale_global_tensor =
            gtensor<TYPE>(WUKV_scale_global_shape, GLOBAL, WUKV_scale_gaddr);
        dim4 WUKV_scale_global_offset = {
            0, 0, core_idx * WU_head_tile * dkvu / quant_block_size, 0};
        ppl::dma::load(WUKV_scale_tensor,
                       WUKV_scale_global_tensor.sub_view(
                           WUKV_scale_real_shape, WUKV_scale_global_offset));
        // fp8 to fp16/bf16
        auto buffer_tensor =
            make_tensor<TYPE>(WUKV_cw_shape, WUKV_cw_real_shape);
        ppl::tiu::cast(buffer_tensor, WUKV_cw_tensor);
        ppl::parallel_end();

        ppl::parallel_start();
        // scale_shape: {dkv / quant_block_size, npu_num, 1, cur_num_heads *
        // dkvu / quant_block_size} in lmem
        dim4 WUKV_scale_trans_shape = {
            ppl::div_up(block_dkv, block_quant_block_size), 1, 1,
            block_num_heads / CORE_NUM * (block_dnope + block_dv) /
                block_quant_block_size};
        dim4 WUKV_scale_trans_real_shape = {
            dkv / quant_block_size, 1, 1,
            cur_num_heads * dkvu / quant_block_size};
        auto WUKV_scale_trans_tensor = make_tensor<TYPE>(
            WUKV_scale_trans_shape, WUKV_scale_trans_real_shape);

        int scale_lstride_c = ppl::align(WUKV_scale_real_shape.h, dtype_eu_num);
        dim4 scale_lstride = {1, scale_lstride_c, 1, WUKV_scale_real_shape.w};
        ppl::tiu::move(
            WUKV_scale_trans_tensor,
            WUKV_scale_tensor.view(WUKV_scale_trans_real_shape, scale_lstride));

        // bcast
        dim4 WUKV_scale_trans_shape_bacst = {
            ppl::div_up(block_dkv, block_quant_block_size), NPU_NUM, 1,
            block_num_heads / CORE_NUM * (block_dnope + block_dv) /
                block_quant_block_size};
        dim4 WUKV_scale_trans_real_shape_bacst = {
            dkv / quant_block_size, NPU_NUM, 1,
            cur_num_heads * dkvu / quant_block_size};
        auto WUKV_scale_trans_tensor_bacst = make_tensor<TYPE>(
            WUKV_scale_trans_shape_bacst, WUKV_scale_trans_real_shape_bacst);
        ppl::tiu::broadcast(WUKV_scale_trans_tensor_bacst,
                            WUKV_scale_trans_tensor);

        dim4 mul_shape = {WUKV_scale_trans_real_shape_bacst.n, quant_block_size,
                          cur_num_heads * dkvu / quant_block_size,
                          quant_block_size};
        int scale_lstride_bcast_n =
            ppl::align(WUKV_scale_trans_real_shape_bacst.h *
                           WUKV_scale_trans_real_shape_bacst.w,
                       dtype_eu_num);
        dim4 scale_lstride_bcast = {scale_lstride_bcast_n, 0, 1, 0};

        ppl::tiu::fmul(
            WUKV_cw_tensor_dtype.view(mul_shape),
            WUKV_scale_trans_tensor_bacst.view(mul_shape, scale_lstride_bcast),
            buffer_tensor.view(mul_shape));

        int dkvu_align = ppl::align(cur_num_heads * dkvu, dtype_eu_num);

        dim4 WUKV_cw_view_shape = {1, dkv, cur_num_heads, dkvu};
        dim4 WUKV_cw_v_sub_view_shape = {1, dkv, cur_num_heads, dv};
        dim4 WUKV_cw_k_sub_view_shape = {1, dkv, cur_num_heads, dnope};
        dim4 WUKV_cw_v_offset = {0, 0, 0, dnope};
        dim4 WUKV_cw_k_offset = {0, 0, 0, 0};
        dim4 V_stride = {dkvu, dkvu_align, dv, 1};
        dim4 K_stride = {dkvu, dkvu_align, dnope, 1};

        dim4 WUKV_V_shape = {block_num_heads / core_num, block_dkv, 1,
                             block_dv};
        dim4 WUKV_V_real_shape = {cur_num_heads, dkv, 1, dv};
        dim4 WUKV_K_shape = {block_num_heads / core_num, block_dkv, 1,
                             block_dnope};
        dim4 WUKV_K_real_shape = {cur_num_heads, dkv, 1, dnope};
        auto WUKV_V_buffer_tensor =
            make_tensor<TYPE>(WUKV_V_shape, WUKV_V_real_shape);
        // WUKV_cw_tensor (1, dkv, 1, cur_num_heads * dkvu) -> view (1, dkv,
        // cur_num_heads, dkvu) -> sub_view (1, dkv, cur_num_heads, dv) -> view
        // (cur_num_heads, dkv, 1, dv)

        ppl::tiu::move(WUKV_V_buffer_tensor,
                       WUKV_cw_tensor_dtype.view(WUKV_cw_view_shape)
                           .sub_view(WUKV_cw_v_sub_view_shape, WUKV_cw_v_offset)
                           .view(WUKV_V_real_shape, V_stride));

        // (cur_num_heads, dkv, 1, dv) -> cw_trans (cur_num_heads, dv, 1, dkv)
        ppl::tiu::transpose_wc(WUKV_V_tensor, WUKV_V_buffer_tensor);
        // WUKV_cw_tensor (1, dkv, 1, cur_num_heads * dkvu) -> view (1, dkv,
        // cur_num_heads, dkvu) -> sub_view (1, dkv, cur_num_heads, dnope) ->
        // view (cur_num_heads, dkv, 1, dnope)

        ppl::tiu::move(WUKV_K_tensor,
                       WUKV_cw_tensor_dtype.view(WUKV_cw_view_shape)
                           .sub_view(WUKV_cw_k_sub_view_shape, WUKV_cw_k_offset)
                           .view(WUKV_K_real_shape, K_stride));

        // gdma load WUQ
        dim4 WUQ_global_shape = {1, num_heads * dqu, 1, dq};
        auto WUQ_global_tensor =
            gtensor<W_TYPE>(WUQ_global_shape, GLOBAL, WUQ_global_addr);
        dim4 WUQ_global_offset = {0, core_idx * WU_head_tile * dqu, 0, 0};
        dim4 WUQ_shape = {
            1, block_num_heads / core_num * (block_dnope + block_dpe), 1,
            block_dq};
        dim4 WUQ_real_shape = {1, cur_num_heads * dqu, 1, dq};

        ppl::assert(quant_block_size % NPU_NUM == 0 &&
                    dq % quant_block_size == 0);

        dim4 WUQ_scale_shape = {
            block_num_heads / CORE_NUM * (block_dnope + block_dpe) /
                    block_quant_block_size +
                2,
            NPU_NUM, ppl::div_up(block_dq, block_quant_block_size), 1};
        dim4 WUQ_scale_real_shape = {cur_num_heads * dqu / quant_block_size + 2,
                                     NPU_NUM, ppl::div_up(dq, quant_block_size),
                                     1};
        auto WUQ_scale_tensor =
            make_tensor<TYPE>(WUQ_scale_shape, WUQ_scale_real_shape);

        dim4 WUQ_scale_global_shape = {num_heads * dqu / quant_block_size, 1,
                                       ppl::div_up(dq, quant_block_size), 1};
        auto WUQ_scale_global_tensor =
            gtensor<TYPE>(WUQ_scale_global_shape, GLOBAL, WUQ_scale_gaddr);

        int dqu_remain = quant_block_size -
                         (core_idx * WU_head_tile * dqu) % quant_block_size;

        dim4 WUQ_scale_global_shape0 = {1, 1, ppl::div_up(dq, quant_block_size),
                                        1};
        dim4 WUQ_scale_global_offset0 = {
            core_idx * WU_head_tile * dqu / quant_block_size, 0, 0, 0};
        dim4 WUQ_scale_real_shape0 = {1, NPU_NUM,
                                      ppl::div_up(dq, quant_block_size), 1};
        dim4 WUQ_scale_offset0 = {0, 0, 0, 0};
        int WUQ_scale_real_shape0_n = WUQ_scale_real_shape0.n;
        if (dqu_remain < quant_block_size) {
            ppl::dma::load_broadcast(
                WUQ_scale_tensor.sub_view(WUQ_scale_real_shape0,
                                          WUQ_scale_offset0),
                WUQ_scale_global_tensor.sub_view(WUQ_scale_global_shape0,
                                                 WUQ_scale_global_offset0));
        } else {
            dqu_remain = 0;
            WUQ_scale_real_shape0_n = 0;
        }

        dim4 WUQ_scale_global_shape1 = {
            (cur_num_heads * dqu - dqu_remain) / quant_block_size, 1,
            ppl::div_up(dq, quant_block_size), 1};
        dim4 WUQ_scale_global_offset1 = {
            WUQ_scale_global_offset0.n + WUQ_scale_real_shape0_n, 0, 0, 0};
        dim4 WUQ_scale_real_shape1 = {
            (cur_num_heads * dqu - dqu_remain) / quant_block_size, NPU_NUM,
            ppl::div_up(dq, quant_block_size), 1};
        dim4 WUQ_scale_offset1 = {WUQ_scale_offset0.n + WUQ_scale_real_shape0_n,
                                  0, 0, 0};
        ppl::dma::load_broadcast(
            WUQ_scale_tensor.sub_view(WUQ_scale_real_shape1, WUQ_scale_offset1),
            WUQ_scale_global_tensor.sub_view(WUQ_scale_global_shape1,
                                             WUQ_scale_global_offset1));

        dim4 WUQ_scale_global_shape2 = {1, 1, ppl::div_up(dq, quant_block_size),
                                        1};
        dim4 WUQ_scale_global_offset2 = {
            WUQ_scale_global_offset1.n + WUQ_scale_real_shape1.n, 0, 0, 0};
        dim4 WUQ_scale_real_shape2 = {1, NPU_NUM,
                                      ppl::div_up(dq, quant_block_size), 1};
        dim4 WUQ_scale_offset2 = {WUQ_scale_offset1.n + WUQ_scale_real_shape1.n,
                                  0, 0, 0};
        if ((cur_num_heads * dqu - dqu_remain) % quant_block_size != 0) {
            ppl::dma::load_broadcast(
                WUQ_scale_tensor.sub_view(WUQ_scale_real_shape2,
                                          WUQ_scale_offset2),
                WUQ_scale_global_tensor.sub_view(WUQ_scale_global_shape2,
                                                 WUQ_scale_global_offset2));
        }

        // load WUQ
        auto WUQ_tensor_w_type = make_tensor<W_TYPE>(WUQ_shape, WUQ_real_shape);
        ppl::dma::load(
            WUQ_tensor_w_type,
            WUQ_global_tensor.sub_view(WUQ_real_shape, WUQ_global_offset));
        ppl::parallel_end();
        
        ppl::parallel_start();
        // fp8 to fp16/bf16
        auto buffer_tensor1 = make_tensor<TYPE>(WUQ_shape, WUQ_real_shape);
        ppl::tiu::cast(buffer_tensor1, WUQ_tensor_w_type);

        if (dqu_remain > 0) {
            dim4 WUQ_scale_mul_shape0 = {WUQ_scale_real_shape0.n, dqu_remain,
                                         WUQ_scale_real_shape0.h,
                                         quant_block_size};
            int WUQ_scale_lstride_n =
                ppl::align(WUQ_scale_real_shape0.h, dtype_eu_num);
            dim4 WUQ_scale_lstride = {WUQ_scale_lstride_n, 0, 1, 0};
            ppl::tiu::fmul(
                WUQ_tensor.view(WUQ_scale_mul_shape0),
                WUQ_scale_tensor.view(WUQ_scale_real_shape0, WUQ_scale_lstride),
                buffer_tensor1.view(WUQ_scale_mul_shape0));
        }

        dim4 WUQ_scale_mul_shape1 = {WUQ_scale_real_shape1.n, quant_block_size,
                                     WUQ_scale_real_shape1.h, quant_block_size};
        int WUQ_scale_lstride_n =
            ppl::align(WUQ_scale_real_shape1.h, dtype_eu_num);
        dim4 WUQ_scale_lstride = {WUQ_scale_lstride_n, 0, 1, 0};
        dim4 WUQ_sub_offset = {0, dqu_remain, 0, 0};
        dim4 WUQ_sub_shape = {1, WUQ_scale_real_shape1.n * quant_block_size, 1,
                              WUQ_scale_real_shape1.h * quant_block_size};
        ppl::tiu::fmul(
            WUQ_tensor.sub_view(WUQ_sub_shape, WUQ_sub_offset)
                .view(WUQ_scale_mul_shape1),
            WUQ_scale_tensor.sub_view(WUQ_scale_real_shape1, WUQ_scale_offset1)
                .view(WUQ_scale_mul_shape1, WUQ_scale_lstride),
            buffer_tensor1.sub_view(WUQ_sub_shape, WUQ_sub_offset)
                .view(WUQ_scale_mul_shape1));

        if ((cur_num_heads * dqu - dqu_remain) % quant_block_size != 0) {
            dim4 WUQ_scale_mul_shape2 = {
                WUQ_scale_real_shape2.n,
                (cur_num_heads * dqu - dqu_remain) % quant_block_size,
                WUQ_scale_real_shape2.h, quant_block_size};
            int WUQ_scale_lstride_n =
                ppl::align(WUQ_scale_real_shape2.h, dtype_eu_num);
            dim4 WUQ_scale_lstride = {WUQ_scale_lstride_n, 0, 1, 0};

            dim4 WUQ_sub_offset = {
                0,
                cur_num_heads * dqu -
                    (cur_num_heads * dqu - dqu_remain) % quant_block_size,
                0, 0};
            dim4 WUQ_sub_shape = {
                1, (cur_num_heads * dqu - dqu_remain) % quant_block_size, 1,
                WUQ_scale_real_shape2.h * quant_block_size};

            ppl::tiu::fmul(
                WUQ_tensor.sub_view(WUQ_sub_shape, WUQ_sub_offset)
                    .view(WUQ_scale_mul_shape2),
                WUQ_scale_tensor
                    .sub_view(WUQ_scale_real_shape2, WUQ_scale_offset2)
                    .view(WUQ_scale_mul_shape2, WUQ_scale_lstride),
                buffer_tensor1.sub_view(WUQ_sub_shape, WUQ_sub_offset)
                    .view(WUQ_scale_mul_shape2));
        }
        ppl::parallel_end();

        // calculate Q_upper
        // (1, batch, 1, dq) * (1, cur_num_heads * dqu, 1, dq) -> (1, batch, 1,
        // cur_num_heads * dqu)
        dim4 Q_upper_shape = {
            1, block_batch, 1,
            block_num_heads / core_num * (block_dnope + block_dpe)};
        dim4 Q_upper_real_shape = {1, batch, 1, cur_num_heads * dqu};
        auto Q_upper_tensor =
            make_tensor<TYPE>(Q_upper_shape, Q_upper_real_shape);
        ppl::tiu::fmm2(Q_upper_tensor, Q_lora_tensor, WUQ_tensor, false, true,
                       false);

        // cpy Q_rope from Q_upper
        // Q_rope: (1, batch, cur_num_heads, dpe)
        // Q_upper (1, batch, 1, cur_num_heads * dqu) -> view
        // (1, batch, cur_num_heads, dqu) -> sub_view (1, batch, cur_num_heads,
        // dpe)
        const int dpe = dqu - dnope;
        dim4 Q_rope_batch_shape = {1, block_batch, block_num_heads / core_num,
                                   block_dpe};
        dim4 Q_rope_batch_real_shape = {1, batch, cur_num_heads, dpe};
        auto Q_rope_batch_tensor =
            make_tensor<TYPE>(Q_rope_batch_shape, Q_rope_batch_real_shape);
        dim4 Q_upper_view_shape = {1, batch, cur_num_heads, dqu};
        dim4 Q_rope_batch_offset = {0, 0, 0, dnope};
        ppl::tiu::move(
            Q_rope_batch_tensor,
            Q_upper_tensor.view(Q_upper_view_shape)
                .sub_view(Q_rope_batch_real_shape, Q_rope_batch_offset));
        
        ppl::parallel_start();
        // store Q_rope to L2
        dim4 Q_rope_batch_global_offset = {0, 0, core_idx * WU_head_tile, 0};
        ppl::dma::store(
            Q_rope_batch_l2_tensor.sub_view(Q_rope_batch_real_shape,
                                            Q_rope_batch_global_offset),
            Q_rope_batch_tensor);

        // cpy Q_nope from Q_upper
        // Q_nope: (cur_num_heads, batch, 1, dnope)
        dim4 Q_nope_batch_shape = {block_num_heads / core_num, block_batch, 1,
                                   block_dnope};
        dim4 Q_nope_batch_real_shape = {cur_num_heads, batch, 1, dnope};
        auto Q_nope_batch_tensor =
            make_tensor<TYPE>(Q_nope_batch_shape, Q_nope_batch_real_shape);
        dim4 Q_nope_batch_real_shape_trans = {1, batch, cur_num_heads, dnope};
        dim4 Q_nope_batch_offset = {0, 0, 0, 0};
        int Q_nope_align = ppl::align(cur_num_heads * dqu, dtype_eu_num);
        dim4 Q_nope_batch_stride = {dqu, Q_nope_align, dnope, 1};

        ppl::tiu::move(
            Q_nope_batch_tensor,
            Q_upper_tensor.view(Q_upper_view_shape)
                .sub_view(Q_nope_batch_real_shape_trans, Q_nope_batch_offset)
                .view(Q_nope_batch_real_shape, Q_nope_batch_stride));
        ppl::parallel_end();
        // calculate Q_nope * WUKV_K
        // (cur_num_heads, batch, 1, dnope) * (cur_num_heads, dkv, 1, dnope) ->
        // (cur_num_heads, batch, 1, dkv) Q_WUKV_K: (cur_num_heads, batch, 1,
        // dkv)
        for (int i = 0; i < cur_num_heads; i++) {
            ppl::enable_pipeline();
            dim4 Q_WUKV_shape = {1, block_batch, 1,
                             block_dkv};
            dim4 Q_WUKV_real_shape = {1, batch, 1, dkv};
            auto Q_WUKV_tensor = make_tensor<TYPE>(Q_WUKV_shape, Q_WUKV_real_shape);
            dim4 Q_nope_batch_per_head_shape = {1, batch, 1, dnope};
            dim4 WUKV_K_per_head_shape = {1, dkv, 1, dnope};
            dim4 offset = {i, 0, 0, 0};
            ppl::tiu::fmm2(
                Q_WUKV_tensor,
                Q_nope_batch_tensor.sub_view(Q_nope_batch_per_head_shape,
                                             offset),
                WUKV_K_tensor.sub_view(WUKV_K_per_head_shape, offset), false,
                true, false);
            // store Q_WUKV to L2
            dim4 Q_WUKV_l2_shape = {batch, 1, 1, dkv};
            dim4 Q_WUKV_l2_offset = {0, core_idx * WU_head_tile + i, 0, 0};
            ppl::dma::store_transpose_nc(
                Q_WUKV_batch_l2_tensor.sub_view(Q_WUKV_l2_shape, Q_WUKV_l2_offset),
                Q_WUKV_tensor);
        }
    } else {
        // gdma load WUKV
        dim4 WUKV_global_shape = {1, num_heads * dkvu, 1, dkv};
        auto WUKV_global_tensor =
            gtensor<W_TYPE>(WUKV_global_shape, GLOBAL, WUKV_global_addr);

        dim4 WUKV_global_offset = {0, core_idx * WU_head_tile * dkvu, 0, 0};
        dim4 WUKV_shape = {
            1, block_num_heads / core_num * (block_dnope + block_dv), 1,
            block_dkv};
        dim4 WUKV_real_shape = {1, cur_num_heads * dkvu, 1, dkv};
        auto WUKV_tensor = make_tensor<W_TYPE>(WUKV_shape, WUKV_real_shape);
        ppl::dma::load(WUKV_tensor, WUKV_global_tensor.sub_view(
                                        WUKV_real_shape, WUKV_global_offset));
        dim4 WUKV_cw_shape = {
            1, block_dkv, 1,
            block_num_heads / core_num * (block_dnope + block_dv)};
        dim4 WUKV_cw_real_shape = {1, dkv, 1, cur_num_heads * dkvu};
        auto WUKV_cw_tensor =
            make_tensor<W_TYPE>(WUKV_cw_shape, WUKV_cw_real_shape);
        
        ppl::parallel_start();
        // (1, cur_num_heads * dkvu, 1, dkv) -> cw_trans (1, dkv, 1,
        // cur_num_heads * dkvu)
        ppl::tiu::transpose_wc(WUKV_cw_tensor, WUKV_tensor);

        auto WUKV_cw_tensor_dtype =
            make_tensor<TYPE>(WUKV_cw_shape, WUKV_cw_real_shape);

        int dkvu_align = ppl::align(cur_num_heads * dkvu, dtype_eu_num);

        dim4 WUKV_cw_view_shape = {1, dkv, cur_num_heads, dkvu};
        dim4 WUKV_cw_v_sub_view_shape = {1, dkv, cur_num_heads, dv};
        dim4 WUKV_cw_k_sub_view_shape = {1, dkv, cur_num_heads, dnope};
        dim4 WUKV_cw_v_offset = {0, 0, 0, dnope};
        dim4 WUKV_cw_k_offset = {0, 0, 0, 0};
        dim4 V_stride = {dkvu, dkvu_align, dv, 1};
        dim4 K_stride = {dkvu, dkvu_align, dnope, 1};

        dim4 WUKV_V_shape = {block_num_heads / core_num, block_dkv, 1,
                             block_dv};
        dim4 WUKV_V_real_shape = {cur_num_heads, dkv, 1, dv};
        dim4 WUKV_K_shape = {block_num_heads / core_num, block_dkv, 1,
                             block_dnope};
        dim4 WUKV_K_real_shape = {cur_num_heads, dkv, 1, dnope};
        auto WUKV_V_buffer_tensor =
            make_tensor<TYPE>(WUKV_V_shape, WUKV_V_real_shape);
        // WUKV_cw_tensor (1, dkv, 1, cur_num_heads * dkvu) -> view (1, dkv,
        // cur_num_heads, dkvu) -> sub_view (1, dkv, cur_num_heads, dv) -> view
        // (cur_num_heads, dkv, 1, dv)

        ppl::tiu::move(WUKV_V_buffer_tensor,
                       WUKV_cw_tensor.template view<TYPE>(WUKV_cw_view_shape)
                           .sub_view(WUKV_cw_v_sub_view_shape, WUKV_cw_v_offset)
                           .view(WUKV_V_real_shape, V_stride));

        // (cur_num_heads, dkv, 1, dv) -> cw_trans (cur_num_heads, dv, 1, dkv)
        ppl::tiu::transpose_wc(WUKV_V_tensor, WUKV_V_buffer_tensor);
        // WUKV_cw_tensor (1, dkv, 1, cur_num_heads * dkvu) -> view (1, dkv,
        // cur_num_heads, dkvu) -> sub_view (1, dkv, cur_num_heads, dnope) ->
        // view (cur_num_heads, dkv, 1, dnope)

        ppl::tiu::move(WUKV_K_tensor,
                       WUKV_cw_tensor.template view<TYPE>(WUKV_cw_view_shape)
                           .sub_view(WUKV_cw_k_sub_view_shape, WUKV_cw_k_offset)
                           .view(WUKV_K_real_shape, K_stride));

        // gdma load WUQ
        dim4 WUQ_global_shape = {1, num_heads * dqu, 1, dq};
        auto WUQ_global_tensor =
            gtensor<W_TYPE>(WUQ_global_shape, GLOBAL, WUQ_global_addr);
        dim4 WUQ_global_offset = {0, core_idx * WU_head_tile * dqu, 0, 0};
        dim4 WUQ_shape = {
            1, block_num_heads / core_num * (block_dnope + block_dpe), 1,
            block_dq};
        dim4 WUQ_real_shape = {1, cur_num_heads * dqu, 1, dq};

        ppl::dma::load(WUQ_tensor,
                       WUQ_global_tensor.template view<TYPE>().sub_view(
                           WUQ_real_shape, WUQ_global_offset));

        // gdma load Q_lora
        dim4 Q_lora_shape = {1, block_batch, 1, block_dq};
        dim4 Q_lora_real_shape = {1, batch, 1, dq};
        auto Q_lora_global_tensor =
            gtensor<TYPE>(Q_lora_real_shape, GLOBAL, Q_global_addr);
        auto Q_lora_tensor = make_tensor<TYPE>(Q_lora_shape, Q_lora_real_shape);
        ppl::dma::load(Q_lora_tensor, Q_lora_global_tensor);
        ppl::parallel_end();

        // calculate Q_upper
        // (1, batch, 1, dq) * (1, cur_num_heads * dqu, 1, dq) -> (1, batch, 1,
        // cur_num_heads * dqu)
        dim4 Q_upper_shape = {
            1, block_batch, 1,
            block_num_heads / core_num * (block_dnope + block_dpe)};
        dim4 Q_upper_real_shape = {1, batch, 1, cur_num_heads * dqu};
        auto Q_upper_tensor =
            make_tensor<TYPE>(Q_upper_shape, Q_upper_real_shape);
        ppl::tiu::fmm2(Q_upper_tensor, Q_lora_tensor, WUQ_tensor, false, true,
                       false);

        // cpy Q_rope from Q_upper
        // Q_rope: (1, batch, cur_num_heads, dpe)
        // Q_upper (1, batch, 1, cur_num_heads * dqu) -> view
        // (1, batch, cur_num_heads, dqu) -> sub_view (1, batch, cur_num_heads,
        // dpe)
        const int dpe = dqu - dnope;
        dim4 Q_rope_batch_shape = {1, block_batch, block_num_heads / core_num,
                                   block_dpe};
        dim4 Q_rope_batch_real_shape = {1, batch, cur_num_heads, dpe};
        auto Q_rope_batch_tensor =
            make_tensor<TYPE>(Q_rope_batch_shape, Q_rope_batch_real_shape);
        dim4 Q_upper_view_shape = {1, batch, cur_num_heads, dqu};
        dim4 Q_rope_batch_offset = {0, 0, 0, dnope};
        ppl::tiu::move(
            Q_rope_batch_tensor,
            Q_upper_tensor.view(Q_upper_view_shape)
                .sub_view(Q_rope_batch_real_shape, Q_rope_batch_offset));
        
        ppl::parallel_start();
        // store Q_rope to L2
        dim4 Q_rope_batch_global_offset = {0, 0, core_idx * WU_head_tile, 0};
        ppl::dma::store(
            Q_rope_batch_l2_tensor.sub_view(Q_rope_batch_real_shape,
                                            Q_rope_batch_global_offset),
            Q_rope_batch_tensor);

        // cpy Q_nope from Q_upper
        // Q_nope: (cur_num_heads, batch, 1, dnope)
        dim4 Q_nope_batch_shape = {block_num_heads / core_num, block_batch, 1,
                                   block_dnope};
        dim4 Q_nope_batch_real_shape = {cur_num_heads, batch, 1, dnope};
        auto Q_nope_batch_tensor =
            make_tensor<TYPE>(Q_nope_batch_shape, Q_nope_batch_real_shape);
        dim4 Q_nope_batch_real_shape_trans = {1, batch, cur_num_heads, dnope};
        dim4 Q_nope_batch_offset = {0, 0, 0, 0};
        int Q_nope_align = ppl::align(cur_num_heads * dqu, dtype_eu_num);
        dim4 Q_nope_batch_stride = {dqu, Q_nope_align, dnope, 1};

        ppl::tiu::move(
            Q_nope_batch_tensor,
            Q_upper_tensor.view(Q_upper_view_shape)
                .sub_view(Q_nope_batch_real_shape_trans, Q_nope_batch_offset)
                .view(Q_nope_batch_real_shape, Q_nope_batch_stride));

        // calculate Q_nope * WUKV_K
        // (cur_num_heads, batch, 1, dnope) * (cur_num_heads, dkv, 1, dnope) ->
        // (cur_num_heads, batch, 1, dkv) Q_WUKV_K: (cur_num_heads, batch, 1,
        // dkv)
        dim4 Q_WUKV_shape = {block_num_heads / core_num, block_batch, 1,
                             block_dkv};
        dim4 Q_WUKV_real_shape = {cur_num_heads, batch, 1, dkv};
        auto Q_WUKV_tensor = make_tensor<TYPE>(Q_WUKV_shape, Q_WUKV_real_shape);
        for (int i = 0; i < cur_num_heads; i++) {
            dim4 Q_WUKV_per_head_shape = {1, batch, 1, dkv};
            dim4 Q_nope_batch_per_head_shape = {1, batch, 1, dnope};
            dim4 WUKV_K_per_head_shape = {1, dkv, 1, dnope};
            dim4 offset = {i, 0, 0, 0};
            ppl::tiu::fmm2(
                Q_WUKV_tensor.sub_view(Q_WUKV_per_head_shape, offset),
                Q_nope_batch_tensor.sub_view(Q_nope_batch_per_head_shape,
                                             offset),
                WUKV_K_tensor.sub_view(WUKV_K_per_head_shape, offset), false,
                true, false);
        }
        ppl::parallel_end();
        // store Q_WUKV to L2
        dim4 Q_WUKV_l2_shape = {batch, cur_num_heads, 1, dkv};
        dim4 Q_WUKV_l2_offset = {0, core_idx * WU_head_tile, 0, 0};
        ppl::dma::store_transpose_nc(
            Q_WUKV_batch_l2_tensor.sub_view(Q_WUKV_l2_shape, Q_WUKV_l2_offset),
            Q_WUKV_tensor);
    }

    // sync all cores
    ppl::sync();
}

template <typename TYPE>
void mla_decode_kvcache_batch_DP(
    tensor<TYPE> &KV_tensor, tensor<TYPE> &PE_tensor,
    gtensor<TYPE> &KVcache_global_tensor, gtensor<TYPE> &PEcache_global_tensor,
    gtensor<TYPE> &KVcache_l2_tensor, gtensor<TYPE> &PEcache_l2_tensor,
    gtensor<TYPE> &PEcache_save_global_tensor,
    gtensor<uint32> &save_slots_global_tensor, tensor<TYPE> &ROPE_cos_tensor,
    tensor<TYPE> &ROPE_sin_tensor, gtensor<TYPE> &Q_rope_batch_l2_tensor,
    gtensor<TYPE> &Q_WUKV_batch_l2_tensor, gtensor<TYPE> &QKV_batch_l2_tensor,
    int q_lora_rank, int kv_lora_rank, int qk_nope_head_dim,
    int qk_rope_head_dim, int v_head_dim, float C, int batch, int batch_idx,
    int batch_idx_percore, int num_heads, int seq_len, int attention_mode)
{
    int dq = q_lora_rank;
    int dkv = kv_lora_rank;
    int dnope = qk_nope_head_dim;
    int dpe = qk_rope_head_dim;
    int dqu = dnope + dpe;
    int dv = v_head_dim;

    // load Q_rope from l2
    dim4 Q_rope_shape = {1, block_num_heads, 1, block_dpe};
    dim4 Q_rope_real_shape = {1, num_heads, 1, dpe};
    auto Q_rope_tensor = make_tensor<TYPE>(Q_rope_shape, Q_rope_real_shape);
    dim4 Q_rope_l2_shape = {batch, num_heads, 1, dpe};
    dim4 Q_rope_l2_offset = {batch_idx, 0, 0, 0};
    ppl::dma::load(Q_rope_tensor,
                   Q_rope_batch_l2_tensor.view(Q_rope_l2_shape)
                       .sub_view(Q_rope_real_shape, Q_rope_l2_offset));

    // load KVcache
    dim4 KVcache_shape = {1, block_tiled_seqlen_k + 1, 1, block_dkv};
    dim4 KVcache_real_shape = {1, seq_len + 1, 1, dkv};
    auto KVcache_tensor = make_tensor<TYPE>(KVcache_shape, KVcache_real_shape);
    dim4 KVcache_sub_shape = {1, seq_len, 1, dkv};
    dim4 KVcache_sub_offset = {0, 0, 0, 0};
    if (attention_mode == 3) {  // PAGE_KV_CACHE_DECODE = 3
        ppl::dma::load(
            KVcache_tensor.sub_view(KVcache_sub_shape, KVcache_sub_offset),
            KVcache_l2_tensor);
    } else {
        ppl::dma::load(
            KVcache_tensor.sub_view(KVcache_sub_shape, KVcache_sub_offset),
            KVcache_global_tensor);
    }

    // load PEcache
    dim4 PEcache_shape = {1, block_tiled_seqlen_k + 1, 1, block_dpe};
    dim4 PEcache_real_shape = {1, seq_len + 1, 1, dpe};
    dim4 PEcache_sub_shape = {1, seq_len, 1, dpe};
    auto PEcache_tensor = make_tensor<TYPE>(PEcache_shape, PEcache_real_shape);
    dim4 PEcache_sub_offset = {0, 0, 0, 0};
    if (attention_mode == 3) {  // PAGE_KV_CACHE_DECODE = 3
        ppl::dma::load(
            PEcache_tensor.sub_view(PEcache_sub_shape, PEcache_sub_offset),
            PEcache_l2_tensor);
    } else {
        ppl::dma::load(
            PEcache_tensor.sub_view(PEcache_sub_shape, PEcache_sub_offset),
            PEcache_global_tensor);
    }

    // load Q_WUKV
    dim4 Q_WUKV_shape = {1, block_num_heads, 1, block_dkv};
    dim4 Q_WUKV_real_shape = {1, num_heads, 1, dkv};
    auto Q_WUKV_tensor = make_tensor<TYPE>(Q_WUKV_shape, Q_WUKV_real_shape);
    dim4 Q_WUKV_l2_offset = {batch_idx, 0, 0, 0};
    ppl::dma::load(Q_WUKV_tensor, Q_WUKV_batch_l2_tensor.sub_view(
                                      Q_WUKV_real_shape, Q_WUKV_l2_offset));

    // load RoPE cos sin
    dim4 RoPE_weight_shape = {1, 1, 1, block_dpe};
    dim4 RoPE_weight_real_shape = {1, 1, 1, dpe};
    auto RoPE_cos_tensor_perbatch =
        make_tensor<TYPE>(RoPE_weight_shape, RoPE_weight_real_shape);
    auto RoPE_sin_tensor_perbatch =
        make_tensor<TYPE>(RoPE_weight_shape, RoPE_weight_real_shape);
    dim4 batch_offset_percore = {0, batch_idx_percore, 0, 0};
    ppl::tiu::move_cross_lane(
        RoPE_cos_tensor_perbatch,
        ROPE_cos_tensor.sub_view(RoPE_weight_real_shape, batch_offset_percore));
    ppl::tiu::move_cross_lane(
        RoPE_sin_tensor_perbatch,
        ROPE_sin_tensor.sub_view(RoPE_weight_real_shape, batch_offset_percore));

    // apply Q_rope
    auto Q_rope_buffer_tensor =
        make_tensor<TYPE>(Q_rope_shape, Q_rope_real_shape);
    auto Q_rope_cos_sin_buffer_tensor =
        make_tensor<TYPE>(Q_rope_shape, Q_rope_real_shape);
    auto Q_rope_res_tensor = make_tensor<TYPE>(Q_rope_shape, Q_rope_real_shape);
    attn_rope_local(Q_rope_res_tensor, Q_rope_tensor, RoPE_cos_tensor_perbatch,
                    RoPE_sin_tensor_perbatch, Q_rope_buffer_tensor,
                    Q_rope_cos_sin_buffer_tensor, num_heads, dpe);

    // load K_rope
    dim4 K_rope_shape = {1, 1, 1, block_dpe};
    dim4 K_rope_real_shape = {1, 1, 1, dpe};
    auto K_rope_tensor = make_tensor<TYPE>(K_rope_shape, K_rope_real_shape);
    ppl::tiu::move_cross_lane(
        K_rope_tensor,
        PE_tensor.sub_view(K_rope_real_shape, batch_offset_percore));

    // apply K_rope
    auto K_rope_res_tensor = make_tensor<TYPE>(K_rope_shape, K_rope_real_shape);
    auto K_rope_buffer_tensor =
        make_tensor<TYPE>(K_rope_shape, K_rope_real_shape);
    auto K_rope_cos_sin_buffer_tensor =
        make_tensor<TYPE>(K_rope_shape, K_rope_real_shape);
    attn_rope_local(K_rope_res_tensor, K_rope_tensor, RoPE_cos_tensor_perbatch,
                    RoPE_sin_tensor_perbatch, K_rope_buffer_tensor,
                    K_rope_cos_sin_buffer_tensor, 1, dpe);

    // concat KV PE
    dim4 cache_offset = {0, seq_len, 0, 0};
    ppl::tiu::move_cross_lane(
        PEcache_tensor.sub_view(K_rope_real_shape, cache_offset),
        K_rope_res_tensor);

    // load KV_lora
    dim4 KV_lora_real_shape = {1, 1, 1, dkv};
    ppl::tiu::move_cross_lane(
        KVcache_tensor.sub_view(KV_lora_real_shape, cache_offset),
        KV_tensor.sub_view(KV_lora_real_shape, batch_offset_percore));

    // bdc cal
    // QK_rope
    // Q_WUKV * new_kvcache
    // Q_WUKV {1,num_heads,1,dkv}
    // new_kvcache {1,seq_len(+1),1, dkv}
    // result: QK_buffer {1,num_heads, 1, seqlen(+1)}
    dim4 QK_buffer_cw_shape = {1, block_tiled_seqlen_k + 1, 1, block_num_heads};
    dim4 QK_buffer_cw_real_shape = {1, seq_len + 1, 1, num_heads};
    auto QK_buffer_cw_tensor =
        make_tensor<QK_TYPE>(QK_buffer_cw_shape, QK_buffer_cw_real_shape);
    ppl::tiu::fmm2(QK_buffer_cw_tensor, KVcache_tensor, Q_WUKV_tensor, false,
                   true, false);

    dim4 QK_shape = {1, block_num_heads, 1, block_tiled_seqlen_k + 1};
    dim4 QK_real_shape = {1, num_heads, 1, seq_len + 1};
    auto QK_buffer_tensor = make_tensor<QK_TYPE>(QK_shape, QK_real_shape);
    ppl::tiu::transpose_wc(QK_buffer_tensor, QK_buffer_cw_tensor);

    // QK_nope
    auto QK_tensor = make_tensor<QK_TYPE>(QK_shape, QK_real_shape);
    ppl::tiu::fmm2(QK_tensor, Q_rope_res_tensor, PEcache_tensor, false, true,
                   false);
    // QK_nope + QK_rope
    ppl::tiu::fadd(QK_tensor, QK_tensor, QK_buffer_tensor);
    // QK * scale
    ppl::tiu::fmul(QK_buffer_tensor, QK_tensor, C);

    // softmax
    auto QK_dtype_tensor = make_tensor<TYPE>(QK_shape, QK_real_shape);
    dim4 softmax_local_shape = {1, block_num_heads, 1, 1};
    dim4 softmax_local_real_shape = {1, num_heads, 1, 1};
    auto softmax_last_max_val_tensor =
        make_tensor<QK_TYPE>(softmax_local_shape, softmax_local_real_shape);
    auto softmax_exp_sum_tensor =
        make_tensor<TYPE>(softmax_local_shape, softmax_local_real_shape);
    attn_softmax_local(QK_dtype_tensor, QK_buffer_tensor,
                       softmax_last_max_val_tensor, softmax_exp_sum_tensor,
                       num_heads, seq_len + 1);

    // QKV
    dim4 QKV_shape = {1, block_num_heads, 1, block_dkv};
    dim4 QKV_real_shape = {1, num_heads, 1, dkv};
    auto QKV_tensor = make_tensor<TYPE>(QKV_shape, QKV_real_shape);
    ppl::tiu::fmm2(QKV_tensor, QK_dtype_tensor, KVcache_tensor);

    // QKV / exp_sum
    ppl::tiu::fdiv(QKV_tensor, QKV_tensor, softmax_exp_sum_tensor);

    dim4 QKV_batch_l2_offset = {batch_idx, 0, 0, 0};
    ppl::dma::store(
        QKV_batch_l2_tensor.sub_view(QKV_real_shape, QKV_batch_l2_offset),
        QKV_tensor);
    // store
    if (attention_mode == 3) {  // PAGE_KV_CACHE_DECODE = 3
        ppl::dma::scatter_h(PEcache_save_global_tensor, K_rope_res_tensor,
                            save_slots_global_tensor);
    } else {
        ppl::dma::store(PEcache_save_global_tensor, K_rope_res_tensor);
    }
}

template <typename TYPE, typename W_TYPE>
void mla_decode_kv_cache_multi_core(
    TYPE *Q_global_addr, TYPE *KV_global_addr, TYPE *PE_global_addr,
    TYPE *KVcache_global_addr, TYPE *PEcache_global_addr,
    TYPE *RoPE_cos_global_addr, TYPE *RoPE_sin_global_addr,
    W_TYPE *WUQ_global_addr, W_TYPE *WUKV_global_addr, TYPE *Y_global_addr,
    TYPE *WUQ_scale_gaddr, TYPE *WUKV_scale_gaddr, uint32 *block_table_gaddr,
    uint32 *save_slots_gaddr, int num_heads, int q_lora_rank, int kv_lora_rank,
    int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim, float C,
    int batch, int quant_block_size,
    int paged_block_size,  // paged block size
    int fetch_block_num,   // paged block num of per-batch
    int max_cache_size,    // cache size of per batch for normal attention
    int attention_mode,
    int seqlen[MAX_BATCH_SIZE]  // the shape is {batch} and the value is cached
                                // seqlen + decode seqlen, and now only support
                                // decode seqlen=1
)
{
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    ppl::set_config_auto_sync(false);

    int dq = q_lora_rank;
    int dqu = qk_nope_head_dim + qk_rope_head_dim;
    int dkv = kv_lora_rank;
    int dnope = qk_nope_head_dim;
    int dpe = qk_rope_head_dim;
    int dv = v_head_dim;
    int dkvu = dnope + dv;

    const int generate_token = 1;
    int max_fetch_tokens = get_max_fetch_tokens(seqlen, batch) - generate_token;
    int tiled_fetch_tokens = ppl::min(block_tiled_seqlen_k, max_fetch_tokens);

    if (attention_mode == 3) {  // PAGE_KV_CACHE_DECODE = 3
        ppl::assert(paged_block_size > 0 && fetch_block_num > 0);
        if (core_idx == 0) {
            dim4 KVcache_global_shape = {1, 1, max_gather_scatter_num,
                                         kv_lora_rank};
            auto KVcache_global_tensor = gtensor<TYPE>(
                KVcache_global_shape, GLOBAL, KVcache_global_addr);
            dim4 KV_global_shape = {1, 1, batch, kv_lora_rank};
            auto KV_global_tensor =
                gtensor<TYPE>(KV_global_shape, GLOBAL, KV_global_addr);
            dim4 save_slots_global_shape = {1, 1, batch, 1};
            auto save_slots_global_tensor = gtensor<uint32>(
                save_slots_global_shape, GLOBAL, save_slots_gaddr);
            ppl::sdma::scatter_h(KVcache_global_tensor, KV_global_tensor,
                                 save_slots_global_tensor);
        }
    } else {
        ppl::assert(max_cache_size > tiled_fetch_tokens);
        const int batch_per_core = ppl::div_up(batch, core_num);
        int batch_slice =
            ppl::min(batch_per_core, batch - batch_per_core * core_idx);
        if (batch_slice > 0) {
            dim4 KVcache_global_shape = {batch, 1, max_cache_size,
                                         kv_lora_rank};
            auto KVcache_global_tensor = gtensor<TYPE>(
                KVcache_global_shape, GLOBAL, KVcache_global_addr);
            dim4 KV_global_shape = {batch, 1, 1, kv_lora_rank};
            auto KV_global_tensor =
                gtensor<TYPE>(KV_global_shape, GLOBAL, KV_global_addr);
            for (int bidx = 0; bidx < batch_slice; ++bidx) {
                const int cur_batch = bidx + core_idx * batch_per_core;
                dim4 KV_per_batch_global_shape = {1, 1, 1, kv_lora_rank};
                dim4 KVcache_global_offset = {
                    cur_batch, 0, seqlen[cur_batch] - generate_token, 0};
                dim4 KV_global_offset = {cur_batch, 0, 0, 0};
                ppl::sdma::move(
                    KVcache_global_tensor.sub_view(KV_per_batch_global_shape,
                                                   KVcache_global_offset),
                    KV_global_tensor.sub_view(KV_per_batch_global_shape,
                                              KV_global_offset));
            }
        }
    }

    int head_per_core = num_heads / core_num;
    int cur_num_heads =
        ppl::min(head_per_core, num_heads - core_idx * head_per_core);

    dim4 WUQ_shape = {1,
                      (block_num_heads / core_num) * (block_dnope + block_dpe),
                      1, block_dq};
    dim4 WUQ_real_shape = {1, cur_num_heads * dqu, 1, dq};
    dim4 WUKV_K_shape = {(block_num_heads / core_num), block_dkv, 1,
                         block_dnope};
    dim4 WUKV_K_real_shape = {cur_num_heads, dkv, 1, dnope};
    dim4 WUKV_V_shape = {(block_num_heads / core_num), block_dv, 1, block_dkv};
    dim4 WUKV_V_real_shape = {cur_num_heads, dv, 1, dkv};

    auto WUQ_tensor = make_tensor<TYPE>(WUQ_shape, WUQ_real_shape);
    auto WUKV_K_tensor = make_tensor<TYPE>(WUKV_K_shape, WUKV_K_real_shape);
    auto WUKV_V_tensor = make_tensor<TYPE>(WUKV_V_shape, WUKV_V_real_shape);

    dim4 Q_rope_batch_shape = {1, block_batch, block_num_heads, block_dpe};
    dim4 Q_rope_batch_real_shape = {1, batch, num_heads, dpe};
    dim4 Q_WUKV_batch_shape = {block_batch, block_num_heads, 1, block_dkv};
    dim4 Q_WUKV_batch_real_shape = {batch, num_heads, 1, dkv};
    auto Q_rope_batch_l2_tensor =
        make_l2tensor<TYPE>(Q_rope_batch_shape, L2, Q_rope_batch_real_shape);
    auto Q_WUKV_batch_l2_tensor =
        make_l2tensor<TYPE>(Q_WUKV_batch_shape, L2, Q_WUKV_batch_real_shape);

    load_upper_weight(WUQ_global_addr, WUKV_global_addr, WUQ_scale_gaddr,
                      WUKV_scale_gaddr, Q_global_addr, Q_rope_batch_l2_tensor,
                      Q_WUKV_batch_l2_tensor, WUQ_tensor, WUKV_K_tensor,
                      WUKV_V_tensor, dq, dkv, dqu, dkvu, dnope, dv,
                      quant_block_size, cur_num_heads, num_heads, head_per_core,
                      core_idx, batch);

    if (batch >= core_num) {
        int batch_per_core = batch / core_num;
        int batch_residue = batch % core_num;
        int batch_num = batch_per_core + (core_idx < batch_residue ? 1 : 0);
        int batch_offset;
        if (core_idx < batch_residue) {
            batch_offset = core_idx * (batch_per_core + 1);
        } else {
            batch_offset = batch_residue * (batch_per_core + 1) +
                           (core_idx - batch_residue) * batch_per_core;
        }

        if (max_fetch_tokens <= block_tiled_seqlen_k) {
            dim4 QKV_batch_l2_shape = {block_batch, block_num_heads, 1,
                                       block_dkv};
            dim4 QKV_batch_l2_real_shape = {batch, num_heads, 1, dkv};
            auto QKV_batch_l2_tensor = make_l2tensor<TYPE>(
                QKV_batch_l2_shape, L2, QKV_batch_l2_real_shape);

            // RoPE cos sin
            dim4 cos_sin_shape = {1, div_up(block_batch, core_num), 1,
                                  block_dpe};
            dim4 cos_sin_real_shape = {1, batch_num, 1, dpe};
            auto ROPE_cos_tensor =
                make_tensor<TYPE>(cos_sin_shape, cos_sin_real_shape);
            auto ROPE_sin_tensor =
                make_tensor<TYPE>(cos_sin_shape, cos_sin_real_shape);

            dim4 cos_sin_global_shape = {1, batch, 1, dpe};
            auto ROPE_cos_global_tensor = gtensor<TYPE>(
                cos_sin_global_shape, GLOBAL, RoPE_cos_global_addr);
            auto ROPE_sin_global_tensor = gtensor<TYPE>(
                cos_sin_global_shape, GLOBAL, RoPE_sin_global_addr);

            dim4 batch_global_offset = {0, batch_offset, 0, 0};
            ppl::dma::load(ROPE_cos_tensor,
                           ROPE_cos_global_tensor.sub_view(
                               cos_sin_real_shape, batch_global_offset));
            ppl::dma::load(ROPE_sin_tensor,
                           ROPE_sin_global_tensor.sub_view(
                               cos_sin_real_shape, batch_global_offset));

            // KV PE
            dim4 KV_shape = {1, div_up(block_batch, core_num), 1, block_dkv};
            dim4 KV_real_shape = {1, batch_num, 1, dkv};
            auto KV_tensor = make_tensor<TYPE>(KV_shape, KV_real_shape);
            dim4 PE_shape = {1, div_up(block_batch, core_num), 1, block_dpe};
            dim4 PE_real_shape = {1, batch_num, 1, dpe};
            auto PE_tensor = make_tensor<TYPE>(PE_shape, PE_real_shape);

            dim4 KV_global_shape = {1, batch, 1, dkv};
            auto KV_global_tensor =
                gtensor<TYPE>(KV_global_shape, GLOBAL, KV_global_addr);
            dim4 PE_global_shape = {1, batch, 1, dpe};
            auto PE_global_tensor =
                gtensor<TYPE>(PE_global_shape, GLOBAL, PE_global_addr);

            ppl::dma::load(KV_tensor, KV_global_tensor.sub_view(
                                          KV_real_shape, batch_global_offset));
            ppl::dma::load(PE_tensor, PE_global_tensor.sub_view(
                                          PE_real_shape, batch_global_offset));
                                          
            for (int batch_idx_percore = 0; batch_idx_percore < batch_num;
                 batch_idx_percore++) {
                ppl::enable_pipeline();
                int batch_idx = batch_offset + batch_idx_percore;
                int seq_len = seqlen[batch_idx] - generate_token;

                // page cache
                dim4 kvcache_shared_shape = {CORE_NUM, block_tiled_seqlen_k + 1,
                                             1, block_dkv};
                dim4 kvcache_shared_real_shape = {CORE_NUM, seq_len, 1, dkv};
                dim4 kvcache_shared_core_real_shape = {1, seq_len, 1, dkv};
                dim4 cache_core_offset = {core_idx, 0, 0, 0};
                auto kvcache_shared_l2_tensor =
                    make_l2tensor<TYPE>(kvcache_shared_shape, L2,
                                        kvcache_shared_real_shape)
                        .sub_view(kvcache_shared_core_real_shape,
                                  cache_core_offset);

                dim4 pecache_shared_shape = {CORE_NUM, block_tiled_seqlen_k + 1,
                                             1, block_dpe};
                dim4 pecache_shared_real_shape = {CORE_NUM, seq_len, 1, dpe};
                dim4 pecache_shared_core_real_shape = {1, seq_len, 1, dpe};
                auto pecache_shared_l2_tensor =
                    make_l2tensor<TYPE>(pecache_shared_shape, L2,
                                        pecache_shared_real_shape)
                        .sub_view(pecache_shared_core_real_shape,
                                  cache_core_offset);

                int pecache_save_global_shape_n = batch;
                int pecache_save_global_shape_h = max_cache_size;
                int pecache_save_global_sub_shape_h = 1;
                int pecache_save_global_offset_n = batch_idx;
                int pecache_save_global_offset_h = seq_len;
                if (attention_mode == 3) {  // PAGE_KV_CACHE_DECODE = 3
                    pecache_save_global_shape_n = 1;
                    pecache_save_global_shape_h = max_gather_scatter_num;
                    pecache_save_global_sub_shape_h = max_gather_scatter_num;
                    pecache_save_global_offset_n = 0;
                    pecache_save_global_offset_h = 0;
                    dim4 block_table_global_shape = {batch, 1, 1,
                                                     fetch_block_num};
                    auto block_table_global_tensor = gtensor<uint32>(
                        block_table_global_shape, GLOBAL, block_table_gaddr);
                    int sub_block_num = ppl::div_up(seq_len, paged_block_size);
                    dim4 block_table_sub_global_shape = {1, 1, 1,
                                                         sub_block_num};
                    dim4 block_table_offset = {batch_idx, 0, 0, 0};
                    // load_cache(
                    //     KVcache_global_addr, kvcache_shared_l2_tensor,
                    //     block_table_global_tensor.sub_view(
                    //         block_table_sub_global_shape,
                    //         block_table_offset),
                    //     paged_block_size, dkv, seq_len);
                    // load_cache(
                    //     PEcache_global_addr, pecache_shared_l2_tensor,
                    //     block_table_global_tensor.sub_view(
                    //         block_table_sub_global_shape,
                    //         block_table_offset),
                    //     paged_block_size, dpe, seq_len);
                }

                // continous cache
                dim4 kvcache_global_shape = {batch, max_cache_size, 1, dkv};
                dim4 pecache_global_shape = {batch, max_cache_size, 1, dpe};
                dim4 cur_kvcache_global_shape = {1, seq_len, 1, dkv};
                dim4 cur_pecache_global_shape = {1, seq_len, 1, dpe};
                dim4 cache_global_offset = {batch_idx, 0, 0, 0};
                auto cur_kvcache_global_tensor =
                    gtensor<TYPE>(kvcache_global_shape, GLOBAL,
                                  KVcache_global_addr)
                        .sub_view(cur_kvcache_global_shape,
                                  cache_global_offset);
                auto cur_pecache_global_tensor =
                    gtensor<TYPE>(pecache_global_shape, GLOBAL,
                                  PEcache_global_addr)
                        .sub_view(cur_pecache_global_shape,
                                  cache_global_offset);

                // PE save
                dim4 pecache_save_global_shape = {
                    pecache_save_global_shape_n, 1, pecache_save_global_shape_h,
                    dpe};
                dim4 pecache_save_global_sub_shape = {
                    1, 1, pecache_save_global_sub_shape_h, dpe};
                dim4 pecache_save_global_offset = {
                    pecache_save_global_offset_n, 0,
                    pecache_save_global_offset_h, 0};
                auto pecache_save_global_sub_tensor =
                    gtensor<TYPE>(pecache_save_global_shape, GLOBAL,
                                  PEcache_global_addr)
                        .sub_view(pecache_save_global_sub_shape,
                                  pecache_save_global_offset);

                // save slots
                dim4 batch_global_offset = {batch_idx, 0, 0, 0};
                dim4 save_slots_global_shape = {batch, 1, 1, 1};
                dim4 save_slots_global_sub_shape = {1, 1, 1, 1};
                auto save_slots_global_sub_tensor =
                    gtensor<uint32>(save_slots_global_shape, GLOBAL,
                                    save_slots_gaddr)
                        .sub_view(save_slots_global_sub_shape,
                                  batch_global_offset);

                mla_decode_kvcache_batch_DP(
                    KV_tensor, PE_tensor, cur_kvcache_global_tensor,
                    cur_pecache_global_tensor, kvcache_shared_l2_tensor,
                    pecache_shared_l2_tensor, pecache_save_global_sub_tensor,
                    save_slots_global_sub_tensor, ROPE_cos_tensor,
                    ROPE_sin_tensor, Q_rope_batch_l2_tensor,
                    Q_WUKV_batch_l2_tensor, QKV_batch_l2_tensor, q_lora_rank,
                    kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
                    v_head_dim, C, batch, batch_idx, batch_idx_percore,
                    num_heads, seq_len, attention_mode);
            }
            ppl::sync();
            // QKV {h, b, 1, dkv}
            // WUKV_V {h, dv, 1, dkv}
            for (int h = 0; h < head_per_core; h++) {
                ppl::enable_pipeline();
                dim4 QKV_batch_shape = {1, block_batch, 1, block_dkv};
                dim4 QKV_batch_real_shape = {1, batch, 1, dkv};
                auto QKV_batch_tensor =
                    make_tensor<TYPE>(QKV_batch_shape, QKV_batch_real_shape);
                dim4 QKV_batch_offset = {0, core_idx * head_per_core + h, 0, 0};
                dim4 QKV_batch_l2_shape_per_core = {batch, 1, 1, dkv};
                ppl::dma::load_transpose_nc(
                    QKV_batch_tensor,
                    QKV_batch_l2_tensor.sub_view(QKV_batch_l2_shape_per_core,
                                                 QKV_batch_offset));
                dim4 QKV_out_shape = {1, block_batch, 1, block_dv};
                dim4 QKV_out_real_shape = {1, batch, 1, dv};
                auto QKV_out_tensor =
                    make_tensor<TYPE>(QKV_out_shape, QKV_out_real_shape);

                dim4 WUKV_V_shape_per_head = {1, dv, 1, dkv};
                dim4 head_offset = {h, 0, 0, 0};
                ppl::tiu::fmm2(
                    QKV_out_tensor, QKV_batch_tensor,
                    WUKV_V_tensor.sub_view(WUKV_V_shape_per_head, head_offset),
                    false, true, false);

                dim4 Y_global_shape = {batch, num_heads, 1, dv};
                dim4 Y_global_sub_shape = {batch, 1, 1, dv};
                dim4 Y_global_offset = {0, core_idx * head_per_core + h, 0, 0};
                auto Y_global_sub_tensor =
                    gtensor<TYPE>(Y_global_shape, GLOBAL, Y_global_addr)
                        .sub_view(Y_global_sub_shape, Y_global_offset);
                ppl::dma::store_transpose_nc(Y_global_sub_tensor,
                                             QKV_out_tensor);
            }
        }

    } else {
    }
}

__KERNEL__ void mla_decode_bf16(
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
    mla_decode_kv_cache_multi_core<bf16, bf16>(
        Q_global_addr, KV_global_addr, PE_global_addr, KVcache_global_addr,
        PEcache_global_addr, RoPE_cos_global_addr, RoPE_sin_global_addr,
        WUQ_global_addr, WUKV_global_addr, Y_global_addr, nullptr, nullptr,
        block_table_global_addr, save_slots_global_addr, n_heads, q_lora_rank,
        kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, C, batch,
        -1, block_size, max_paged_block_num, max_cache_size, attention_mode,
        data);
}

__KERNEL__ void mla_decode_bf16_fp8e4m3(
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
    mla_decode_kv_cache_multi_core<bf16, fp8e4m3>(
        Q_addr, KV_addr, PE_addr, KVcache_addr, PEcache_addr, RoPE_cos_addr,
        RoPE_sin_addr, WUQ_addr, WUKV_addr, Y_addr, WUQ_scale_addr,
        WUKV_scale_addr, block_table_addr, save_slots_addr, num_heads,
        q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
        v_head_dim, softmax_scale, batch, quant_block_size,
        paged_cache_block_size, max_paged_block_num, max_cache_size,
        attention_mode, seqlen);
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

    mla_decode_bf16_fp8e4m3(
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

    mla_decode_bf16(Q_addr, KV_addr, PE_addr, KVcache_addr, PEcache_addr,
                    KVU_addr, RoPE_cos_addr, RoPE_sin_addr, WUQ_addr, WUKV_addr,
                    Mask_addr, Y_addr, block_table_addr, save_slots_addr,
                    max_paged_block_num, paged_cache_block_size, num_heads,
                    q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                    qk_rope_head_dim, v_head_dim, softmax_scale, has_mask,
                    batch, mask_max, max_cache_size, attention_mode, seqlen);
}