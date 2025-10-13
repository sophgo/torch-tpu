#include "ppl.h"
using namespace ppl;

#ifdef __bm1690__
#define CORE_NUM 8
#elif defined(__sg2260e__)
#define CORE_NUM 4
#else
#define CORE_NUM 1
#endif

template <typename T>
void matmul_mn_mn_kernel(T* out_ptr, T* left_ptr, T* right_ptr,
                         int M, int K, int N,
                         const int m_slice, const int n_slice, const int k_slice,
                         const int group_core) {
    ppl::set_core_num(CORE_NUM);
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    if (core_idx >= core_num) return;

    int cores_per_group = core_num / group_core;
    if (cores_per_group <= 0) return;

    int group_id = core_idx / cores_per_group;      // 组号，决定 M 分块
    int core_in_group = core_idx % cores_per_group; // 组内核号，决定 N 分块

    // --- M 维按组划分 ---
    int m_group_slice = div_up(M, group_core);
    int m_offset = group_id * m_group_slice;
    int cur_m_total = min(m_group_slice, M - m_offset);
    if (cur_m_total <= 0) return;

    // --- N 维在组内划分 ---
    int n_core_slice = div_up(N, cores_per_group);
    int n_per_core = min(n_core_slice, N - core_in_group * n_core_slice);
    int n_core_offset = core_in_group * n_core_slice;
    if (n_per_core <= 0) return;

    int m_secs = div_up(cur_m_total, m_slice);
    int n_secs = div_up(n_per_core, n_slice);
    bool enable_core_interleave = n_secs * n_slice == n_per_core;

    // 创建全局张量
    dim4 left_global_shape = {1, M, 1, K};
    dim4 right_global_shape = {1, N, 1, K};
    dim4 res_global_shape   = {1, M, 1, N};

    auto left_gtensor  = gtensor<T>(left_global_shape, GLOBAL, left_ptr);
    auto right_gtensor = gtensor<T>(right_global_shape, GLOBAL, right_ptr);
    auto res_gtensor   = gtensor<T>(res_global_shape, GLOBAL, out_ptr);

    dim4 left_block_shape = {1, m_slice, 1, K};
    dim4 right_block_shape = {1, n_slice, 1, K};
    dim4 res_block_shape = {1, m_slice, 1, n_slice};

    for (int m_idx = 0; m_idx < m_secs; m_idx++) {

        int idx_m = m_offset + m_idx * m_slice;
        int cur_m = min(m_slice, cur_m_total - m_idx * m_slice);
        dim4 left_real_shape = {1, cur_m, 1, K};

        auto left_local = make_tensor<T>(left_block_shape, left_real_shape);
        dim4 left_offset = {0, idx_m, 0, 0};
        dma::load(left_local, left_gtensor.sub_view(left_real_shape, left_offset));

        for (int n_idx = 0; n_idx < n_secs; n_idx++) {
            ppl::enable_pipeline();

            int tmp_n_idx = enable_core_interleave ? (n_idx + core_in_group) % n_secs : n_idx;
            int idx_n = tmp_n_idx * n_slice;
            int cur_n = min(n_slice, n_per_core - idx_n);

            // 加载右矩阵
            dim4 right_real_shape = {1, cur_n, 1, K};
            auto right_local = make_tensor<T>(right_block_shape, right_real_shape);
            dim4 right_offset = {0, idx_n + n_core_offset, 0, 0};
            dma::load(right_local, right_gtensor.sub_view(right_real_shape, right_offset));

            // 矩阵乘法
            dim4 res_trans_block_shape = {1, m_slice, 1, n_slice};
            dim4 res_trans_real_shape = {1, cur_m, 1, cur_n};
            auto res_local  = make_tensor<T>(res_trans_block_shape, res_trans_real_shape);
            tiu::fmm2(res_local, left_local, right_local,false, true, false, false, false, 0);

            // 存储结果
            dim4 res_real_shape   = {1, cur_m, 1, cur_n};
            dim4 res_offset = {0, idx_m, 0, idx_n + n_core_offset};
            dma::store(res_gtensor.sub_view(res_real_shape, res_offset), res_local);
        }
    }
}

template <typename T>
void matmul_mn_k_kernel(T* out_ptr, T* left_ptr, T* right_ptr,
                             int M, int K, int N,
                             const int m_slice, const int n_slice, const int k_slice,
                             const int group_core) {
    ppl::set_core_num(CORE_NUM);
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    if (core_idx >= core_num) return;

    int cores_per_group = core_num / group_core;
    if (cores_per_group <= 0) return;

    int group_id = core_idx / cores_per_group;      // 组号，决定 M 分块
    int core_in_group = core_idx % cores_per_group; // 组内核号，决定 N 分块

    // --- M 维按组划分 ---
    int m_group_slice = div_up(M, group_core);
    int m_offset = group_id * m_group_slice;
    int cur_m_total = min(m_group_slice, M - m_offset);
    if (cur_m_total <= 0) return;

    // --- N 维在组内划分 ---
    int n_core_slice = div_up(N, cores_per_group);
    int n_per_core = min(n_core_slice, N - core_in_group * n_core_slice);
    int n_core_offset = core_in_group * n_core_slice;
    if (n_per_core <= 0) return;

    int m_secs = div_up(cur_m_total, m_slice);
    int n_secs = div_up(n_per_core, n_slice);
    bool enable_core_interleave = n_secs * n_slice == n_per_core;

    // 创建全局张量
    dim4 left_global_shape = {1, M, 1, K};
    dim4 right_global_shape = {1, N, 1, K};
    dim4 res_global_shape   = {1, M, 1, N};

    auto left_gtensor  = gtensor<T>(left_global_shape, GLOBAL, left_ptr);
    auto right_gtensor = gtensor<T>(right_global_shape, GLOBAL, right_ptr);
    auto res_gtensor   = gtensor<T>(res_global_shape, GLOBAL, out_ptr);

    dim4 left_block_shape = {1, m_group_slice, 1, k_slice};
    dim4 right_block_shape = {1, n_core_slice, 1, k_slice};
    dim4 res_block_shape = {1, m_group_slice, 1, n_core_slice};

    // 创建局部张量
    dim4 res_real_shape = {1, cur_m_total, 1, n_per_core};
    auto res_local   = make_tensor<T>(res_block_shape, res_real_shape);

    auto res_trans_local = make_tensor<fp32>(res_block_shape, res_real_shape);
    tiu::zero(res_trans_local);
    // 按 K 切分
    int k_secs = div_up(K, k_slice);
    for (int k_idx = 0; k_idx < k_secs; k_idx++) {
        ppl::enable_pipeline();

        int idx_k = k_idx * k_slice;
        int cur_k = min(k_slice, K - idx_k);

        // 左矩阵子块
        dim4 left_real_shape = {1, cur_m_total, 1, cur_k};
        auto left_local = make_tensor<T>(left_block_shape, left_real_shape);
        dim4 left_offset = {0, m_offset, 0, idx_k};
        dma::load(left_local, left_gtensor.sub_view(left_real_shape, left_offset));

        // 右矩阵子块
        dim4 right_real_shape = {1, n_per_core, 1, cur_k};
        auto right_local = make_tensor<T>(right_block_shape, right_real_shape);
        dim4 right_offset = {0, n_core_offset, 0, idx_k};
        dma::load(right_local, right_gtensor.sub_view(right_real_shape, right_offset));

        tiu::fmm2(res_trans_local, left_local, right_local, false, true, false, false, true, DT_FP32);

    }

    // 结果类型转换
    tiu::cast(res_local, res_trans_local, RM_HALF_TO_EVEN);

    // 存储结果
    dim4 res_offset = {0, m_offset, 0, n_core_offset};
    dma::store(res_gtensor.sub_view(res_real_shape, res_offset), res_local);
}


template <typename T>
void matmul_k_mn_kernel(
        T* out_ptr, T* left_ptr, T* right_ptr, int M, int K, int N,
        const int m_slice, const int n_slice, const int k_slice) {

    ppl::set_core_num(CORE_NUM);
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    if (core_idx >= core_num) return;

    int k_core_slice = div_up(K, core_num);
    int k_per_core = min(k_core_slice, K - core_idx * k_core_slice);
    int k_core_offset = core_idx * k_core_slice;

    // 全局张量
    dim4 left_global_shape = {1, M, 1, K};
    dim4 right_global_shape = {1, N, 1, K};
    dim4 res_global_shape   = {1, M, 1, N};

    auto left_gtensor  = gtensor<T>(left_global_shape,  GLOBAL, left_ptr);
    auto right_gtensor = gtensor<T>(right_global_shape, GLOBAL, right_ptr);
    auto res_gtensor   = gtensor<T>(res_global_shape,   GLOBAL, out_ptr);
    auto res_l2   = gtensor<T>(res_global_shape, L2);
    dma::zero(res_l2);

    dim4 res_accum_shape   = {1, M, 1, N};
    auto res_accum = make_tensor<fp32>(res_accum_shape, res_accum_shape);
    auto res_accum_bf16 = make_tensor<T>(res_accum_shape, res_accum_shape);
    tiu::zero(res_accum);
    tiu::zero(res_accum_bf16);

    if(k_per_core <= 0) {
      sync();
      dma::reduce(res_l2, res_accum_bf16, ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
      sync();
      dma::move(res_gtensor, res_l2);
      return;
    }

    dim4 left_block_shape = {1, m_slice, 1, k_core_slice};
    dim4 right_block_shape = {1, n_slice, 1, k_core_slice};
    dim4 res_block_shape = {1, m_slice, 1, n_slice};

    int m_secs = div_up(M, m_slice);
    int n_secs = div_up(N, n_slice);

    for (int m_idx = 0; m_idx < m_secs; m_idx++) {
        int idx_m = m_idx * m_slice;
        int cur_m = min(m_slice, M - idx_m);
        dim4 left_real_shape = {1, cur_m, 1, k_per_core};

        // 加载左矩阵
        auto left_local = make_tensor<T>(left_block_shape, left_real_shape);
        dim4 left_offset = {0, idx_m, 0, k_core_offset};
        dma::load(left_local, left_gtensor.sub_view(left_real_shape, left_offset));

        for(int n_idx = 0; n_idx < n_secs; n_idx++){
            ppl::enable_pipeline();
            int idx_n = n_idx * n_slice;
            int cur_n = min(n_slice, N - idx_n);

            // 加载右矩阵
            dim4 right_real_shape = {1, cur_n, 1, k_per_core};
            auto right_local = make_tensor<T>(right_block_shape, right_real_shape);
            dim4 right_offset = {0, idx_n, 0, k_core_offset};
            dma::load(right_local, right_gtensor.sub_view(right_real_shape, right_offset));

            // 矩阵乘法
            dim4 res_trans_block_shape = {1, m_slice, 1, n_slice};
            dim4 res_trans_real_shape = {1, cur_m, 1, cur_n};
            auto res_trans_local = make_tensor<fp32>(res_trans_block_shape, res_trans_real_shape);

            tiu::fmm2(res_trans_local, left_local, right_local, false, true, false, false, false, DT_FP32);

            // 存储结果
            dim4 res_real_shape   = {1, cur_m, 1, cur_n};
            dim4 res_offset = {0, idx_m, 0, idx_n};
            dma::move(res_accum.sub_view(res_real_shape, res_offset), res_trans_local);
        }
    }

    tiu::cast(res_accum_bf16, res_accum, RM_HALF_AWAY_FROM_ZERO);
    sync();
    dma::reduce(res_l2, res_accum_bf16, ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
    sync();

    dma::move(res_gtensor, res_l2);
}


template <typename T>
void matmul_k_k_kernel(
        T* out_ptr, T* left_ptr, T* right_ptr, int M, int K, int N,
        const int m_slice, const int n_slice, const int k_slice) {

    ppl::set_core_num(CORE_NUM);
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    if (core_idx >= core_num) return;

    int k_core_slice = div_up(K, core_num);
    int k_per_core = min(k_core_slice, K - core_idx * k_core_slice);
    int k_core_offset = core_idx * k_core_slice;

    // 全局张量
    dim4 left_global_shape = {1, M, 1, K};
    dim4 right_global_shape = {1, N, 1, K};
    dim4 res_global_shape   = {1, M, 1, N};

    auto left_gtensor  = gtensor<T>(left_global_shape,  GLOBAL, left_ptr);
    auto right_gtensor = gtensor<T>(right_global_shape, GLOBAL, right_ptr);
    auto res_gtensor   = gtensor<T>(res_global_shape,   GLOBAL, out_ptr);
    auto res_l2   = gtensor<T>(res_global_shape, L2);
    dma::zero(res_l2);

    dim4 offset = {0, 0, 0, 0};
    dim4 res_accum_shape   = {1, M, 1, N};
    auto res_accum = make_tensor<fp32>(res_accum_shape, res_accum_shape);
    auto res_accum_bf16 = make_tensor<T>(res_accum_shape, res_accum_shape);
    tiu::zero(res_accum);
    tiu::zero(res_accum_bf16);

    if(k_per_core <= 0) {
      sync();
      dma::reduce(res_l2, res_accum_bf16, ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
      sync();
      dma::move(res_gtensor, res_l2);
      return;
    }

    dim4 left_block_shape = {1, M, 1, k_slice};
    dim4 right_block_shape = {1, N, 1, k_slice};
    dim4 res_block_shape = {1, M, 1, N};

    int k_secs = div_up(k_per_core, k_slice);
    bool enable_core_interleave = k_secs * k_slice == k_per_core;
    for (int k_idx = 0; k_idx < k_secs; k_idx++) {
        ppl::enable_pipeline();
        int tmp_k_idx = enable_core_interleave ? (k_idx + core_idx) % k_secs : k_idx;
        int idx_k = tmp_k_idx * k_slice;
        int cur_k = min(k_slice, k_per_core - idx_k);

        // 左矩阵子块
        dim4 left_real_shape = {1, M, 1, cur_k};
        auto left_local = make_tensor<T>(left_block_shape, left_real_shape);
        dim4 left_offset = {0, 0, 0, idx_k + k_core_offset};
        dma::load(left_local, left_gtensor.sub_view(left_real_shape, left_offset));

        // 右矩阵子块
        dim4 right_real_shape = {1, N, 1, cur_k};
        auto right_local = make_tensor<T>(right_block_shape, right_real_shape);
        dim4 right_offset = {0, 0, 0, idx_k + k_core_offset};
        dma::load(right_local, right_gtensor.sub_view(right_real_shape, right_offset));

        tiu::fmm2(res_accum, left_local, right_local, false, true, false, false, true, DT_FP32);
    }

    tiu::cast(res_accum_bf16, res_accum, RM_HALF_AWAY_FROM_ZERO);
    sync();
    dma::reduce(res_l2, res_accum_bf16, ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
    sync();

    dma::move(res_gtensor, res_l2);
}

template <typename T>
void matmul_mn_mnk_kernel(T* out_ptr, T* left_ptr, T* right_ptr,
                         int M, int K, int N,
                         const int m_slice, const int n_slice, const int k_slice,
                         const int group_core) {
    ppl::set_core_num(CORE_NUM);
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    if (core_idx >= core_num) return;

    int cores_per_group = core_num / group_core;
    if (cores_per_group <= 0) return;

    int group_id = core_idx / cores_per_group;      // 组号，决定 M 分块
    int core_in_group = core_idx % cores_per_group; // 组内核号，决定 N 分块

    // --- M 维按组划分 ---
    int m_group_slice = div_up(M, group_core);
    int m_offset = group_id * m_group_slice;
    int cur_m_total = min(m_group_slice, M - m_offset);
    if (cur_m_total <= 0) return;

    // --- N 维在组内划分 ---
    int n_core_slice = div_up(N, cores_per_group);
    int n_per_core = min(n_core_slice, N - core_in_group * n_core_slice);
    int n_core_offset = core_in_group * n_core_slice;
    if (n_per_core <= 0) return;

    int m_secs = div_up(cur_m_total, m_slice);
    int n_secs = div_up(n_per_core, n_slice);
    int k_secs = div_up(K, k_slice);
    bool enable_core_interleave = n_secs * n_slice == n_per_core;

    // 创建全局张量
    dim4 left_global_shape = {1, M, 1, K};
    dim4 right_global_shape = {1, N, 1, K};
    dim4 res_global_shape   = {1, M, 1, N};

    auto left_gtensor  = gtensor<T>(left_global_shape, GLOBAL, left_ptr);
    auto right_gtensor = gtensor<T>(right_global_shape, GLOBAL, right_ptr);
    auto res_gtensor   = gtensor<T>(res_global_shape, GLOBAL, out_ptr);

    dim4 left_block_shape = {1, m_slice, 1, k_slice};
    dim4 right_block_shape = {1, n_slice, 1, k_slice};
    dim4 res_block_shape = {1, m_slice, 1, n_slice};

    for (int m_idx = 0; m_idx < m_secs; m_idx++) {
        int idx_m = m_offset + m_idx * m_slice;
        int cur_m = min(m_slice, cur_m_total - m_idx * m_slice);
        for (int n_idx = 0; n_idx < n_secs; n_idx++) {
            int idx_n = n_idx * n_slice;
            int cur_n = min(n_slice, n_per_core - idx_n);


            dim4 res_trans_block_shape = {1, m_slice, 1, n_slice};
            dim4 res_trans_real_shape = {1, cur_m, 1, cur_n};
            auto res_trans_local = make_tensor<fp32>(res_trans_block_shape, res_trans_real_shape);
            auto res_local = make_tensor<T>(res_trans_block_shape, res_trans_real_shape);
            tiu::zero(res_trans_local);
            for (int k_idx = 0; k_idx < k_secs; k_idx++) {
                ppl::enable_pipeline();
                int idx_k = k_idx * k_slice;
                int cur_k = min(k_slice, K - idx_k);

                // 加载左矩阵
                dim4 left_real_shape = {1, cur_m, 1, cur_k};
                auto left_local = make_tensor<T>(left_block_shape, left_real_shape);
                dim4 left_offset = {0, idx_m, 0, idx_k};
                dma::load(left_local, left_gtensor.sub_view(left_real_shape, left_offset));

                // 加载右矩阵
                dim4 right_real_shape = {1, cur_n, 1, cur_k};
                auto right_local = make_tensor<T>(right_block_shape, right_real_shape);
                dim4 right_offset = {0, idx_n + n_core_offset, 0, idx_k};
                dma::load(right_local, right_gtensor.sub_view(right_real_shape, right_offset));

                tiu::fmm2(res_trans_local, left_local, right_local, false, true, false, false, true, DT_FP32);
            }

            // 结果类型转换
            tiu::cast(res_local, res_trans_local, RM_HALF_TO_EVEN);

            // 存储结果
            dim4 res_offset = {0, idx_m, 0, idx_n + n_core_offset};
            dma::store(res_gtensor.sub_view(res_trans_real_shape, res_offset), res_local);
        }
    }
}


template <typename T>
void matmul_mn_mn_bias_kernel(T* out_ptr, T* left_ptr, T* right_ptr, T* bias_ptr, int M, int K, int N,
                         const int m_slice, const int n_slice, const int k_slice, const int group_core) {

    ppl::set_core_num(CORE_NUM);
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    if (core_idx >= core_num) return;

    int cores_per_group = core_num / group_core;
    if (cores_per_group <= 0) return;

    int group_id = core_idx / cores_per_group;      // 组号，决定 M 分块
    int core_in_group = core_idx % cores_per_group; // 组内核号，决定 N 分块

    // --- M 维按组划分 ---
    int m_group_slice = div_up(M, group_core);
    int m_offset = group_id * m_group_slice;
    int cur_m_total = min(m_group_slice, M - m_offset);
    if (cur_m_total <= 0) return;

    // --- N 维在组内划分 ---
    int n_core_slice = div_up(N, cores_per_group);
    int n_per_core = min(n_core_slice, N - core_in_group * n_core_slice);
    int n_core_offset = core_in_group * n_core_slice;
    if (n_per_core <= 0) return;

    int m_secs = div_up(cur_m_total, m_slice);
    int n_secs = div_up(n_per_core, n_slice);
    bool enable_core_interleave = n_secs * n_slice == n_per_core;

    // 创建全局张量
    dim4 left_global_shape = {1, M, 1, K};
    dim4 right_global_shape = {1, N, 1, K};
    dim4 bias_global_shape = {1, 1, 1, N};
    dim4 res_global_shape   = {1, M, 1, N};

    auto left_gtensor  = gtensor<T>(left_global_shape, GLOBAL, left_ptr);
    auto right_gtensor = gtensor<T>(right_global_shape, GLOBAL, right_ptr);
    auto bias_gtensor =  gtensor<T>(bias_global_shape, GLOBAL, bias_ptr);
    auto res_gtensor   = gtensor<T>(res_global_shape, GLOBAL, out_ptr);

    dim4 left_block_shape = {1, m_slice, 1, K};
    dim4 right_block_shape = {1, n_slice, 1, K};
    dim4 bias_block_shape = {1, LANE_NUM, 1, n_slice};
    dim4 res_block_shape = {1, m_slice, 1, n_slice};

    for (int m_idx = 0; m_idx < m_secs; m_idx++) {

        int idx_m = m_offset + m_idx * m_slice;
        int cur_m = min(m_slice, cur_m_total - m_idx * m_slice);
        dim4 left_real_shape = {1, cur_m, 1, K};

        auto left_local = make_tensor<T>(left_block_shape, left_real_shape);
        dim4 left_offset = {0, idx_m, 0, 0};
        dma::load(left_local, left_gtensor.sub_view(left_real_shape, left_offset));

        for (int n_idx = 0; n_idx < n_secs; n_idx++) {
            ppl::enable_pipeline();

            int tmp_n_idx = enable_core_interleave ? (n_idx + core_in_group) % n_secs : n_idx;
            int idx_n = tmp_n_idx * n_slice;
            int cur_n = min(n_slice, n_per_core - idx_n);

            // 加载右矩阵
            dim4 right_real_shape = {1, cur_n, 1, K};
            auto right_local = make_tensor<T>(right_block_shape, right_real_shape);
            dim4 right_offset = {0, idx_n + n_core_offset, 0, 0};
            dma::load(right_local, right_gtensor.sub_view(right_real_shape, right_offset));

            // 加载偏置
            dim4 bias_bc_shape = {1, LANE_NUM, 1, cur_n};
            dim4 bias_real_shape = {1, 1, 1, cur_n};
            auto bias_local = make_tensor<T>(bias_block_shape, bias_bc_shape);
            dim4 bias_offset = {0, 0, 0, idx_n + n_core_offset};
            dma::load_broadcast(bias_local, bias_gtensor.sub_view(bias_real_shape, bias_offset));
            auto bias_local_fp32 = make_tensor<fp32>(bias_block_shape, bias_bc_shape);
            tiu::cast(bias_local_fp32, bias_local, RM_HALF_TO_EVEN);

            // 矩阵乘法
            dim4 res_trans_block_shape = {1, m_slice, 1, n_slice};
            dim4 res_trans_real_shape = {1, cur_m, 1, cur_n};
            auto res_trans_local = make_tensor<fp32>(res_trans_block_shape, res_trans_real_shape);
            tiu::fmm2(res_trans_local, left_local, right_local, bias_local_fp32, false, true, false, false, false, DT_FP32, true);

            // 结果类型转换
            auto res_local  = make_tensor<T>(res_trans_block_shape, res_trans_real_shape);
            tiu::cast(res_local, res_trans_local, RM_HALF_TO_EVEN);

            // 存储结果
            dim4 res_real_shape   = {1, cur_m, 1, cur_n};
            dim4 res_offset = {0, idx_m, 0, idx_n + n_core_offset};
            dma::store(res_gtensor.sub_view(res_real_shape, res_offset), res_local);
        }
    }
}


template <typename T>
void matmul_k_mn_bias_kernel(
        T* out_ptr, T* left_ptr, T* right_ptr, T* bias_ptr, int M, int K, int N,
        const int m_slice, const int n_slice, const int k_slice) {

    ppl::set_core_num(CORE_NUM);
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    if (core_idx >= core_num) return;

    int k_core_slice = div_up(K, core_num);
    int k_per_core = min(k_core_slice, K - core_idx * k_core_slice);
    int k_core_offset = core_idx * k_core_slice;

    // 全局张量
    dim4 left_global_shape = {1, M, 1, K};
    dim4 right_global_shape = {1, N, 1, K};
    dim4 bias_global_shape = {1, 1, 1, N};
    dim4 res_global_shape   = {1, M, 1, N};

    auto left_gtensor  = gtensor<T>(left_global_shape,  GLOBAL, left_ptr);
    auto right_gtensor = gtensor<T>(right_global_shape, GLOBAL, right_ptr);
    auto bias_gtensor =  gtensor<T>(bias_global_shape, GLOBAL, bias_ptr);
    auto res_gtensor   = gtensor<T>(res_global_shape,   GLOBAL, out_ptr);
    auto res_l2   = gtensor<T>(res_global_shape, L2);
    dma::zero(res_l2);

    dim4 res_accum_shape   = {1, M, 1, N};
    auto res_accum = make_tensor<fp32>(res_accum_shape, res_accum_shape);
    auto res_accum_bf16 = make_tensor<T>(res_accum_shape, res_accum_shape);
    tiu::zero(res_accum);
    tiu::zero(res_accum_bf16);

    if(k_per_core <= 0) {
      sync();
      dma::reduce(res_l2, res_accum_bf16, ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
      sync();
      dma::move(res_gtensor, res_l2);
      return;
    }

    dim4 left_block_shape = {1, m_slice, 1, k_core_slice};
    dim4 right_block_shape = {1, n_slice, 1, k_core_slice};
    dim4 bias_block_shape = {1, LANE_NUM, 1, n_slice};
    dim4 res_block_shape = {1, m_slice, 1, n_slice};

    int m_secs = div_up(M, m_slice);
    int n_secs = div_up(N, n_slice);

    for (int m_idx = 0; m_idx < m_secs; m_idx++) {
        int idx_m = m_idx * m_slice;
        int cur_m = min(m_slice, M - idx_m);
        dim4 left_real_shape = {1, cur_m, 1, k_per_core};

        // 加载左矩阵
        auto left_local = make_tensor<T>(left_block_shape, left_real_shape);
        dim4 left_offset = {0, idx_m, 0, k_core_offset};
        dma::load(left_local, left_gtensor.sub_view(left_real_shape, left_offset));

        for(int n_idx = 0; n_idx < n_secs; n_idx++){
            ppl::enable_pipeline();
            int idx_n = n_idx * n_slice;
            int cur_n = min(n_slice, N - idx_n);

            // 加载右矩阵
            dim4 right_real_shape = {1, cur_n, 1, k_per_core};
            auto right_local = make_tensor<T>(right_block_shape, right_real_shape);
            dim4 right_offset = {0, idx_n, 0, k_core_offset};
            dma::load(right_local, right_gtensor.sub_view(right_real_shape, right_offset));

            // 矩阵乘法
            dim4 res_trans_block_shape = {1, m_slice, 1, n_slice};
            dim4 res_trans_real_shape = {1, cur_m, 1, cur_n};
            auto res_trans_local = make_tensor<fp32>(res_trans_block_shape, res_trans_real_shape);
            if(core_idx==0){
                // 加载偏置
                dim4 bias_real_shape = {1, 1, 1, cur_n};
                dim4 bias_bc_shape = {1, LANE_NUM, 1, cur_n};
                auto bias_local = make_tensor<T>(bias_block_shape, bias_bc_shape);
                dim4 bias_offset = {0, 0, 0, idx_n};
                dma::load_broadcast(bias_local, bias_gtensor.sub_view(bias_real_shape, bias_offset));
                auto bias_local_fp32 = make_tensor<fp32>(bias_block_shape, bias_bc_shape);
                tiu::cast(bias_local_fp32, bias_local, RM_HALF_TO_EVEN);
                tiu::fmm2(res_trans_local, left_local, right_local, bias_local_fp32, false, true, false, false, false, DT_FP32, true);
            }
            else{
                tiu::fmm2(res_trans_local, left_local, right_local, false, true, false, false, false, DT_FP32);
            }

            // 存储结果
            dim4 res_offset = {0, idx_m, 0, idx_n};
            dma::move(res_accum.sub_view(res_trans_real_shape, res_offset), res_trans_local);
        }
    }

    tiu::cast(res_accum_bf16, res_accum, RM_HALF_AWAY_FROM_ZERO);
    sync();
    dma::reduce(res_l2, res_accum_bf16, ALL_REDUCE_PSUM_WR, ALL_REDUCE_ADD);
    sync();

    dma::move(res_gtensor, res_l2);
}

template <typename T>
void matmul_mn_mnk_bias_kernel(T* out_ptr, T* left_ptr, T* right_ptr, T* bias_ptr,
                         int M, int K, int N,
                         const int m_slice, const int n_slice, const int k_slice,
                         const int group_core) {
    ppl::set_core_num(CORE_NUM);
    int core_num = ppl::get_core_num();
    int core_idx = ppl::get_core_index();
    if (core_idx >= core_num) return;

    int cores_per_group = core_num / group_core;
    if (cores_per_group <= 0) return;

    int group_id = core_idx / cores_per_group;      // 组号，决定 M 分块
    int core_in_group = core_idx % cores_per_group; // 组内核号，决定 N 分块

    // --- M 维按组划分 ---
    int m_group_slice = div_up(M, group_core);
    int m_offset = group_id * m_group_slice;
    int cur_m_total = min(m_group_slice, M - m_offset);
    if (cur_m_total <= 0) return;

    // --- N 维在组内划分 ---
    int n_core_slice = div_up(N, cores_per_group);
    int n_per_core = min(n_core_slice, N - core_in_group * n_core_slice);
    int n_core_offset = core_in_group * n_core_slice;
    if (n_per_core <= 0) return;

    int m_secs = div_up(cur_m_total, m_slice);
    int n_secs = div_up(n_per_core, n_slice);
    int k_secs = div_up(K, k_slice);
    bool enable_core_interleave = n_secs * n_slice == n_per_core;

    // 创建全局张量
    dim4 left_global_shape = {1, M, 1, K};
    dim4 right_global_shape = {1, N, 1, K};
    dim4 bias_global_shape = {1, 1, 1, N};
    dim4 res_global_shape   = {1, M, 1, N};

    auto left_gtensor  = gtensor<T>(left_global_shape, GLOBAL, left_ptr);
    auto right_gtensor = gtensor<T>(right_global_shape, GLOBAL, right_ptr);
    auto bias_gtensor =  gtensor<T>(bias_global_shape, GLOBAL, bias_ptr);
    auto res_gtensor   = gtensor<T>(res_global_shape, GLOBAL, out_ptr);

    dim4 left_block_shape = {1, m_slice, 1, k_slice};
    dim4 right_block_shape = {1, n_slice, 1, k_slice};
    dim4 bias_block_shape = {1, LANE_NUM, 1, n_slice};
    dim4 res_block_shape = {1, m_slice, 1, n_slice};

    for (int m_idx = 0; m_idx < m_secs; m_idx++) {
        int idx_m = m_offset + m_idx * m_slice;
        int cur_m = min(m_slice, cur_m_total - m_idx * m_slice);
        for (int n_idx = 0; n_idx < n_secs; n_idx++) {
            int idx_n = n_idx * n_slice;
            int cur_n = min(n_slice, n_per_core - idx_n);


            dim4 res_trans_block_shape = {1, m_slice, 1, n_slice};
            dim4 res_trans_real_shape = {1, cur_m, 1, cur_n};
            auto res_trans_local = make_tensor<fp32>(res_trans_block_shape, res_trans_real_shape);
            auto res_local = make_tensor<T>(res_trans_block_shape, res_trans_real_shape);
            tiu::zero(res_trans_local);
            for (int k_idx = 0; k_idx < k_secs; k_idx++) {
                ppl::enable_pipeline();
                int idx_k = k_idx * k_slice;
                int cur_k = min(k_slice, K - idx_k);

                // 加载左矩阵
                dim4 left_real_shape = {1, cur_m, 1, cur_k};
                auto left_local = make_tensor<T>(left_block_shape, left_real_shape);
                dim4 left_offset = {0, idx_m, 0, idx_k};
                dma::load(left_local, left_gtensor.sub_view(left_real_shape, left_offset));

                // 加载右矩阵
                dim4 right_real_shape = {1, cur_n, 1, cur_k};
                auto right_local = make_tensor<T>(right_block_shape, right_real_shape);
                dim4 right_offset = {0, idx_n + n_core_offset, 0, idx_k};
                dma::load(right_local, right_gtensor.sub_view(right_real_shape, right_offset));

                if(k_idx==0){
                    // 加载偏置
                    dim4 bias_bc_shape = {1, LANE_NUM, 1, cur_n};
                    dim4 bias_real_shape = {1, 1, 1, cur_n};
                    auto bias_local = make_tensor<T>(bias_block_shape, bias_bc_shape);
                    dim4 bias_offset = {0, 0, 0, idx_n + n_core_offset};
                    dma::load_broadcast(bias_local, bias_gtensor.sub_view(bias_real_shape, bias_offset));
                    auto bias_local_fp32 = make_tensor<fp32>(bias_block_shape, bias_bc_shape);
                    tiu::cast(bias_local_fp32, bias_local, RM_HALF_TO_EVEN);
                    tiu::fmm2(res_trans_local, left_local, right_local, bias_local_fp32, false, true, false, false, false, DT_FP32, true);

                }
                else{
                    tiu::fmm2(res_trans_local, left_local, right_local, false, true, false, false, true, DT_FP32);
                }

            }
            // 存储结果
            dim4 res_offset = {0, idx_m, 0, idx_n + n_core_offset};
            dma::store(res_gtensor.sub_view(res_trans_real_shape, res_offset), res_local);
        }
    }
}

__KERNEL__ void matmul_mn_mn( bf16* out_ptr, bf16* left_ptr, bf16* right_ptr, int M, int K, int N, 
                              const int m_slice, const int n_slice, const int k_slice, const int group_core) {
    matmul_mn_mn_kernel(out_ptr, left_ptr, right_ptr, M, K, N, m_slice, n_slice, k_slice, group_core);
}

__KERNEL__ void matmul_mn_k(bf16* out_ptr, bf16* left_ptr, bf16* right_ptr, int M, int K, int N,
                                  const int m_slice, const int n_slice, const int k_slice, const int group_core) {
    matmul_mn_k_kernel(out_ptr, left_ptr, right_ptr, M, K, N, m_slice, n_slice, k_slice, group_core);
}

__KERNEL__ void matmul_k_mn( bf16* out_ptr, bf16* left_ptr, bf16* right_ptr, int M, int K, int N,
                                const int m_slice, const int n_slice, const int k_slice) {
    matmul_k_mn_kernel(out_ptr, left_ptr, right_ptr, M, K, N, m_slice, n_slice, k_slice);
}

__KERNEL__ void matmul_k_k(bf16* out_ptr, bf16* left_ptr, bf16* right_ptr,int M, int K, int N, const int m_slice, const int n_slice, const int k_slice) {
    matmul_k_k_kernel(out_ptr, left_ptr, right_ptr,M, K, N, m_slice, n_slice, k_slice);
}

__KERNEL__ void matmul_mn_mnk(bf16* out_ptr, bf16* left_ptr, bf16* right_ptr,
                                   int M, int K, int N,
                                  const int m_slice, const int n_slice, const int k_slice, const int group_core) {
    matmul_mn_mnk_kernel(out_ptr, left_ptr, right_ptr, M, K, N, m_slice, n_slice, k_slice, group_core);
}

__KERNEL__ void matmul_mn_mn_bias( bf16* out_ptr, bf16* left_ptr, bf16* right_ptr, bf16* bias_ptr,
                                   int M, int K, int N, const int m_slice, const int n_slice, const int k_slice, const int group_core) {
    matmul_mn_mn_bias_kernel(out_ptr, left_ptr, right_ptr, bias_ptr, M, K, N, m_slice, n_slice, k_slice, group_core);
}

__KERNEL__ void matmul_k_mn_bias(
        bf16* out_ptr, bf16* left_ptr, bf16* right_ptr, bf16* bias_ptr, int M, int K, int N,
        const int m_slice, const int n_slice, const int k_slice) {
    matmul_k_mn_bias_kernel(out_ptr, left_ptr, right_ptr, bias_ptr, M, K, N, m_slice, n_slice, k_slice);
}

__KERNEL__ void matmul_mn_mnk_bias(bf16* out_ptr, bf16* left_ptr, bf16* right_ptr, bf16* bias_ptr,
                                   int M, int K, int N,
                                  const int m_slice, const int n_slice, const int k_slice, const int group_core) {
    matmul_mn_mnk_bias_kernel(out_ptr, left_ptr, right_ptr, bias_ptr, M, K, N, m_slice, n_slice, k_slice, group_core);
}
