#include "tpu_kernel.h"
#include "math.h"
#include <string.h>

// #define DEBUG_TRANSPOSE
#define FW_MAX_SHAPE_DIMS 8

#define GET_OPTIMIZED_FACTORIZATION(factor, polynomial)       \
    {                                                         \
        factor = 0;                                           \
        if(polynomial > NPU_NUM){                             \
            for(int i = NPU_NUM; i >= (NPU_NUM/2); i--) {     \
                if(polynomial % i == 0) {                     \
                    factor = i;                               \
                    break;                                    \
                }                                             \
            }                                                 \
        }                                                     \
    }

#define HALF_LOCAL_MEM_SIZE  (LOCAL_MEM_SIZE >> 1)
#define QUARTER_LOCAL_MEM_SIZE  (LOCAL_MEM_SIZE >> 2)
typedef enum {
    TRANS_GENERAL = 0,
    TRANS_NPU_N_SWITCH_W,
    TRANS_GDMA_NCH,
    TRANS_NPU_H_SWITCH_W,
} trans_axis;

typedef struct {
    trans_axis trans_method;
    int N;
    int C;
    int H;
    int W;
    int max_trans_counts;
} trans_info_t;

inline static void pipeline_move(int *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

trans_info_t get_transpose_info(int fixed_dim, int left_dim, int right_dim, int type_len)
{
    trans_info_t trans_info;
    int factor = 0;
    bool use_left_dim = false;
    bool use_right_dim = false;
    int min_transpose_size = 0;
    float overflow = 0.f;
    if(left_dim > right_dim) {
        GET_OPTIMIZED_FACTORIZATION(factor, left_dim)
        if((factor > 0) && (factor != left_dim) && (fixed_dim < left_dim)) {
             use_left_dim = true;
        } else {
            GET_OPTIMIZED_FACTORIZATION(factor, right_dim)
            if((factor > 0) && (factor != right_dim) && (fixed_dim == 1)) {
                use_right_dim = true;
            }
        }
    } else {
        GET_OPTIMIZED_FACTORIZATION(factor, right_dim)
        if((factor > 0) && (factor != right_dim) && (fixed_dim < right_dim)) {
            use_right_dim = true;
        } else {
            GET_OPTIMIZED_FACTORIZATION(factor, left_dim)
            if((factor > 0) && (factor != left_dim) && (fixed_dim == 1)) {
                use_left_dim = true;
            }
        }
    }
    // 当Y(left_dim)和Z(right_dim)不可以在c通道上因式分解，或者X(fixed_dim)大于Y/Z时（因为X较大时，可能转置需要的指令过多），
    // 尝试分解X到n和c通道, Y放到h通道，Z放到w通道，做hw通道的YZ转置
    if((use_left_dim == false && use_right_dim == false) ||
        (fixed_dim >= left_dim && fixed_dim >= right_dim)) {
        GET_OPTIMIZED_FACTORIZATION(factor, fixed_dim)
        if(factor > 0) {
            trans_info.N = 1; // fixed_dim/factor;
            trans_info.C = factor;
            trans_info.H = left_dim;
            trans_info.W = right_dim;
            min_transpose_size = trans_info.H * trans_info.W  * type_len;
            //计算一个NPU的 1/4 local mem size 是否能够容下一次最小转置数据
            overflow = ((float)((float)min_transpose_size)/(float)QUARTER_LOCAL_MEM_SIZE);

            //计算最大转置次数
            trans_info.max_trans_counts = fixed_dim/factor;
        } else if (fixed_dim <= NPU_NUM) {
            trans_info.N = 1;
            trans_info.C = fixed_dim;
            trans_info.H = left_dim;
            trans_info.W = right_dim;
            min_transpose_size = trans_info.H * trans_info.W  * type_len;
            overflow = ((float)((float)min_transpose_size)/(float)QUARTER_LOCAL_MEM_SIZE);
            trans_info.max_trans_counts = 1;
        }
        trans_info.trans_method = TRANS_NPU_H_SWITCH_W;

    // 当Y较大并能够在c通道上因式分解， 则将Y分解到c和h通道上，Z放到w通道上，做n和w通道的转置，即完成Y和Z的转置
    } else if(use_left_dim){
        #ifdef DEBUG_TRANSPOSE
        printf("+++++++++++++++++++++++I'm here!++++++++++++++++++++++++++++NW calc left\n");
        #endif
        trans_info.N = 1;
        trans_info.C = factor;
        trans_info.H = left_dim/factor;
        trans_info.W = right_dim;
        trans_info.trans_method = TRANS_NPU_N_SWITCH_W;
        min_transpose_size = trans_info.H * trans_info.W * type_len;
        overflow = ((float)((float)min_transpose_size)/(float)QUARTER_LOCAL_MEM_SIZE);
        trans_info.max_trans_counts = fixed_dim;

     // 当Z较大并能够在c通道上因式分解， 则将Z分解到c和h通道上， Y放到n通道上，做n和w通道的转置，即完成Y和Z的转置
    } else if(use_right_dim) {
        #ifdef DEBUG_TRANSPOSE
        printf("+++++++++++++++++++++++I'm here!++++++++++++++++++++++++++++NW calc right\n");
        #endif      
        trans_info.N = left_dim;
        trans_info.C = factor;
        trans_info.H = right_dim/factor;
        trans_info.W = 1;
        trans_info.trans_method = TRANS_NPU_N_SWITCH_W;
        min_transpose_size = trans_info.H * trans_info.N * type_len;
        overflow = ((float)((float)min_transpose_size)/(float)QUARTER_LOCAL_MEM_SIZE);
        trans_info.max_trans_counts = fixed_dim;
    }

    // 如果TRANS_NPU_* 方法的最小转置数据导致local mem溢出，则使用TRANS_GENERAL方法做转置
    if(overflow > 1.0f || overflow == 0.f){
        memset(&trans_info, 0, sizeof(trans_info));
        trans_info.trans_method = TRANS_GENERAL;
    }
    return trans_info;
}

void nodechip_permute_xyz2xzy(
        global_addr_t       input_global_addr,
        global_addr_t       output_global_addr,
        int                 input_x,
        int                 input_y,
        int                 input_z,
        data_type_t         dtype)
{
    int type_len = tpu_data_type_size(dtype);
    unsigned int sum = NPU_NUM * NPU_NUM ,i = 1,j = 1;
    int max_x = input_x;
    unsigned int y = DIV_UP(input_y, NPU_NUM);
    unsigned int z = DIV_UP(input_z, NPU_NUM);
    // double buffer for in/out, so each is 4 banks
    unsigned int tensor_size = tpu_local_mem_size_per_npu() / 4;
    bool is_memory_overflow = true;
    while(is_memory_overflow ){
        unsigned int threshold = (unsigned int)floor((float) ((tensor_size >> 2)*NPU_NUM) / (float)max_x);
        while (sum <= threshold) {
            is_memory_overflow = false;
            if(i >= z && j >= y) {
                break;
            }
            if(i < z){
              i++;
              sum =   (i)*NPU_NUM*(j)*NPU_NUM;
              if(sum >= threshold) {
                  i--;
                  break;
              }
            }
            if(j < y){
                j++;
                sum = (i)*NPU_NUM*(j)*NPU_NUM;
                if(sum >= threshold) {
                    j--;
                    break;
              }
            }
        }
        if(!is_memory_overflow) break;
        if(max_x > 3)
            max_x = max_x - 3;
        else
            max_x = 1;
    }

    int max_z = i * NPU_NUM;
    int max_y = j * NPU_NUM;
    dim4 in_global_stride = {
        input_y * input_z,
        input_z,
        input_z,
        1
    };
    dim4 out_global_stride = {
        input_y * input_z,
        input_y,
        input_y,
        1
    };

    local_addr_t input_local_addr[2] = {0, tensor_size};
    local_addr_t output_local_addr[2] = {2 * tensor_size, 3 * tensor_size};
    int x_idx[3] = {0}, y_idx[3] = {0}, z_idx[3] = {0};
    int x_slice[3] = {0}, y_slice[3] = {0}, z_slice[3] = {0};
    int stage_idx = 0, draining_idx = 0;
    while(x_idx[2] < input_x) {
        tpu_parallel_start();
        // update load info
        if (draining_idx < 1) {
            x_slice[0] = MIN(input_x - x_idx[0], max_x);
            y_slice[0] = MIN(input_y - y_idx[0], max_y);
            z_slice[0] = MIN(input_z - z_idx[0], max_z);
        }
        // store output
        if (stage_idx > 1) {
            int x_offset = x_idx[2] * input_z * input_y;
            int z_offset = z_idx[2] * input_y;
            int y_offset = y_idx[2];
            global_addr_t cur_out_addr = output_global_addr + (x_offset + z_offset + y_offset) * type_len;
            dim4 shape = {x_slice[2], z_slice[2], 1, y_slice[2]};
            tpu_gdma_cpy_L2S(
                cur_out_addr,
                output_local_addr[stage_idx & 0x1],
                &shape,
                &out_global_stride,
                0x00,
                dtype);
        }

        // load input
        if (draining_idx < 1) {
            int x_offset = x_idx[0] * input_y * input_z;
            int y_offset = y_idx[0] * input_z;
            int z_offset = z_idx[0];
            global_addr_t cur_in_addr = input_global_addr + (x_offset + y_offset + z_offset) * type_len;
            dim4 shape = {x_slice[0], y_slice[0], 1, z_slice[0]};
            tpu_gdma_cpy_S2L(
                input_local_addr[stage_idx & 0x1],
                cur_in_addr,
                &shape,
                0x00,
                &in_global_stride,
                dtype);
        }

        // compute: use bdc transpose cw
        if (stage_idx > 0 && draining_idx < 2) {
            dim4 shape = {x_slice[1], z_slice[1], 1, y_slice[1]};
            if (y_slice[1] >= z_slice[1]) {
                tpu_bdc_cw_trans(
                    output_local_addr[(stage_idx - 1) & 0x1],
                    input_local_addr[(stage_idx - 1) & 0x1],
                    &shape,
                    dtype);
            } else {
                tpu_bdc_wc_trans(
                    output_local_addr[(stage_idx - 1) & 0x1],
                    input_local_addr[(stage_idx - 1) & 0x1],
                    &shape,
                    dtype);
            }
        }

        tpu_parallel_end();
        pipeline_move(x_idx, 3);
        pipeline_move(y_idx, 3);
        pipeline_move(z_idx, 3);
        pipeline_move(x_slice, 3);
        pipeline_move(y_slice, 3);
        pipeline_move(z_slice, 3);
        if (draining_idx < 1) {
            z_idx[0] += z_slice[0];
            if (z_idx[0] >= input_z) {
                z_idx[0] = 0;
                y_idx[0] += y_slice[0];
                if (y_idx[0] >= input_y) {
                    y_idx[0] = 0;
                    x_idx[0] += x_slice[0];
                    if (x_idx[0] >= input_x) {
                        draining_idx++;
                    }
                }
            }
        } else {
          draining_idx++;
        }
        stage_idx++;
    }
}

void nodechip_permute_gdma(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        const int* input_shape,
        const int* order,
        int  dims,
        data_type_t   dtype
){
    TPUKERNEL_ASSERT(dims >= 4 && "this require dims >= 4");
    TPUKERNEL_ASSERT(order[3] == 3 && "this require wstride == 1");
    unsigned int src_n_stride, src_c_stride, src_h_stride;
    unsigned int tmp_dst_stride[4];
    int i;
    unsigned int src_shape[4] = {0},out_stride[4];

    src_shape[0] = input_shape[0];
    src_shape[1] = input_shape[1];
    src_shape[2] = input_shape[2];
    src_shape[3] = input_shape[3];

    for(i = 4; i <  dims; i++) {
        src_shape[3] *= input_shape[i];
    }

    src_n_stride = src_shape[1]*src_shape[2]*src_shape[3];
    src_c_stride = src_shape[2]*src_shape[3];
    src_h_stride = src_shape[3];

    tmp_dst_stride[3] = 1;
    tmp_dst_stride[2] = src_shape[order[3]];
    tmp_dst_stride[1] = src_shape[order[2]] * src_shape[order[3]];
    tmp_dst_stride[0] = src_shape[order[1]] * src_shape[order[2]] * src_shape[order[3]];
    for(i = 0; i < 4; i++) {
        out_stride[order[i]] = tmp_dst_stride[i];
    }

    dim4 shape = {
             .n = src_shape[0],
             .c = src_shape[1],
             .h = src_shape[2],
             .w = src_shape[3]
           };
    dim4 src_stride = {src_n_stride, src_c_stride, src_h_stride, 1};
    dim4 dst_stride = {out_stride[0], out_stride[1], out_stride[2], 1};
    tpu_gdma_cpy_S2S(output_global_addr, input_global_addr, &shape, &dst_stride, &src_stride, dtype);
}

//transpose method : do N and W transpose.(N, C, H, W)  => (W, C, H, N)
void nodechip_permute_tpu_nw(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        int           N,
        int           C,
        int           H,
        int           W,
        int           total_trans_times,
        data_type_t   dtype)
{
    #ifdef DEBUG_TRANSPOSE
    printf("+++++++++++++++++++++++I'm here!++++++++++++++++++++++++++++NW\n");
    #endif
    int type_len = tpu_data_type_size(dtype);
    unsigned int tensor_size = QUARTER_LOCAL_MEM_SIZE;

    unsigned int slice_batches =(unsigned int)floor((float)((float)tensor_size / (float) (N * H * W * type_len))); //  / ( * c )
    // why no DIV_UP(C, NPU_NUM) here: C is guaranteed to be less than NPU_NUM in get_transpose_info()
    // if any further change results in a different c=DIV_UP(C, NPU_NUM), it should be added in proper places.
    TPUKERNEL_ASSERT(DIV_UP(C, NPU_NUM) == 1);
    unsigned int slice_total = total_trans_times;
    unsigned int lmem_in_nstride = H * W; // * c
    unsigned int lmem_out_nstride = H * N; // * c

    unsigned int in_stride[] = {
        C * H * W,
        H * W,
        W,
        1
    };
    unsigned int lmem_in_stride[] = {
        lmem_in_nstride,
        H * W,
        W,
        1
    };
    unsigned int lmem_out_stride[] = {
        lmem_out_nstride,
        H * N,
        N,
        1
    };
    unsigned int out_stride[] = {
        C * H * N,
        H * N,
        N,
        1
    };

    unsigned long long in_local_addr[2] = {0, tensor_size};
    unsigned long long out_local_addr[2] = {2 * tensor_size, 3 * tensor_size};
    unsigned long long slice_size = N * C * H * W * type_len;
    int cur_slice_idx[3] = {0}, cur_slice[3] = {0};
    int stage_idx = 0, draining_idx = 0;
    while(cur_slice_idx[2] < (int)slice_total) {
        tpu_parallel_start();
        //update load info
        if (draining_idx < 1) {
            cur_slice[0] = MIN(slice_total - cur_slice_idx[0], slice_batches);
        }

        // store output
        if (stage_idx > 1) {
            dim4 shape = {
                .n = cur_slice[2] * W,
                .c = C,
                .h = H,
                .w = N
            };
            dim4 src_stride = {
                .n = lmem_out_stride[0],
                .c = lmem_out_stride[1],
                .h = lmem_out_stride[2],
                .w = lmem_out_stride[3]
            };
            dim4 dst_stride = {
                .n = out_stride[0],
                .c = out_stride[1],
                .h = out_stride[2],
                .w = out_stride[3]
            };

            tpu_gdma_cpy_L2S(
                output_global_addr + cur_slice_idx[2] * slice_size, 
                out_local_addr[stage_idx & 0x1], 
                &shape, 
                &dst_stride, 
                &src_stride, 
                dtype);
        }
        
        // load input
        if (draining_idx < 1) {
            dim4 shape = {
               .n = cur_slice[0] * N,
               .c = C,
               .h = H,
               .w = W
            };
            dim4 src_stride = {
                .n = in_stride[0], 
                .c = in_stride[1], 
                .h = in_stride[2], 
                .w = in_stride[3]
            };
            dim4 dst_stride = {
                .n = lmem_in_stride[0], 
                .c = lmem_in_stride[1], 
                .h = lmem_in_stride[2], 
                .w = lmem_in_stride[3]
            };

            tpu_gdma_cpy_S2L(
                in_local_addr[stage_idx & 0x1], 
                input_global_addr + cur_slice_idx[0] * slice_size, 
                &shape, 
                &dst_stride, 
                &src_stride, 
                dtype);
        }

        // compute: bdc_cpy
        // get_transpose_info() has ensured that either N or W is 1.
        // Note that this method is empirical and will definitely NOT work when neither N nor W is 1.
        TPUKERNEL_ASSERT(N == 1 || W == 1);
        #ifdef DEBUG_TRANSPOSE
        if (N == 1){
            printf("+++++++++++++++++++++++I'm here!++++++++++++++++++++++++++++NW use left\n");
        }
        if (W == 1){
            printf("+++++++++++++++++++++++I'm here!++++++++++++++++++++++++++++NW use right\n");
        }
        #endif

        if (stage_idx > 0 && draining_idx < 2) {
            dim4 tensor_dim = {
                .n = cur_slice[1], 
                .c = C, 
                .h = H, 
                .w = N * W // N == 1 ? W : N
            };
            dim4 tensor_src_stride = {
                .n = H * N * W, // N == 1 ? H * W : H * N, // * c
                .c = H * W,
                .h = W,
                .w = N == 1 ? N : H // : * c
            };
            dim4 tensor_dst_stride = {
                .n = H * N * W, // N == 1 ? H * W : H * N, // * c
                .c = H * N,
                .h = N,
                .w = N == 1 ? H : W // * c :
            };
            tpu_bdc_cpy(
                out_local_addr[(stage_idx - 1) & 0x1],
                in_local_addr[(stage_idx - 1) & 0x1],
                &tensor_dim,
                &tensor_dst_stride,
                &tensor_src_stride,
                dtype);
        }

        tpu_parallel_end();
        pipeline_move(cur_slice_idx, 3);
        pipeline_move(cur_slice, 3);
        if (draining_idx < 1) {
            cur_slice_idx[0] += cur_slice[0];
            if (cur_slice_idx[0] >= (int)slice_total) {
                draining_idx++;
            }
        } else {
            draining_idx++;
        }
        stage_idx++;
   }
}

//transpose method : do H and W transpose.(N, C, H, W)  => (N, C, W, H)
void nodechip_permute_tpu_hw(
        global_addr_t input_global_addr,
        global_addr_t output_global_addr,
        int           N,
        int           C,
        int           H,
        int           W,
        int           total_trans_times,
        data_type_t   dtype)
{
    #ifdef DEBUG_TRANSPOSE
    printf("+++++++++++++++++++++++I'm here!++++++++++++++++++++++++++++HW\n");
    #endif
    int type_len = tpu_data_type_size(dtype);
    unsigned int tensor_size = QUARTER_LOCAL_MEM_SIZE;
    
    unsigned int slice_batches = (unsigned int)floor((float)((float)tensor_size / (float) (N * H * W * type_len))); // * c
    // why no DIV_UP(C, NPU_NUM) here: C is guaranteed to be less than NPU_NUM in get_transpose_info()
    // if any further change results in a different c=DIV_UP(C, NPU_NUM), it should be added in proper places.
    TPUKERNEL_ASSERT(DIV_UP(C, NPU_NUM) == 1);
    unsigned int slice_total = total_trans_times;
    unsigned int lmem_in_nstride = H * W; // * c
    unsigned int lmem_out_nstride = lmem_in_nstride;

    unsigned int in_stride[] = {
        C * H * W,
        H * W,
        W,
        1
    };
    unsigned int lmem_in_stride[] = {
        lmem_in_nstride,
        H * W,
        W,
        1
    };
    unsigned int lmem_out_stride[] = {
        lmem_out_nstride,
        H * W,
        H,
        1
    };
    unsigned int out_stride[] = {
        C * H * W,
        H * W,
        H,
        1
    };

    unsigned long long in_local_addr[2] = {0, tensor_size};
    unsigned long long out_local_addr[2] = {2 * tensor_size, 3 * tensor_size};
    unsigned long long slice_size = N * C * H * W * type_len;
    int cur_slice_idx[3] = {0}, cur_slice[3] = {0};
    int stage_idx = 0, draining_idx = 0;
    while(cur_slice_idx[2] < (int)slice_total) {
        #ifdef DEBUG_TRANSPOSE
        printf("+++++++++++++++++++++++I'm here!++++++++++++++++++++++++++++HW in cycle\n");
        #endif
        tpu_parallel_start();
        //update load info
        if (draining_idx < 1) {
            cur_slice[0] = MIN(slice_total - cur_slice_idx[0], slice_batches);
        }

        // store output
        if (stage_idx > 1) {
            dim4 shape = {
                .n = cur_slice[2] * N,
                .c = C,
                .h = W,
                .w = H
            };
            dim4 src_stride = {
                .n = lmem_out_stride[0],
                .c = lmem_out_stride[1],
                .h = lmem_out_stride[2],
                .w = lmem_out_stride[3]
            };
            dim4 dst_stride = {
                .n = out_stride[0],
                .c = out_stride[1],
                .h = out_stride[2],
                .w = out_stride[3]
            };
            tpu_gdma_cpy_L2S(
                output_global_addr + cur_slice_idx[2] * slice_size, 
                out_local_addr[stage_idx & 0x1], 
                &shape, 
                &dst_stride, 
                &src_stride, 
                dtype);
        }

        // load input
        if (draining_idx < 1) {
            dim4 shape = {
               .n = cur_slice[0] * N,
               .c = C,
               .h = H,
               .w = W
            };
            dim4 src_stride = {
                .n = in_stride[0], 
                .c = in_stride[1], 
                .h = in_stride[2], 
                .w = in_stride[3]
            };
            dim4 dst_stride = {
                .n = lmem_in_stride[0], 
                .c = lmem_in_stride[1], 
                .h = lmem_in_stride[2], 
                .w = lmem_in_stride[3]
            };

            tpu_gdma_cpy_S2L(
                in_local_addr[stage_idx & 0x1], 
                input_global_addr + cur_slice_idx[0] * slice_size, 
                &shape, 
                &dst_stride, 
                &src_stride, 
                dtype);
        }
        
        // compute: bdc_cpy
        if (stage_idx > 0 && draining_idx < 2) {
            dim4 tensor_dim = {
                .n = cur_slice[1] * N, 
                .c = C, 
                .h = H, 
                .w = W
            };
            // for NCHW
            dim4 tensor_src_stride = {
                .n = lmem_in_nstride,
                .c = H * W,
                .h = W,
                .w = 1
            };
            //for NCWH
            dim4 tensor_dst_stride = {
                .n = lmem_out_nstride,
                .c = H * W,
                .h = 1,
                .w = H
            };
            tpu_bdc_cpy(
                out_local_addr[(stage_idx - 1) & 0x1],
                in_local_addr[(stage_idx - 1) & 0x1],
                &tensor_dim,
                &tensor_dst_stride,
                &tensor_src_stride,
                dtype);
        }
        
        tpu_parallel_end();
        pipeline_move(cur_slice_idx, 3);
        pipeline_move(cur_slice, 3);
        if (draining_idx < 1) {
            cur_slice_idx[0] += cur_slice[0];
            if (cur_slice_idx[0] >= (int)slice_total) {
                draining_idx++;
            }
        } else {
            draining_idx++;
        }
        stage_idx++;   
   }
}
void nodechip_transpose(
    global_addr_t         input_global_addr,
    global_addr_t         output_global_addr,
    int*                  input_shape,
    int*                  order,
    int                   dims,
    global_addr_t         buffer_global_addr,
    unsigned long long*   buffer_size,
    data_type_t           dtype
) {
    TPUKERNEL_ASSERT(dims <= FW_MAX_SHAPE_DIMS);
    int type_len = tpu_data_type_size(dtype);

    int real_input_shape[FW_MAX_SHAPE_DIMS];
    memcpy(real_input_shape, input_shape, sizeof(int) * dims);

    int raw_order[FW_MAX_SHAPE_DIMS];
    int tmp_order[FW_MAX_SHAPE_DIMS];
    int steps[FW_MAX_SHAPE_DIMS][4];
    int step_num = 0;
    for (int i = 0; i < dims; i++) raw_order[i] = i;

    int real_dims = dims;
    int fixed_dim = 1; //不需要置换的维度或已经置换好的维度，指向shape的低维度
    int left_dim = 1;  //将要置换的左边的维度，即后面函数中y的维度，指向shape的中间维度
    int right_dim = 1; //将要置换的右边的维度，即后面函数中z的维度，指向shape的高维度

    trans_axis trans_method[FW_MAX_SHAPE_DIMS] = {0};
    int   trans_total_times[FW_MAX_SHAPE_DIMS] = {0};
    for (int i = 0; i < real_dims; i++) {
        if (order[i] == raw_order[i]) {
            fixed_dim *= real_input_shape[order[i]];
            continue;
        }
        int pivot = 0;
        for (int j = i + 1; j < real_dims; j++) {
            if (order[i] == raw_order[j]) {
                pivot = j - i;
                break;
            }
        }

        left_dim = 1;
        right_dim = 1;
        for (int j = i; j < real_dims; j++) {
            if (j < pivot + i) {
                left_dim *= real_input_shape[raw_order[j]];
            } else {
                right_dim *= real_input_shape[raw_order[j]];
            }
            tmp_order[j] = (j + pivot < real_dims) ? raw_order[j + pivot] : raw_order[i + j + pivot - real_dims];
        }
        for (int j = i; j < real_dims; j++) {
            raw_order[j] = tmp_order[j];
        }
        if(left_dim != 1 && right_dim != 1) {
            trans_info_t trans_info = get_transpose_info(fixed_dim, left_dim, right_dim, type_len);
            if( trans_info.trans_method == TRANS_GENERAL) {
                steps[step_num][0] = fixed_dim;
                steps[step_num][1] = left_dim;
                steps[step_num][2] = right_dim;
            } else {
                steps[step_num][0] = trans_info.N;
                steps[step_num][1] = trans_info.C;
                steps[step_num][2] = trans_info.H;
                steps[step_num][3] = trans_info.W;
                trans_total_times[step_num] = trans_info.max_trans_counts;
                trans_method[step_num] = trans_info.trans_method;
            }
            step_num++;
        }
        fixed_dim *= real_input_shape[order[i]];
    }
    if (buffer_size) {
        *buffer_size = 0;
        if (step_num > 1) {
            *buffer_size = type_len * input_shape[0];
            for (int i = 1; i < dims; i++) {
                *buffer_size *= input_shape[i];
            }
        }
        return;
    }

    if (step_num == 0) {
        // no need to transpose, just copy
        unsigned int tensor_size = (unsigned int)real_input_shape[0];
        for(int i = 1; i <  dims; i++) {
            tensor_size *= (unsigned int)real_input_shape[i];
        }
        tpu_gdma_system_cpy(output_global_addr, input_global_addr, tensor_size , dtype);
        return;
    }

    for(int i = 3; i < dims; i++) {
        if(i != order[i]) break;   // for meetting src/dst wstride==1 to use gdma neuron
        if(i == (dims -1)) {
            step_num = 1;
            trans_method[0] = TRANS_GDMA_NCH;
        }
    }

    unsigned long long permute_in_addr = input_global_addr;
    unsigned long long permute_out_addr = (step_num & 0x1) ? output_global_addr: buffer_global_addr;
    for (int i = 0; i < step_num; i++) {
        if(trans_method[i] == TRANS_NPU_N_SWITCH_W){
            nodechip_permute_tpu_nw(
                permute_in_addr,
                permute_out_addr,
                steps[i][0],
                steps[i][1],
                steps[i][2],
                steps[i][3],
                trans_total_times[i],
                dtype);
        } else if(trans_method[i] == TRANS_GDMA_NCH) {
           nodechip_permute_gdma(
                 input_global_addr,
                 output_global_addr,
                 input_shape,
                 order,
                 dims,
                 dtype);
        } else if(trans_method[i] == TRANS_NPU_H_SWITCH_W) {
            nodechip_permute_tpu_hw(
                permute_in_addr,
                permute_out_addr,
                steps[i][0],
                steps[i][1],
                steps[i][2],
                steps[i][3],
                trans_total_times[i],
                dtype);
        } else {
            nodechip_permute_xyz2xzy(
                permute_in_addr,
                permute_out_addr,
                steps[i][0],
                steps[i][1],
                steps[i][2],
                dtype);
        }
        permute_in_addr = permute_out_addr;
        permute_out_addr = (permute_out_addr == buffer_global_addr) ? output_global_addr : buffer_global_addr;
    }
}
