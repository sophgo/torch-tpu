#pragma once

#include "common_def.h"

extern "C"
{

enum tpudnnStatus_t {
    TPUDNN_STATUS_SUCCESS,
    TPUDNN_STATUS_FAILED
};

typedef void *tpudnnHandle_t;

tpudnnHandle_t tpudnnCreate(int deviceID = 0);
void tpudnnDestroy(tpudnnHandle_t handle);

void *tpudnnPhysToVirt(tpudnnHandle_t handle, void *addr);
void *tpudnnVirtToPhys(tpudnnHandle_t handle, void *addr);

// TODO
//tpudnnHandle_t handle_from_bmlib();
//tpudnnHandle_t handle_from_tpurt();

tpudnnStatus_t tpudnnActive(
    tpudnnHandle_t handle,
    void *input,
    void *output,
    const int* shape,
    int shape_dim,
    sg_active_type_t active_type,
    const float* coeff,
    sg_data_type_t dtype);

tpudnnStatus_t tpudnnActiveMultiCore(
    tpudnnHandle_t      handle,
    void *              input,
    void *              output,
    const int*          shape,
    int                 dims,
    sg_active_type_t    active_type,
    const float*        coeff,
    sg_data_type_t      dtype);

tpudnnStatus_t tpudnnMatmulMultiCore(
    tpudnnHandle_t      handle,
    void *              left,
    void *              right,
    void *              output,
    const int*          L_shape,
    const int*          R_shape,
    const int           L_dims,
    const int           R_dims,
    const int           L_trans,
    const int           R_trans,
    sg_data_type_t      in_dtype,
    sg_data_type_t      out_dtype,
    int                 slice_m_core,
    int                 slice_n_core,
    int                 slice_m,
    int                 slice_n,
    int                 slice_k,
    const int           left_slice_dim,
    const int           right_slice_dim,
    const int           result_slice_dim,
    const int           left_8ch_buf_size,
    const int           right_8ch_buf_size,
    const int           result_8ch_buf_size,
    char*               left_8ch_buf[8],
    char*               right_8ch_buf[8],
    char*               result_8ch_buf[8]);

}

#if defined(__sg2260__)
tpudnnStatus_t tpudnnC2CAllReduce(
    tpudnnHandle_t      handle,
    void *              send_buff,
    void *              recv_buff,
    int                 count,
    sg_data_type_t      dtype,
    sg_reduce_method_t  reduce_method,
    sccl_args_t         sccl_args);

tpudnnStatus_t tpudnnC2CReduce(
    tpudnnHandle_t      handle,
    void *              send_buff,
    void *              recv_buff,
    int                 count,
    sg_data_type_t      dtype,
    sg_reduce_method_t  reduce_method,
    int                 root,
    sccl_args_t         sccl_args);

tpudnnStatus_t tpudnnC2CGather(
    tpudnnHandle_t      handle,
    void *              send_buff,
    int                 send_count,
    void *              recv_buff,
    int                 recv_count,
    sg_data_type_t      dtype,
    int                 root,
    sccl_args_t         sccl_args);

tpudnnStatus_t tpudnnC2CAllGather(
    tpudnnHandle_t      handle,
    void *              send_buff,
    int                 send_count,
    void *              recv_buff,
    int                 recv_count,
    sg_data_type_t      dtype,
    sccl_args_t         sccl_args);

tpudnnStatus_t tpudnnC2CBroadcast(
    tpudnnHandle_t      handle,
    void *              buff,
    int                 count,
    sg_data_type_t      dtype,
    int                 root,
    sccl_args_t         sccl_args);

tpudnnStatus_t tpudnnC2CScatter(
    tpudnnHandle_t      handle,
    void *              send_mem,
    int                 send_count,
    sg_data_type_t      send_type,
    void *              recv_mem,
    int                 recv_count,
    sg_data_type_t      recv_type,
    int                 root,
    sccl_args_t         sccl_args);

tpudnnStatus_t tpudnnC2CAllToAll(
    tpudnnHandle_t      handle,
    void *              send_mem,
    int                 send_count,
    sg_data_type_t      send_type,
    void *              recv_mem,
    int                 recv_count,
    sg_data_type_t      recv_type,
    sccl_args_t         sccl_args);
#endif