#ifndef NODECHIP_BIANRY_H_
#define NODECHIP_BIANRY_H_

#include "sg_api_struct.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

// eltwise binary and broadcast binary
void nodechip_bcbinary_fp(
    global_addr_t A_global_addr,
    global_addr_t B_global_addr,
    global_addr_t res_global_addr,
    const int* A_shape,
    const int* B_shape,
    int A_dim,
    int B_dim,
    int binary_type,
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit);

void nodechip_const_binary_fp(
    global_addr_t A_global_addr,
    global_addr_t res_global_addr,
    const int* shape,
    int shape_dim,
    float B_const_val,
    int inversed,
    int binary_type,
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit);

#ifdef __cplusplus
}
#endif

#endif
