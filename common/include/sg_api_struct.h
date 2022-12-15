#ifndef SG_API_STRUCT_H
#define SG_API_STRUCT_H

#pragma pack(push, 1)
#include "common_def.h"

#ifndef FW_MAX_SHAPE_DIMS
#define FW_MAX_SHAPE_DIMS      8
#endif
#define MAX_CONCATLAYER_INPUT_NUM 10
typedef struct {
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long grad_output_global_addr;
    unsigned long long grad_input_global_addr;
    unsigned long long grad_weight_global_addr;
    unsigned long long grad_bias_global_addr;
    unsigned long long buffer_global_addr;
    int                input_shape[4];
    int                output_shape[4];
    int                kernel[2];
    int                stride[2];
    int                dilation[2];
    int                pad[4];
    int                grad_input_enable;
    int                grad_weight_enable;
    int                grad_bias_enable;
#ifndef WIN32
} __attribute__((packed)) sg_api_conv_backward_t;
#else
} sg_api_conv_backward_t;
#endif

typedef struct{
    unsigned long long grad_output_global_addr;
    unsigned long long input_global_addr;
    unsigned long long weight_global_addr;
    unsigned long long saved_mean_global_addr;
    unsigned long long saved_invstd_global_addr;
    unsigned long long grad_input_global_addr;
    unsigned long long grad_weight_global_addr;
    unsigned long long grad_bias_global_addr;
    int                shape[4];
    int                grad_input_enable;
    int                grad_weight_enable;
    int                grad_bias_enable;
#ifndef WIN32
} __attribute__((packed)) sg_api_batchnorm_backward_t;
#else
} sg_api_batchnorm_backward_t;
#endif

#pragma pack(pop)
#endif  // SG_API_STRUCT_H
