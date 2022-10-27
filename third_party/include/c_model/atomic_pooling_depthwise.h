#ifndef ATOMIC_POOLING_DEPTHWISE_H
#define ATOMIC_POOLING_DEPTHWISE_H

#include "cmodel_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void atomic_pooling_depthwise(
    int nodechip_idx,
    P_COMMAND p_command);

#ifdef __cplusplus
}
#endif

#endif /* ATOMIC_POOLING_DEPTHWISE_H */