#ifndef ATOMIC_POOLING_H
#define ATOMIC_POOLING_H

#include "cmodel_common.h"
#include "atomic_pooling_depthwise_core.h"

#ifdef __cplusplus
extern "C" {
#endif

void atomic_pooling(
    int nodechip_idx,
    P_COMMAND p_command);

#ifdef __cplusplus
}
#endif

#endif /* ATOMIC_POOLING_H */