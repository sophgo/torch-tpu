#ifndef ATOMIC_ROI_DEPTHWISE_H
#define ATOMIC_ROI_DEPTHWISE_H

#include "cmodel_common.h"
#include "atomic_pooling_depthwise_core.h"

#ifdef __cplusplus
extern "C" {
#endif

void atomic_roi_depthwise(
    int nodechip_idx,
    P_COMMAND p_command);

#ifdef __cplusplus
}
#endif

#endif /* ATOMIC_ROI_DEPTHWISE_H */