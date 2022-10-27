#ifndef ATOMIC_SG_H_
#define ATOMIC_SG_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void atomic_sg(int node_idx, P_COMMAND cmd);
void atomic_sgl(int node_idx, P_COMMAND cmd);

#ifdef __cplusplus
}
#endif

#endif