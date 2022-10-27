#ifndef ATOMIC_RQDQ_H_
#define ATOMIC_RQDQ_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void atomic_rqdq(int node_idx, P_COMMAND cmd);

#ifdef __cplusplus
}
#endif

#endif