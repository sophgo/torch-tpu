#ifndef ATOMIC_SYS_H_
#define ATOMIC_SYS_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void atomic_sys(int node_idx, P_COMMAND cmd);

#ifdef __cplusplus
}
#endif

#endif