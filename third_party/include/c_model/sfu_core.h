#ifndef ATOMIC_SFU_CORE_H
#define ATOMIC_SFU_CORE_H
#include "common.h"

DataUnion microp_rsqrt(DataUnion x, int iter_num);
DataUnion microop_normal(DataUnion x, PREC in_prec, PREC out_prec);
DataUnion microop_tailor(DataUnion x, PREC in_prec, int coeff_len, void *p_coeff);

#endif /* ATOMIC_SFU_CORE_H */


