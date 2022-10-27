#ifndef  ATOMIC_FUSED_CMP_H
#define  ATOMIC_FUSED_CMP_H

#include "common.h"

typedef struct fused_cmp_param {
#ifdef SG_TV_GEN
  u32 n;
  u32 c;
  u32 h;
  u32 w;
  u32 A_addr;
  u32 B_addr;
  u32 C_addr;
  u32 D_addr;
  u32 R0_addr;
  u32 R1_addr;
#endif
  u32 size;
  u32 a_is_const;
  u32 b_is_const;
  u32 c_is_const;
  u32 d_is_const;
  u32 a_constant;
  u32 b_constant;
  u32 c_constant;
  u32 d_constant;
  u32 a_short_str;
  u32 b_short_str;
  PREC a_precision; // tensorA, tensorB
  PREC c_precision; // tensorC, tensorD
  u32 a_is_sign;   // tensorA, tensorB
  u32 eu_type;
} FUSED_CMP_PARAM;

void fused_cmp_core(
  char *p_a, char *p_b, char *p_c, char *p_d,
  char *p_res0, char *p_res1,
  FUSED_CMP_PARAM *cmp_param);

#ifdef __cplusplus
extern "C" {
#endif

void atomic_fused_cmp(
    int nodechip_idx,
    P_COMMAND p_command);

#ifdef __cplusplus
}
#endif

#endif /* ATOMIC_FUSED_CMP_H */
