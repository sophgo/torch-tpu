#pragma once

#include <iostream>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#if defined BACKEND_SG2260
#include <tpuv7_rt.h>
#define C10_TPU_CHECK( EXPR )                                     \
do {                                                              \
    auto status = EXPR;                                \
    TORCH_CHECK ( status == tpuRtSuccess, __FILE__ ,":" ,__func__ ); \
} while (0)
#elif defined BACKEND_1684X
#include <bmlib_runtime.h>
#define C10_TPU_CHECK( EXPR )                                     \
do {                                                              \
    auto status = EXPR;                                \
    TORCH_CHECK ( status == SG_SUCCESS, __FILE__ ,":" ,__func__ ); \
} while (0)
#endif