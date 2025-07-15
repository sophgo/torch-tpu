#pragma once

#include <iostream>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#define C10_TPU_CHECK( EXPR )                                     \
do {                                                              \
    auto status = EXPR;                                \
    TORCH_CHECK ( status == tpuSuccess, __FILE__ ,":" ,__func__ ); \
} while (0)
