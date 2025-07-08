#pragma once
#include <c10/util/Flags.h>

C10_DECLARE_typed_var(int, caffe2_log_level);
void resetLogLevel();