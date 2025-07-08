#include "TPULog.h"

int getTPU_CPP_LOG_LEVEL()
{
  const char *level = getenv("TORCH_TPU_CPP_LOG_LEVEL");
  if (!level) return 2;
  return atoi(level);
}

void resetLogLevel() {
    // 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL
    int level = getTPU_CPP_LOG_LEVEL();
    FLAGS_caffe2_log_level = level;
}