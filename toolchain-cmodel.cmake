if(SOC_MODE)
  set(COMPILER_PREFIX $ENV{PREBUILT_DIR}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-)
else()
  set(COMPILER_PREFIX $ENV{PREBUILT_DIR}/x86-64-core-i7--glibc--stable/bin/x86_64-linux-)
endif()

set(CMAKE_C_COMPILER ${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${COMPILER_PREFIX}g++)


