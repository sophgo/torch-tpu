if(EXISTS $ENV{PREBUILT_DIR})

if (($ENV{CHIP_ARCH} STREQUAL "bm1684x"))
  if(SOC_MODE)
    set(COMPILER_PREFIX $ENV{PREBUILT_DIR}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-)
  else()
    set(COMPILER_PREFIX $ENV{PREBUILT_DIR}/x86-64-core-i7--glibc--stable/bin/x86_64-linux-)
  endif()
endif()

if (($ENV{CHIP_ARCH} STREQUAL "bm1686"))
  if(SOC_MODE)
    set(COMPILER_PREFIX $ENV{PREBUILT_DIR}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-)
  endif()
endif()

if (($ENV{CHIP_ARCH} STREQUAL "bm1684x"))
  list(APPEND link_dirs $ENV{PREBUILT_DIR}/x86-64-core-i7--glibc--stable/x86_64-buildroot-linux-gnu/lib64)
  link_directories(${link_dirs})
endif()

endif()

set(CMAKE_C_COMPILER ${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${COMPILER_PREFIX}g++)

if($ENV{THIRD_PARTY_LIBS})
  set(link_dirs string(REPLACE ":" ";" $ENV{THIRD_PARTY_LIBS}))
endif()

#set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -Wno-format-truncation -Wno-stringop-truncation -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -D_GLIBCXX_USE_CXX11_ABI=1 -fno-strict-aliasing ${SAFETY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fno-strict-aliasing ${SAFETY_FLAGS}")
