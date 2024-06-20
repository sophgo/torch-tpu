#########################################################
# General configuration
#########################################################
option(USING_PERF_MODE       "Using performance mode"                          OFF)
option(BACKEND_1684X         "Opt for 1684x"                                   OFF)
option(BACKEND_SG2260        "Opt for 2260"                                    OFF)

option(USING_CMODEL          "option for using cmodel"                         OFF)

option(USING_PERF_MODE       "Using performance mode"                          OFF)

option(USE_TPUv7_RUNTIME     "option for control use new runtime"              OFF)

option(DUMP_INS              "option for control dump ins, cmodel only"        OFF)
######################################
### CHIP BACKEND
######################################
if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
  set(BACKEND_1684X         1)
  set(BACKEND_SG2260        0)
  add_definitions(-DBACKEND_1684X)
elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
  set(BACKEND_1684X         0)
  set(BACKEND_SG2260        1)
  add_definitions(-DBACKEND_SG2260)
else()
  message(FATAL_ERROR "unsupport CHIP backend")
endif()

######################################
## HEADER/LIB PATH
######################################
set(TPUv7_RUNTIME_PATH    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0)
set(LIBSOPHON_PATH        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/bmlib)

######################################
###             RUNTIME
######################################
if (BACKEND_1684X)
  set(RUNTIME_INCLUDE_PATH ${LIBSOPHON_PATH}/include)
  set(RUNTIME_LIB_PATH ${LIBSOPHON_PATH}/lib)
elseif(BACKEND_SG2260)
  set(RUNTIME_INCLUDE_PATH ${TPUv7_RUNTIME_PATH}/include)
  set(RUNTIME_LIB_PATH ${TPUv7_RUNTIME_PATH}/lib)
endif()

message(STATUS "RUNTIME_INCLUDE_PATH : ${RUNTIME_INCLUDE_PATH}")
message(STATUS "RUNTIME_LIB_PATH : ${RUNTIME_LIB_PATH}")
message(STATUS "BACKEND_CHIP : $ENV{CHIP_ARCH}")


if(USING_PERF_MODE)
  add_definitions(-DUSING_PERF_MODE)
endif()


if(DUMP_INS)
  add_definitions(-DDUMP_INS)
endif()

set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS} -Wno-invalid-offsetof")
