#########################################################
# General configuration
#########################################################
option(USING_PERF_MODE       "Using performance mode"                          OFF)
option(BACKEND_1684X         "Opt for 1684x"                                   OFF)
option(BACKEND_SG2260        "Opt for 2260"                                    OFF)

option(USING_CMODEL          "option for using cmodel"                         OFF)

option(DUMP_INS              "option for control dump ins, cmodel only"        OFF)
######################################
### CHIP BACKEND
######################################
if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
  set(BACKEND_1684X         1)
  set(BACKEND_SG2260        0)
  add_definitions(-DBACKEND_1684X)
  if($ENV{SOC_CROSS_MODE})
    set(PLATFORM "aarch64" CACHE STRING "The architecture of the target machine")
    set(MCPU "crotex-a53" CACHE STRING "CPU type of the target machine")
  endif()
elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
  set(BACKEND_1684X         0)
  set(BACKEND_SG2260        1)
  add_definitions(-DBACKEND_SG2260)
  if($ENV{SOC_CROSS_MODE})
    set(PLATFORM "riscv64" CACHE STRING "The architecture of the target machine")
    set(MCPU "c920" CACHE STRING "CPU type of the target machine")
  endif()
else()
  message(FATAL_ERROR "unsupport CHIP backend")
endif()

######################################
## HEADER/LIB PATH
######################################
### tpuv7 repo path
set(TPUv7_RUNTIME_REPO_PATH          ${CMAKE_SOURCE_DIR}/../tpuv7-runtime)
set(TPUv7_REPO_RT_INCLUDE_PATH       ${TPUv7_RUNTIME_REPO_PATH}/cdmlib/host/cdm_runtime/include)
set(TPUv7_REPO_RT_LIB_PATH           ${TPUv7_RUNTIME_REPO_PATH}/build/cdmlib/host/cdm_runtime)
set(TPUv7_REPO_RT_AP_LIB_PATH        ${TPUv7_RUNTIME_REPO_PATH}/build/cdmlib/fw/ap/daemon)
set(TPUv7_REPO_MODEL_RT_INCLUDE_PATH ${TPUv7_RUNTIME_REPO_PATH}/model-runtime/runtime/include)
set(TPUv7_REPO_MODEL_RT_LIB_PATH     ${TPUv7_RUNTIME_REPO_PATH}/build/model-runtime/runtime)
set(TPUv7_REPO_MODEL_BMODEL_LIB_PATH ${TPUv7_RUNTIME_REPO_PATH}/build/model-runtime/bmodel)

### runtime/modelruntime in third-party
set(TPUv7_RUNTIME_PATH            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0)
set(LIBSOPHON_PATH                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/bmlib)
set(BMRUNTIME_PATH                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/bmruntime)
######################################
###             RUNTIME
######################################
if (BACKEND_1684X)
  set(RUNTIME_INCLUDE_PATH        ${LIBSOPHON_PATH}/include)
  set(MODELRUNTIME_INCLUDE_PATH      ${BMRUNTIME_PATH}/include)
  if($ENV{SOC_CROSS_MODE} STREQUAL "ON")
    set(RUNTIME_LIB_PATH          ${LIBSOPHON_PATH}/lib/arm)
    set(MODELRUNTIME_LIB_PATH     ${BMRUNTIME_PATH}/lib/bm1684x/arm)
  else()
    set(RUNTIME_LIB_PATH          ${LIBSOPHON_PATH}/lib)
    set(MODELRUNTIME_LIB_PATH     ${BMRUNTIME_PATH}/lib/bm1684x)
  endif()
elseif(BACKEND_SG2260)
  set(RUNTIME_INCLUDE_PATH         ${TPUv7_RUNTIME_PATH}/include)
  set(RUNTIME_LIB_PATH             ${TPUv7_RUNTIME_PATH}/lib)
  set(MODELRUNTIME_LIB_PATH        ${TPUv7_RUNTIME_PATH}/lib)
endif()

message(STATUS "BACKEND_CHIP :         $ENV{CHIP_ARCH}")
message(STATUS "RUNTIME_INCLUDE_PATH : ${RUNTIME_INCLUDE_PATH}")
message(STATUS "RUNTIME_LIB_PATH :     ${RUNTIME_LIB_PATH}")

if(USING_PERF_MODE)
  add_definitions(-DUSING_PERF_MODE)
endif()

if(DUMP_INS)
  add_definitions(-DDUMP_INS)
endif()


