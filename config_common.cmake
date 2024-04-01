#########################################################
# General configuration
#########################################################
option(BACKEND_1684X   "Opt for 1684x"                                   OFF)
option(BACKEND_SG2260    "Opt for 2260"                                    OFF)

option(USING_CMODEL          "option for using cmodel"                         OFF)
option(PCIE_MODE             "option for pcie mode"                            OFF)
option(SOC_MODE              "run on soc platform"                             OFF)

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
## MODE CHOICE
######################################
if ($ENV{MODE_PATTERN} STREQUAL "local")
  set(USING_CMODEL        1)
  set(PCIE_MODE           0)
  set(SOC_MODE            0)
elseif($ENV{MODE_PATTERN} STREQUAL "stable")
  set(USING_CMODEL        0)
  set(PCIE_MODE           1)
  set(SOC_MODE            0)
endif()

if(USING_CMODEL)
  set(TPUv7_RUNTIME_PATH    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0)
  set(LIBSOPHON_PATH        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/bmlib)
  add_definitions(-DUSING_CMODEL)
  message(STATUS "MODE : CMODEL" )
elseif(PCIE_MODE)
  set(TPUv7_RUNTIME_PATH    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tpuv7_runtime/tpuv7_0.1.0)
  set(LIBSOPHON_PATH        /opt/sophon/libsophon-current)
  add_definitions(-DPCIE_MODE)
  message(STATUS "MODE : PCIE" )
elseif(SOC_MODE)
  message(FATAL_ERROR "NO CHEK NOW")
  add_definitions(-DSOC_MODE)
  message(STATUS "MODE : SOC" )
endif()

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