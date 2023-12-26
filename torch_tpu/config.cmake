#########################################################
# Torch TPU configuration
#########################################################
option(DEBUG "option for debug"                                           OFF)
option(JIT_TRAIN_ENABLE "option for enable jit train"                     OFF)
option(TPU_OP_TIMING    "option for op timing"                            OFF)
option(SHOW_OP_INFO     "option for op calls infomation, for debug only " OFF)
option(SHOW_CPU_OP      "option for cpu op using "                        OFF)
option(SHOW_MALLOC_INFO "option for memory usage info"                    OFF)

if(DEBUG OR "$ENV{TPUTRAIN_BUILD_TYPE}" STREQUAL "ON")
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-DDEBUG)
else()
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(TPU_OP_TIMING)
  add_definitions(-DTPU_OP_TIMING)
endif()

if(SHOW_OP_INFO)
  add_definitions(-DSHOW_OP_INFO)
endif()

if(SHOW_CPU_OP)
  add_definitions(-DSHOW_CPU_OP)
endif()

if(SHOW_MALLOC_INFO)
  add_definitions(-DSHOW_MALLOC_INFO)
endif()