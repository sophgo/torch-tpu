#########################################################
# Torch TPU configuration
#########################################################
option(DEBUG "option for debug"                                           OFF)
option(TPU_OP_TIMING    "option for op timing"                            OFF)
option(SHOW_OP_INFO     "option for op calls infomation, for debug only " OFF)
option(SHOW_MALLOC_INFO "option for memory usage info"                    OFF)
option(BUILD_LIBTORCH   "option for control use only build libtorch"      OFF)
option(SHOW_EACH_OP_TIME     "option for show op time"                    OFF)
option(HOSTCCL          "option for compile Hostsccl"                     OFF)

if (DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0")
    add_definitions(-DDEBUG)
endif()

if(TPU_OP_TIMING)
  add_definitions(-DTPU_OP_TIMING)
endif()

if(SHOW_OP_INFO)
  add_definitions(-DSHOW_OP_INFO)
endif()

if(SHOW_MALLOC_INFO)
  add_definitions(-DSHOW_MALLOC_INFO)
endif()

if(SHOW_EACH_OP_TIME)
  add_definitions(-DSHOW_EACH_OP_TIME)
endif()

if(HOSTCCL)
  add_definitions(-DHOSTCCL)
endif()

######################################
###             LIBTORCH
######################################
if(BUILD_LIBTORCH)
  add_definitions(-DBUILD_LIBTORCH)
endif()
