macro(sg2260_variable)
    set(${ARGV0} ${ARGV1} CACHE STRING "")
    message(STATUS "sg2260_variable ${ARGV0}=${${ARGV0}}")
    add_definitions(-DCONFIG_${ARGV0}=${${ARGV0}})
endmacro()

## Version number
sg2260_variable(MAJOR_VERSION 1)
sg2260_variable(MINOR_VERSION 0)

add_definitions(-DCMODEL_CHIPID=0x2260)

#########################################################
# TPU Parameters
#########################################################
## Initialize parameters
if(NPU_SHIFT)
  sg2260_variable(NPU_SHIFT ${NPU_SHIFT})
else()
  sg2260_variable(NPU_SHIFT 6)
endif()
sg2260_variable(CORE_ID 0)
sg2260_variable(EU_SHIFT 5)
sg2260_variable(LOCAL_MEM_BANKS 16)
sg2260_variable(LOCAL_MEM_ADDRWIDTH 18)   # 256kB
sg2260_variable(L2_SRAM_SIZE 0x8000000)   # 128MB
sg2260_variable(STATIC_MEM_SIZE 0x10000)  # 64KB
sg2260_variable(GLOBAL_DATA_INITIAL 0xdeadbeef)
sg2260_variable(LOCAL_DATA_INITIAL 0xdeadbeef)
sg2260_variable(SMEM_DATA_INITIAL 0xdeadbeef)
sg2260_variable(GLOBAL_MEM_SIZE 0x100000000)
sg2260_variable(FW_SIZE 0x20000000) # 512M
sg2260_variable(MSG_ID_WIDTH 9)
if(MAX_TPU_CORE_NUM)
  sg2260_variable(MAX_TPU_CORE_NUM ${MAX_TPU_CORE_NUM})
else()
  sg2260_variable(MAX_TPU_CORE_NUM 8)
endif()
sg2260_variable(MAX_CDMA_NUM 10)

include(${PROJECT_ROOT}/options.cmake)

#########################################################
# OpenMP definitions
#########################################################
option(USING_OMP "using OpenMP for parallel calculation" OFF)
option(USING_ONEDNN "using oneDNN to accelerate cmodel" OFF)
option(USING_CUDA "Using CUDA for GPU acceleration" OFF)
option(NO_L2_DUMP "Not dump l2 mem data tv_gen" OFF)
option(NO_GLOBAL_DUMP "Not dump global mem data tv_gen" OFF)
option(NO_LOCAL_DUMP "Not dump local mem data tv_gen" OFF)
option(NO_STATIC_DUMP "Not dump static ram data tv_gen" OFF)
option(USING_FAKE_DDR_MODE "Enable pld sys test mode" OFF)
option(USING_MPI, "using MPI for multi-node cmodel" OFF)

if(USING_CMODEL)
    set(USING_OMP ON)
    set(USING_ONEDNN ON)
else()
    set(USING_OMP OFF)
endif()
if(NO_GLOBAL_DUMP)
    add_definitions(-DNO_GLOBAL_DUMP)
endif()
if(NO_LOCAL_DUMP)
    add_definitions(-DNO_LOCAL_DUMP)
endif()
if(NO_STATIC_DUMP)
    add_definitions(-DNO_STATIC_DUMP)
endif()
if(NO_L2_DUMP)
    add_definitions(-DNO_L2_DUMP)
endif()
if(USING_FAKE_DDR_MODE)
    add_definitions(-DUSING_FAKE_DDR_MODE)
endif()

if(SG_TV_GEN)
  set(USING_OMP OFF)
  set(USING_ONEDNN OFF)
endif()

if(USING_OMP)
  message(STATUS "Using OMP")
  FIND_PACKAGE(OpenMP REQUIRED)
  if (OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
    message(FATAL_ERROR "openmp not found!")
  endif()
endif()

if(USING_MPI)
  message(STATUS, "Using MPI")
  find_package(MPI REQUIRED)
  if (MPI_FOUND)
    include_directories(${MPI_CXX_HEADER_DIR})
    add_definitions(-DUSING_MPI)
  else()
    message(STATUS, "Please install OpenMPI")
    set(USING_MPI OFF)
  endif()
endif()
