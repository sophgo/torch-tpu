macro(sg1684x_variable)
    message(STATUS "sg1684x_variable ${ARGV0}=${ARGV1}")
    add_definitions(-DCONFIG_${ARGV0}=${ARGV1})
endmacro()

## Version number
sg1684x_variable(MAJOR_VERSION 1)
sg1684x_variable(MINOR_VERSION 0)

#########################################################
# TPU Parameters
#########################################################
## The local memory of BM1684X is 256KB.
## But we also need to evalute 128KB lmem.
option(USING_128KB_LMEM "use 128KB Local memory per lane" OFF)

## Initialize parameters
sg1684x_variable(NPU_SHIFT 6)
sg1684x_variable(EU_SHIFT 4)
sg1684x_variable(LOCAL_MEM_BANKS 16)
if(USING_128KB_LMEM)
  sg1684x_variable(LOCAL_MEM_ADDRWIDTH 17)   # 128KB
else()
  sg1684x_variable(LOCAL_MEM_ADDRWIDTH 18)   # 256KB
endif()
sg1684x_variable(L2_SRAM_SIZE 0x1FB000)    # 2MB-20KB
sg1684x_variable(STATIC_MEM_SIZE 0x10000)  # 64KB
sg1684x_variable(GLOBAL_DATA_INITIAL 0xdeadbeef)
sg1684x_variable(LOCAL_DATA_INITIAL 0xdeadbeef)
sg1684x_variable(GLOBAL_MEM_SIZE 0x100000000)

#########################################################
# General configuration
#########################################################
option(DEBUG "option for debug" ON)
option(USING_CMODEL "option for using cmodel" ON)
option(SOC_MODE "run on soc platform" OFF)

if(DEBUG)
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-DDEBUG)
else()
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(USING_CMODEL)
  add_definitions(-DUSING_CMODEL)
endif()

if(USING_CMODEL AND DEBUG)
  add_definitions(-rdynamic)
endif()

if(NOT USING_CMODEL)
  add_definitions(-Wno-address-of-packed-member)
endif()

add_definitions(-fno-builtin-memcpy)
add_definitions(-fno-builtin-memset)
add_definitions(-fno-builtin-memmove)

if(USING_PLD_TEST)
  add_definitions(-DUSING_PLD_TEST)
elseif(USING_BRINGUP_TEST)
  add_definitions(-DUSING_BRINGUP_TEST)
endif()
add_definitions(-fsigned-char -Wsign-compare -Wunused-variable -Werror)

if(NOT DEBUG)
  add_definitions(-O3)
endif()
