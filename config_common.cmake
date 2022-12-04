#########################################################
# General configuration
#########################################################
option(DEBUG "option for debug" ON)
option(USING_CMODEL "option for using cmodel" ON)
option(PCIE_MODE "option for pcie mode" OFF)
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

if(PCIE_MODE)
  add_definitions(-DPCIE_MODE)
endif()

if(SOC_MODE)
  add_definitions(-DSOC_MODE)
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

add_definitions(-fsigned-char -Wsign-compare -Wunused-variable -Werror -Wno-dev)

if(NOT DEBUG)
  add_definitions(-O3)
endif()
