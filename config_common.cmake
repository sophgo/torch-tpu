#########################################################
# General configuration
#########################################################
option(DEBUG "option for debug" ON)
option(USING_CMODEL "option for using cmodel" ON)
option(PCIE_MODE "option for pcie mode" OFF)
option(SOC_MODE "run on soc platform" OFF)
option(ENABLE_PYBIND "enable sgdnn pybind" OFF)

if(DEBUG)
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-DDEBUG)
else()
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(USING_CMODEL)
  if ($ENV{LIBSOPHON_PATTERN} MATCHES $ENV{LIBSOPHON_STABLE})
    message(FATAL_ERROR "RUNING Cmodel Mode！！！ Can't Use Release Libsophon, Please Use local or Latest Version")
  elseif ($ENV{LIBSOPHON_PATTERN} MATCHES $ENV{LIBSOPHON_LATEST})
    message(WARNING "RUNING Cmodel Mode，Make sure Latest libsophon built in CModel mode")
  endif()
  add_definitions(-DUSING_CMODEL)
endif()

if(PCIE_MODE)
  if ($ENV{LIBSOPHON_PATTERN} MATCHES $ENV{LIBSOPHON_LOCAL})
    message(FATAL_ERROR "RUNING PCIE Mode！！！ Can't Use local cmodel version bmlib, Please Use Release or Latest Version")
  elseif ($ENV{LIBSOPHON_PATTERN} MATCHES $ENV{LIBSOPHON_LATEST})
    message(WARNING "RUNING PCIE Mode，Make sure Latest libsophon built in PCIE mode")
  endif()
  add_definitions(-DPCIE_MODE)
endif()

if(SOC_MODE)
  message(FATAL_ERROR "NO CHEK NOW")
  add_definitions(-DSOC_MODE)
endif()

if(USING_CMODEL AND DEBUG)
  add_definitions(-rdynamic)
endif()

add_definitions(-Wno-address-of-packed-member)

add_definitions(-fno-builtin-memcpy)
add_definitions(-fno-builtin-memset)
add_definitions(-fno-builtin-memmove)

add_definitions(-fsigned-char -Wsign-compare -Wunused-variable -Werror -Wno-dev)

if(NOT DEBUG)
  add_definitions(-O3)
endif()

if(ENABLE_PYBIND)
  add_definitions(-DENABLE_PYBIND)
endif()
