#########################################################
# General configuration
#########################################################
option(USING_CMODEL "option for using cmodel" OFF)
option(PCIE_MODE "option for pcie mode" ON)
option(SOC_MODE "run on soc platform" OFF)
option(USING_PERF_MODE "Using performance mode" OFF)

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

if(USING_PERF_MODE)
  add_definitions(-DUSING_PERF_MODE)
endif()
