project(FIRMWARE C)

set(FIRMWARE_BACKEND_1684X 0)
set(FIRMWARE_BACKEND_2260 0)
if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
  set(FIRMWARE_BACKEND_1684X 1)
  set(FIRMWARE_BACKEND_2260 0)
elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
  set(FIRMWARE_BACKEND_1684X 0)
  set(FIRMWARE_BACKEND_2260 1)
endif()
configure_file( 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h" )
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

if (USING_CMODEL)
    include(cmodel.cmake)
elseif (PCIE_MODE)
    include(pcie.cmake)
elseif (SOC_MODE)
    include(soc.cmake)
endif()
