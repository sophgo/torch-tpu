project(FIRMWARE C)

include(config.cmake)

if(USING_CMODEL AND (DEBUG OR "$ENV{TPUTRAIN_BUILD_TYPE}" STREQUAL "ON"))
  add_definitions(-rdynamic)
endif()

set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS}")

add_definitions(-Wno-address-of-packed-member)
add_definitions(-fno-builtin-memcpy)
add_definitions(-fno-builtin-memset)
add_definitions(-fno-builtin-memmove)
add_definitions(-fsigned-char -Wsign-compare -Wunused-variable -Werror -Wno-dev)

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
