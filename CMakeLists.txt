cmake_minimum_required(VERSION 2.8)
project(tpu-train)

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
        CACHE PATH "default install path" FORCE)
endif()

if(NOT DEFINED ENV{TPUTRAIN_TOP})
  message(FATAL_ERROR "execute envsetup.sh first")
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "Install prefix")
message(STATUS "CMAKE PREFIX: " ${CMAKE_INSTALL_PREFIX})

message(STATUS "Building torch-tpu for chip $ENV{CHIP_ARCH}")

# ftp package
string(TIMESTAMP BUILD_TIME "%Y%m%d")
set(TORCHTPU_VERSION "${BUILD_TIME}" CACHE STRING "torch-tpu version" FORCE)
message(STATUS "torch-tpu version: ${TORCHTPU_VERSION}")
add_definitions(-DTORCHTPU_VERSION="${TORCHTPU_VERSION}")

include(${PROJECT_SOURCE_DIR}/config_common.cmake)
set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS} -Wno-invalid-offsetof")

set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})

# add_subdirectory(third_party/gloo_sophon)
add_subdirectory(torch_tpu)
