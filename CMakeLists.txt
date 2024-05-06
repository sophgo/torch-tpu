cmake_minimum_required(VERSION 2.8)
project(tpu-train)

if(NOT DEFINED ENV{TPUTRAIN_TOP})
  message(FATAL_ERROR "execute envsetup.sh first")
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "Install prefix")
message(STATUS "CMAKE PREFIX: " ${CMAKE_INSTALL_PREFIX})

# ftp package
string(TIMESTAMP BUILD_TIME "%Y%m%d")
set(TORCHTPU_VERSION "${BUILD_TIME}" CACHE STRING "torch-tpu version" FORCE)
message(STATUS "torch-tpu version: ${TORCHTPU_VERSION}")
add_definitions(-DTORCHTPU_VERSION="${TORCHTPU_VERSION}")

include(${PROJECT_SOURCE_DIR}/config_common.cmake)

set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})

add_subdirectory(firmware_core)
add_subdirectory(sgdnn)
add_dependencies(sgdnn firmware_core)
add_subdirectory(torch_tpu)