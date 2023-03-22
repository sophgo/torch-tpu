cmake_minimum_required(VERSION 2.8)
project(tpu-train)

if(NOT DEFINED ENV{TPUTRAIN_TOP})
  message(FATAL_ERROR "execute envsetup.sh first")
endif()

set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread -D_GLIBCXX_USE_CXX11_ABI=1 -fno-strict-aliasing ${SAFETY_FLAGS}")
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "Install prefix")
message(STATUS "CMAKE PREFIX: " ${CMAKE_INSTALL_PREFIX})

include(${PROJECT_SOURCE_DIR}/config_common.cmake)

set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})

add_subdirectory(firmware_core)
add_subdirectory(sgdnn)
add_dependencies(sgdnn firmware_core)
