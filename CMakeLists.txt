cmake_minimum_required(VERSION 2.8)
project(tpu-train)

find_program(CCACHE ccache)
if(CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE})
endif()

if(NOT DEFINED ENV{TRAIN_TOP})
  message(FATAL_ERROR "execute envsetup.sh first")
endif()

set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread -D_GLIBCXX_USE_CXX11_ABI=1 -fno-strict-aliasing ${SAFETY_FLAGS}")
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "Install prefix")
message(STATUS "CMAKE PREFIX: " ${CMAKE_INSTALL_PREFIX})

include(${PROJECT_SOURCE_DIR}/config_common.cmake)

enable_testing()
set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})

add_subdirectory(firmware_core)
add_subdirectory(sgdnn)
