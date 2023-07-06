project(SGDNN CXX)

set(SGDNN_BACKEND_1684X 0)
set(SGDNN_BACKEND_2260 0)
if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
  set(SGDNN_BACKEND_1684X 1)
  set(SGDNN_BACKEND_2260 0)
elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
  set(SGDNN_BACKEND_1684X 0)
  set(SGDNN_BACKEND_2260 1)
endif()

configure_file( 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config_sgdnn_backend.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config_sgdnn_backend.h" )

if(ENABLE_PYBIND)
  add_subdirectory(${PROJECT_ROOT}/third_party/pybind11 ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(SOC_MODE)
  set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
  set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_ROOT}/common/include)
include_directories(${PROJECT_ROOT}/third_party/include)
include_directories(/opt/sophon/libsophon-current/include)
include_directories(${CMAKE_BINARY_DIR}/firmware_core)

if(USING_CMODEL)
  link_directories(${PROJECT_ROOT}/third_party/$ENV{CHIP_ARCH})
endif()

link_directories(/opt/sophon/libsophon-current/lib)

add_library(sgdnn SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn.cpp)

if(USING_CMODEL)
  target_link_libraries(sgdnn bmodel bmlib_cmodel)
else()
  target_link_libraries(sgdnn bmodel bmlib)
endif()

if (ENABLE_PYBIND)
  pybind11_add_module(sgdnn_pybind ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn_pybind.cpp)
  if(USING_CMODEL)
    target_link_libraries(sgdnn_pybind sgdnn bmodel bmlib_cmodel)
  else()
    target_link_libraries(sgdnn_pybind sgdnn bmodel bmlib)
  endif()
endif()
