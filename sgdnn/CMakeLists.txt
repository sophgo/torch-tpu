project(SGDNN CXX)

include(config.cmake)
set(SOC_CROSS_MODE $ENV{SOC_CROSS_MODE})
if(SOC_CROSS_MODE STREQUAL "ON")
    set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
else()
    message(STATUS "SOC_CROSS_MODE is not ON")
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_ROOT}/common/include)
include_directories(${PROJECT_SOURCE_DIR}/../third_party/firmware/include)

## runtime
include_directories(${RUNTIME_INCLUDE_PATH})
link_directories(${RUNTIME_LIB_PATH})

## kernel_module
include_directories(${CMAKE_BINARY_DIR}/firmware_core)

if(BACKEND_1684X)
  add_library(sgdnn SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn.cpp)
  target_link_libraries(sgdnn bmlib)
elseif(BACKEND_SG2260)
  add_library(sgdnn SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/tpukernel_launcher.cpp)
  target_link_libraries(sgdnn tpuv7_rt)
endif()
