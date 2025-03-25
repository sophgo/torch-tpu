cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(LibtorchExample)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

if($ENV{SOC_CROSS} STREQUAL "ON")
    set(CMAKE_CXX_COMPILER $ENV{ARM_COMPILE}/bin/aarch64-none-linux-gnu-gcc)
endif()

set(Torch_DIR "$ENV{PYTORCH_INSTALL_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(/opt/sophon/libsophon-current/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories($ENV{TPUTRAIN_INSTALL_TOP}/include)
include_directories($ENV{TPUTRAIN_INSTALL_TOP}/include/torch_tpu)
include_directories($ENV{TPUTRAIN_INSTALL_TOP}/include/torch_tpu/csrc)
include_directories($ENV{TPUTRAIN_INSTALL_TOP}/include/torch_tpu/csrc/core)
link_directories($ENV{TPUTRAIN_INSTALL_TOP}/lib)
link_directories($ENV{TPUTRAIN_INSTALL_TOP}/libtorch)

set(SOURCES
    basic_with_device.cpp
    basic_net.cpp
    basic_net_with_device.cpp
    basic_resnet_with_device.cpp
    basic_vgg_with_device.cpp
    basic_compile_mode.cpp
)

foreach(SOURCE ${SOURCES})
    get_filename_component(TARGET_NAME ${SOURCE} NAME_WE)

    add_executable(${TARGET_NAME} ${SOURCE})

    target_compile_definitions(${TARGET_NAME} PRIVATE BACKEND_1684X NO_PYTHON_API)

    target_link_libraries(${TARGET_NAME}
                            torch
                            torch_cpu
                            libtorch_tpu
                            tpudnn.bm1684x
                            bmlib
                            bmrt
                            stdc++
                            gfortran
                            m)

    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
endforeach()
