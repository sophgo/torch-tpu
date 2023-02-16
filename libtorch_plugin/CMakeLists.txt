project(LIBTORCH_PLUGIN CXX)

include($ENV{TRAIN_TOP}/toolchain-cmodel.cmake)

add_compile_options(-std=c++14)
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")

set(CMAKE_BUILD_TYPE "Debug")
add_definitions(-DDEBUG)

include_directories(/opt/sophon/libsophon-current/include)
include_directories($ENV{TRAIN_TOP}/libtorch/include/torch/csrc/api/include)
include_directories($ENV{TRAIN_TOP}/libtorch/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tpu)
include_directories($ENV{TRAIN_TOP}/sgdnn/include)
include_directories($ENV{TRAIN_TOP}/common/include)

link_directories($ENV{TRAIN_TOP}/libtorch/lib)
link_directories(/opt/sophon/libsophon-current/lib)
link_directories($ENV{TRAIN_TOP}/third_party)
link_directories($ENV{TRAIN_TOP}/build/sgdnn)

file(GLOB_RECURSE SRCS
  tpu/TPUGuardImpl.cpp
  tpu/TPUAllocator.cpp
  tpu/TPUDeviceManager.cpp
  tpu/TPUModule.cpp
  ops/*.cpp
)

add_library(libtorch_plugin SHARED ${SRCS})
target_link_libraries(libtorch_plugin c10 torch_cpu)
target_link_libraries(libtorch_plugin sgdnn)

#if(USING_CMODEL)
#  target_link_libraries(libtorch_plugin bmlib_cmodel)
#else()
  target_link_libraries(libtorch_plugin bmlib)
#endif()

add_executable(test test.cpp)

target_link_libraries(test libtorch_plugin)

# Test
add_executable(test_conv2d test/conv2d.cpp)
target_link_libraries(test_conv2d libtorch_plugin)
add_executable(test_conv2d_backward test/conv2d_backward.cpp)
target_link_libraries(test_conv2d_backward libtorch_plugin)

# Benchmar
add_executable(benchmark_conv2d benchmark/conv2d.cpp)
target_link_libraries(benchmark_conv2d libtorch_plugin)
