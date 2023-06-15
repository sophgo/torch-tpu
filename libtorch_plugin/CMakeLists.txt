project(LIBTORCH_PLUGIN CXX)

add_compile_options(-std=c++14 -Wno-attributes)
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")

set(CMAKE_BUILD_TYPE "Debug")
add_definitions(-DDEBUG)

set(TPU_OP_TIMING 1)
#set(SHOW_OP_INFO 1)

configure_file( 
    "${PROJECT_SOURCE_DIR}/common/config.h.in"
    "${PROJECT_SOURCE_DIR}/common/config.h" )

include_directories(/opt/sophon/libsophon-current/include)
include_directories($ENV{TPUTRAIN_TOP}/libtorch/include/torch/csrc/api/include)
include_directories($ENV{TPUTRAIN_TOP}/libtorch/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tpu)
include_directories($ENV{TPUTRAIN_TOP}/sgdnn/include)
include_directories($ENV{TPUTRAIN_TOP}/common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories($ENV{TPUTRAIN_TOP}/libtorch/lib)
link_directories($ENV{TPUTRAIN_TOP}/build/sgdnn)

file(GLOB_RECURSE SRCS
  tpu/TPUGuardImpl.cpp
  tpu/TPUAllocator.cpp
  tpu/TPUDeviceManager.cpp
  tpu/TPUTorchUtils.cpp
  tpu/TPUCtypeApi.cpp
  ops/*.cpp
)

add_library(libtorch_plugin SHARED ${SRCS})
target_link_libraries(libtorch_plugin c10 torch_cpu)
target_link_libraries(libtorch_plugin sgdnn)

# Test
#add_executable(test_conv2d test/conv2d.cpp)
#target_link_libraries(test_conv2d libtorch_plugin)
#add_executable(test_conv2d_fp16 test/conv2d_fp16.cpp)
#target_link_libraries(test_conv2d_fp16 libtorch_plugin)
#add_executable(test_conv2d_backward test/conv2d_backward.cpp)
#target_link_libraries(test_conv2d_backward libtorch_plugin)
#add_executable(test_conv2d_backward_fp16 test/conv2d_backward_fp16.cpp)
#target_link_libraries(test_conv2d_backward_fp16 libtorch_plugin)
#add_executable(test_batchnorm test/batchnorm.cpp)
#target_link_libraries(test_batchnorm libtorch_plugin)
#add_executable(test_layernorm test/layernorm.cpp)
#target_link_libraries(test_layernorm libtorch_plugin)
#add_executable(test_batchnorm_fp16 test/batchnorm_fp16.cpp)
#target_link_libraries(test_batchnorm_fp16 libtorch_plugin)
#add_executable(test_batchnorm_backward test/batchnorm_backward.cpp)
#target_link_libraries(test_batchnorm_backward libtorch_plugin)
#add_executable(test_layernorm_backward test/layernorm_backward.cpp)
#target_link_libraries(test_layernorm_backward libtorch_plugin)
#add_executable(test_relu test/relu.cpp)
#target_link_libraries(test_relu libtorch_plugin)
#add_executable(test_mul test/mul.cpp)
#target_link_libraries(test_mul libtorch_plugin)
#add_executable(test_bmm test/bmm.cpp)
#target_link_libraries(test_bmm libtorch_plugin)
#add_executable(test_gelu test/gelu.cpp)
#target_link_libraries(test_gelu libtorch_plugin)
#add_executable(test_gelu_backward test/gelu_backward.cpp)
#target_link_libraries(test_gelu_backward libtorch_plugin)
#add_executable(test_softmax test/softmax.cpp)
#target_link_libraries(test_softmax libtorch_plugin)
#add_executable(test_softmax_backward test/softmax_backward.cpp)
#target_link_libraries(test_softmax_backward libtorch_plugin)
#add_executable(test_reduce test/reduce_sum.cpp)
#target_link_libraries(test_reduce libtorch_plugin)

# Benchmark
#add_executable(benchmark_conv2d benchmark/conv2d.cpp)
#target_link_libraries(benchmark_conv2d libtorch_plugin)
#add_executable(benchmark_conv2d_backward benchmark/conv2d_backward.cpp)
#target_link_libraries(benchmark_conv2d_backward libtorch_plugin)
#add_executable(benchmark_conv2d_fp16 benchmark/conv2d_fp16.cpp)
#target_link_libraries(benchmark_conv2d_fp16 libtorch_plugin)
#add_executable(benchmark_conv2d_backward_fp16 benchmark/conv2d_backward_fp16.cpp)
#target_link_libraries(benchmark_conv2d_backward_fp16 libtorch_plugin)
#add_executable(benchmark_mm benchmark/mm.cpp)
#target_link_libraries(benchmark_mm libtorch_plugin)
#add_executable(benchmark_bmm benchmark/bmm.cpp)
#target_link_libraries(benchmark_bmm libtorch_plugin)

