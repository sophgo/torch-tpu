cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_TPU CXX)

set(SOC_CROSS_MODE $ENV{SOC_CROSS_MODE})
message(STATUS "SOC_CROSS_MODE: ${SOC_CROSS_MODE}")
if(SOC_CROSS_MODE STREQUAL "ON")
    set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
else()
    message(STATUS "SOC_CROSS_MODE is not ON")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
add_compile_options(-std=c++17 -Wno-attributes -Wall -fPIC)
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")
add_definitions("-DPYBIND11_BUILD_ABI=\"_cxxabi1011\"")

## pytorch
if (DEFINED PYTORCH_INSTALL_DIR)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include)
  link_directories(${PYTORCH_INSTALL_DIR}/lib)
else()
  message(WARNING "USING $ENV{PYTORCH_INSTALL_DIR}")
  include_directories($ENV{PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
  include_directories($ENV{PYTORCH_INSTALL_DIR}/include)
  link_directories($ENV{PYTORCH_INSTALL_DIR}/lib)
endif()

## sg_api_struct
include_directories($ENV{TPUTRAIN_TOP}/common/include)

## sgdnn
include_directories($ENV{TPUTRAIN_TOP}/sgdnn/include)
link_directories($ENV{TPUTRAIN_TOP}/build/sgdnn)

## torch_tpu
include_directories($ENV{TPUTRAIN_TOP})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/core)

if(JIT_TRAIN_ENABLE)
  message(STATUS "JIT_TRAIN_ENABLE")
  file(GLOB_RECURSE SRCS
    csrc/core/TPUGuardImpl.cpp
    csrc/core/TPUAllocator.cpp
    csrc/core/TPUDeviceManager.cpp
    csrc/core/TPUTorchUtils.cpp
    csrc/core/TPUCtypeApi.cpp
    csrc/jit/*.cpp
  )
else()
  file(GLOB_RECURSE SRCS
    csrc/core/*.cpp
    csrc/core/Interface/*.cpp
    csrc/ops/native_ops/*.cpp
    csrc/ops/my_ops/*.cpp
    csrc/ops/autocast_ops/*.cpp
  )
endif()

if (BUILD_LIBTORCH)
  message(STATUS "PYTHON API DISABLE, ONLY C INTERFACE")
  file(GLOB_RECURSE CSRC_SRCS
    csrc/libtorch/*.cpp
  )
  LIST(APPEND SRCS ${CSRC_SRCS})
else()
  include_directories(${PYTHON_INCLUDE_DIR})
  message(STATUS "PYTHON API ENABLE")
  file(GLOB_RECURSE CSRC_SRCS
    csrc/aten/*.cpp
    csrc/aten/*/*.cpp
    csrc/tpu/*.cpp
    csrc/utils/*.cpp
  )
  LIST(APPEND SRCS ${CSRC_SRCS})
endif()

set(concated_src all_in_one.cpp)
add_custom_command(
    OUTPUT ${concated_src}
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/concat_all_torch_plugin_sources.sh ${SRCS}
        > ${CMAKE_CURRENT_BINARY_DIR}/${concated_src}
    DEPENDS ${SRCS}
    VERBATIM)

if(BACKEND_SG2260)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d)
  list(APPEND concated_src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d/ProcessGroupSCCL.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d/SCCLDeviceFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d/ProcessGroupSCCLHost.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d/SCCLHostDeviceFactory.cpp
  )
endif()

find_package(TPU1686 REQUIRED)

add_library(torch_tpu SHARED ${concated_src})
target_link_libraries(torch_tpu TPU1686::tpuDNN c10 torch_cpu)

get_filename_component(tpuDNNPath ${tpuDNN_LIBRARY} DIRECTORY)

if(BACKEND_1684X)
  include_directories(${RUNTIME_INCLUDE_PATH})
  target_link_directories(torch_tpu PRIVATE ${RUNTIME_LIB_PATH})
  target_link_libraries(torch_tpu sgdnn bmlib)
  set_target_properties(torch_tpu PROPERTIES INSTALL_RPATH
      "$ORIGIN;$ORIGIN/../../third_party/bmlib/lib;${tpuDNNPath}")
  install(TARGETS torch_tpu DESTINATION lib)
elseif(BACKEND_SG2260)
  if (EXISTS ${CMAKE_SOURCE_DIR}/../tpuv7-runtime)
    include_directories(${CMAKE_SOURCE_DIR}/../tpuv7-runtime/cdmlib/host/cdm_runtime/include)
    set(tpuv7_rpath "$ORIGIN/../../../tpuv7-runtime/build/cdmlib/host/cdm_runtime")
    target_link_directories(torch_tpu PRIVATE "${CMAKE_SOURCE_DIR}/../tpuv7-runtime/build/cdmlib/host/cdm_runtime")
  else()
    include_directories(${RUNTIME_INCLUDE_PATH})
    target_link_directories(torch_tpu PRIVATE ${RUNTIME_LIB_PATH})
    set(tpuv7_rpath "$ORIGIN/../../third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib")
  endif()
  set_target_properties(torch_tpu PROPERTIES INSTALL_RPATH
      "$ORIGIN;${tpuv7_rpath};${tpuDNNPath};${tpuDNNPath}/sccl")
  target_link_libraries(torch_tpu sgdnn tpuv7_rt sophon TPU1686::sccl)
  if (USING_CMODEL)
    target_link_libraries(torch_tpu cdm_daemon_emulator pthread)
  endif()
  install(TARGETS torch_tpu DESTINATION lib)
endif()

set_target_properties(torch_tpu PROPERTIES SUFFIX ".$ENV{CHIP_ARCH}.so")
add_custom_command(TARGET torch_tpu POST_BUILD
    COMMAND patchelf --set-soname libtorch_tpu.so $<TARGET_FILE:torch_tpu>
    COMMENT "Change SONAME of libtorch_tpu.so"
    DEPENDS torch_tpu VERBATIM)

# Test
#add_executable(test_conv2d test/conv2d.cpp)
#target_link_libraries(test_conv2d torch_tpu)
#add_executable(test_conv2d_fp16 test/conv2d_fp16.cpp)
#target_link_libraries(test_conv2d_fp16 torch_tpu)
#add_executable(test_conv2d_backward test/conv2d_backward.cpp)
#target_link_libraries(test_conv2d_backward torch_tpu)
#add_executable(test_conv2d_backward_fp16 test/conv2d_backward_fp16.cpp)
#target_link_libraries(test_conv2d_backward_fp16 torch_tpu)
#add_executable(test_batchnorm test/batchnorm.cpp)
#target_link_libraries(test_batchnorm torch_tpu)
#add_executable(test_layernorm test/layernorm.cpp)
#target_link_libraries(test_layernorm torch_tpu)
#add_executable(test_batchnorm_fp16 test/batchnorm_fp16.cpp)
#target_link_libraries(test_batchnorm_fp16 torch_tpu)
#add_executable(test_batchnorm_backward test/batchnorm_backward.cpp)
#target_link_libraries(test_batchnorm_backward torch_tpu)
#add_executable(test_layernorm_backward test/layernorm_backward.cpp)
#target_link_libraries(test_layernorm_backward torch_tpu)
#add_executable(test_relu test/relu.cpp)
#target_link_libraries(test_relu torch_tpu)
#add_executable(test_mul test/mul.cpp)
#target_link_libraries(test_mul torch_tpu)
#add_executable(test_bmm test/bmm.cpp)
#target_link_libraries(test_bmm torch_tpu)
#add_executable(test_gelu test/gelu.cpp)
#target_link_libraries(test_gelu torch_tpu)
#add_executable(test_gelu_backward test/gelu_backward.cpp)
#target_link_libraries(test_gelu_backward torch_tpu)
#add_executable(test_softmax test/softmax.cpp)
#target_link_libraries(test_softmax torch_tpu)
#add_executable(test_softmax_backward test/softmax_backward.cpp)
#target_link_libraries(test_softmax_backward torch_tpu)
#add_executable(test_reduce test/reduce_sum.cpp)
#target_link_libraries(test_reduce torch_tpu)

# Benchmark
#add_executable(benchmark_conv2d benchmark/conv2d.cpp)
#target_link_libraries(benchmark_conv2d torch_tpu)
#add_executable(benchmark_conv2d_backward benchmark/conv2d_backward.cpp)
#target_link_libraries(benchmark_conv2d_backward torch_tpu)
#add_executable(benchmark_conv2d_fp16 benchmark/conv2d_fp16.cpp)
#target_link_libraries(benchmark_conv2d_fp16 torch_tpu)
#add_executable(benchmark_conv2d_backward_fp16 benchmark/conv2d_backward_fp16.cpp)
#target_link_libraries(benchmark_conv2d_backward_fp16 torch_tpu)
#add_executable(benchmark_mm benchmark/mm.cpp)
#target_link_libraries(benchmark_mm torch_tpu)
#add_executable(benchmark_bmm benchmark/bmm.cpp)
#target_link_libraries(benchmark_bmm torch_tpu)

