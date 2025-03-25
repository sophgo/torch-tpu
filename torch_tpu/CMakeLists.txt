cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_TPU CXX)

set(SOC_CROSS_MODE $ENV{SOC_CROSS_MODE})
message(STATUS "SOC_CROSS_MODE: ${SOC_CROSS_MODE}")
if(SOC_CROSS_MODE STREQUAL "ON")
    message(STATUS "SOC_CROSS_MODE is ON")
    if (BACKEND_SG2260)
        set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/firmware_core/sg2260-toolchain.cmake)
    else()
        set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
    endif()
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

if(SOC_CROSS_MODE STREQUAL "ON")
include_directories($ENV{CROSS_TOOLCHAINS}/riscv64-linux-x86_64/python3.10)
endif()
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
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d/ProcessGroupSCCLHost.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d/SCCLHostDeviceFactory.cpp
  )
  get_filename_component(scclPath ${SCCL_LIBRARY} DIRECTORY)
endif()

find_package(TPU1686 REQUIRED)

add_library(torch_tpu SHARED ${concated_src})
target_link_libraries(torch_tpu TPU1686::tpuDNN c10 torch_cpu)

get_filename_component(tpuDNNPath ${tpuDNN_LIBRARY} DIRECTORY)
# print all include files path
get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS ">>>> Include directories: ${dirs}")

if(SOC_CROSS_MODE STREQUAL "ON")
  SET( BMRUNTIME ${CMAKE_SOURCE_DIR}/third_party/bmruntime/lib/bm1684x/arm)
else()
  SET( BMRUNTIME ${CMAKE_SOURCE_DIR}/third_party/bmruntime/lib/bm1684x)
endif()

if(BACKEND_1684X)
  include_directories(${RUNTIME_INCLUDE_PATH})
  include_directories(${CMAKE_SOURCE_DIR}/third_party/bmruntime/include)
  target_link_directories(torch_tpu PRIVATE ${RUNTIME_LIB_PATH} ${BMRUNTIME})
  target_link_libraries(torch_tpu sgdnn bmlib bmrt)
  if(SOC_CROSS_MODE STREQUAL "ON")
    set_target_properties(torch_tpu PROPERTIES INSTALL_RPATH
        "$ORIGIN;$ORIGIN/../../third_party/bmlib/lib/arm;$ORIGIN/../../third_party/bmruntime/lib/bm1684x/arm;${tpuDNNPath}")
  else()
    set_target_properties(torch_tpu PROPERTIES INSTALL_RPATH
        "$ORIGIN;$ORIGIN/../../third_party/bmlib/lib;$ORIGIN/../../third_party/bmruntime/lib/bm1684x;${tpuDNNPath}")
  endif()
  install(TARGETS torch_tpu DESTINATION lib)
elseif(BACKEND_SG2260)

  if (EXISTS ${CMAKE_SOURCE_DIR}/../tpuv7-runtime)
    include_directories(${CMAKE_SOURCE_DIR}/../tpuv7-runtime/cdmlib/host/cdm_runtime/include
                        ${CMAKE_SOURCE_DIR}/../tpuv7-runtime/model-runtime/runtime/include)
    set(tpuv7_rpath "/opt/tpuv7/tpuv7-current/lib:$ORIGIN/../../../tpuv7-runtime/build/cdmlib/host/cdm_runtime:$ORIGIN/../../../tpuv7-runtime/build/model-runtime/runtime:$ORIGIN/../../../tpuv7-runtime/build/model-runtime/bmodel")
    target_link_directories(torch_tpu PRIVATE "${CMAKE_SOURCE_DIR}/../tpuv7-runtime/build/cdmlib/host/cdm_runtime"
                                              "${CMAKE_SOURCE_DIR}/../tpuv7-runtime/build/cdmlib/fw/ap/daemon"
                                              "${CMAKE_SOURCE_DIR}/../tpuv7-runtime/build/model-runtime/runtime"
                                              "${CMAKE_SOURCE_DIR}/../tpuv7-runtime/build/model-runtime/bmodel" )
  else()
  # Use tpu-train current repo third-party library path
    message("Use tpu-train current repo third-party library path")
    include_directories(${RUNTIME_INCLUDE_PATH})
    target_link_directories(torch_tpu PRIVATE ${RUNTIME_LIB_PATH})
    set(tpuv7_rpath "/opt/tpuv7/tpuv7-current/lib:$ORIGIN/../../third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib")
    # set(tpuv7_rpath "${TPUV7_DRIVER_CURRENT}/lib:$ORIGIN/../../third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib")
  endif()
  set_target_properties(torch_tpu PROPERTIES INSTALL_RPATH
      "$ORIGIN;${tpuv7_rpath};${tpuDNNPath};${scclPath}")
  target_link_libraries(torch_tpu sgdnn tpuv7_rt tpuv7_modelrt sophon TPU1686::sccl)
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
