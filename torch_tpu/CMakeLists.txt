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

#################### DEPENDS #######################
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

## runtime & model-runtime
include_directories( ${RUNTIME_INCLUDE_PATH} ${MODELRUNTIME_INCLUDE_PATH} )

### tpudnn && sccl
find_package(TPU1686 REQUIRED)
get_filename_component(tpuDNNPath ${tpuDNN_LIBRARY} DIRECTORY)
if($ENV{CHIP_ARCH} STREQUAL "sg2260")
get_filename_component(scclPath   ${SCCL_LIBRARY}   DIRECTORY)
endif()
#################### END DEPENDS ####################

######## torch-tpu inc/src
include_directories($ENV{TPUTRAIN_TOP})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d)

file(GLOB_RECURSE SRCS
  csrc/core/*.cpp
  csrc/core/Interface/*.cpp
  csrc/ops/native_ops/*.cpp
  csrc/ops/my_ops/*.cpp
  csrc/ops/autocast_ops/*.cpp
  csrc/aten/*.cpp
  csrc/aten/*/*.cpp
)

file(GLOB_RECURSE DIST_SRCS
  csrc/distributed/c10d/*.cpp
)

set(concated_src all_in_one.cpp)
add_custom_command(
    OUTPUT ${concated_src}
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/concat_all_torch_plugin_sources.sh ${SRCS}
        > ${CMAKE_CURRENT_BINARY_DIR}/${concated_src}
    DEPENDS ${SRCS}
    VERBATIM)

if(BACKEND_SG2260)
  list(APPEND concated_src ${DIST_SRCS})
endif()

add_library(torch_tpu SHARED ${concated_src})
target_link_libraries(torch_tpu c10 torch_cpu TPU1686::tpuDNN sgdnn)

if(BACKEND_1684X)
  target_link_directories(torch_tpu PRIVATE ${RUNTIME_LIB_PATH} ${MODELRUNTIME_LIB_PATH})
  target_link_libraries(torch_tpu bmlib bmrt)
  set_target_properties(torch_tpu PROPERTIES INSTALL_RPATH
                        "$ORIGIN;${RUNTIME_LIB_PATH};${MODELRUNTIME_LIB_PATH};${tpuDNNPath}")
  install(TARGETS torch_tpu DESTINATION lib)
elseif(BACKEND_SG2260)
  if (EXISTS ${TPUv7_RUNTIME_REPO_PATH})
    include_directories( ${TPUv7_REPO_RT_INCLUDE_PATH} ${TPUv7_REPO_MODEL_RT_INCLUDE_PATH} )
    set(tpuv7_rpath "/opt/tpuv7/tpuv7-current/lib:$ORIGIN/../../../tpuv7-runtime/build/cdmlib/host/cdm_runtime:$ORIGIN/../../../tpuv7-runtime/build/model-runtime/runtime:$ORIGIN/../../../tpuv7-runtime/build/model-runtime/bmodel")
    target_link_directories(torch_tpu PRIVATE ${TPUv7_REPO_RT_LIB_PATH}
                                              ${TPUv7_REPO_RT_AP_LIB_PATH}
                                              ${TPUv7_REPO_MODEL_RT_LIB_PATH}
                                              ${TPUv7_REPO_MODEL_BMODEL_LIB_PATH})
  else()
    # Use tpu-train current repo third-party library path
    message("Use tpu-train current repo third-party library path")
    target_link_directories(torch_tpu PRIVATE ${RUNTIME_LIB_PATH} ${MODELRUNTIME_LIB_PATH})
    set(tpuv7_rpath "/opt/tpuv7/tpuv7-current/lib")
  endif()
  set_target_properties(torch_tpu PROPERTIES INSTALL_RPATH
                      "$ORIGIN;${tpuv7_rpath};${RUNTIME_LIB_PATH};${MODELRUNTIME_LIB_PATH};
                       ${tpuDNNPath};${scclPath}")
  target_link_libraries(torch_tpu tpuv7_rt tpuv7_modelrt sophon TPU1686::sccl)
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

### torch-tpu python api interface 
if(SOC_CROSS_MODE STREQUAL "ON")
  include_directories($ENV{CROSS_TOOLCHAINS}/riscv64-linux-x86_64/python3.10)
else()
  include_directories(${PYTHON_INCLUDE_DIR})
endif()
message(STATUS "PYTHON API ENABLE")
file(GLOB_RECURSE PYC_SRCS
  csrc/tpu/*.cpp
  csrc/utils/*.cpp
)
add_library(torch_tpu_python SHARED ${PYC_SRCS})
target_link_directories(torch_tpu_python PRIVATE ${RUNTIME_LIB_PATH} ${MODELRUNTIME_LIB_PATH})
target_link_libraries(torch_tpu_python torch_tpu)

set_target_properties(torch_tpu_python PROPERTIES INSTALL_RPATH 
      "$ORIGIN;${tpuv7_rpath};${tpuDNNPath};${scclPath};${RUNTIME_LIB_PATH};${MODELRUNTIME_LIB_PATH}")
install(TARGETS torch_tpu_python DESTINATION lib)

set_target_properties(torch_tpu_python PROPERTIES SUFFIX ".$ENV{CHIP_ARCH}.so")
add_custom_command(TARGET torch_tpu_python POST_BUILD
    COMMAND patchelf --set-soname libtorch_tpu_python.so $<TARGET_FILE:torch_tpu_python>
    COMMENT "Change SONAME of torch_tpu_python.so"
    DEPENDS torch_tpu_python VERBATIM)