cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_TPU CXX)
include(${CMAKE_SOURCE_DIR}/cross-toolchain.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
add_compile_options(-std=c++17 -Wno-attributes -Wall -fPIC)
if ( NOT $ENV{TORCH_CXX11_ABI} )
  message(STATUS "TORCH_CXX11_ABI=0" )
  add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")
  add_definitions("-DPYBIND11_BUILD_ABI=\"_cxxabi1011\"")
endif()

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

## runtime & model-runtime
include_directories( ${RUNTIME_INCLUDE_PATH} ${MODELRUNTIME_INCLUDE_PATH} )

### tpudnn && sccl
find_package(TPU1686 REQUIRED)
get_filename_component(tpuDNNPath ${tpuDNN_LIBRARY} DIRECTORY)
get_filename_component(tpurtPath ${tpurt_LIBRARY} DIRECTORY)
get_filename_component(scclPath   ${SCCL_LIBRARY}   DIRECTORY)
#################### END DEPENDS ####################

######## torch-tpu inc/src
include_directories($ENV{TPUTRAIN_TOP})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/distributed/c10d)

file(GLOB SRCS
  csrc/core/*.cpp
  csrc/core/Interface/*.cpp
  csrc/ops/native_ops/*.cpp
  csrc/ops/my_ops/*.cpp
  csrc/ops/autocast_ops/*.cpp
  csrc/aten/*.cpp
  csrc/aten/*/*.cpp
)

if(BACKEND_SG2260E)

macro(collect_ppl_host relative_path)
    set(abs_path "${CMAKE_CURRENT_SOURCE_DIR}/${relative_path}")
    file(GLOB pl_files LIST_DIRECTORIES false "${abs_path}/*.pl")

    foreach(pl_file IN LISTS pl_files)
        get_filename_component(base_name ${pl_file} NAME_WE)
        get_filename_component(dir_path ${pl_file} DIRECTORY)
        list(APPEND SRCS "${dir_path}/${base_name}/host/${base_name}.cpp")
        include_directories("${dir_path}/${base_name}/include")
    endforeach()
endmacro()

collect_ppl_host(csrc/ops/my_ops)
collect_ppl_host(csrc/ops/native_ops)

include_directories($ENV{PPL_INSTALL_PATH}/deps/common/host/include/)
endif()

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

list(APPEND concated_src ${DIST_SRCS})

if (NOT DEFINED KERNEL_MODULE_PATH)
    message(FATAL_ERROR "${KERNEL_MODULE_PATH} Please specify -DKERNEL_MODULE_PATH=")
endif()
include(ResourceTools)
compile_binary_file(${KERNEL_MODULE_PATH} torch_tpu_kernel_data)
set(kernel_srcs $<TARGET_PROPERTY:torch_tpu_kernel_data,SOURCES>)

add_library(torch_tpu SHARED ${concated_src} ${kernel_srcs})
target_link_libraries(torch_tpu c10 torch_cpu TPU1686::tpuDNN torch_tpu_kernel_data TPU1686::sccl TPU1686::tpurt sophon)

if(BACKEND_1684X OR BACKEND_1686)
  target_link_directories(torch_tpu PRIVATE ${RUNTIME_LIB_PATH} ${MODELRUNTIME_LIB_PATH})
  target_link_libraries(torch_tpu bmlib bmrt)
  set_target_properties(torch_tpu PROPERTIES INSTALL_RPATH
                        "$ORIGIN;${RUNTIME_LIB_PATH};${MODELRUNTIME_LIB_PATH};${tpuDNNPath}:${scclPath}")
  install(TARGETS torch_tpu DESTINATION lib)
elseif(BACKEND_SG2260 OR BACKEND_SG2260E)
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
                       ${tpuDNNPath};${scclPath};${tpurtPath}")
  if(($ENV{SOC_CROSS_MODE} STREQUAL "ON") AND (NOT EXISTS ${TPUv7_RUNTIME_REPO_PATH}))
    target_link_libraries(torch_tpu tpuv7_rt_riscv tpuv7_modelrt_riscv)
  else()
    target_link_libraries(torch_tpu tpuv7_rt tpuv7_modelrt)
  endif()
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
if($ENV{SOC_CROSS_MODE} STREQUAL "ON")
  if($ENV{CHIP_ARCH} STREQUAL "sg2260" OR $ENV{CHIP_ARCH} STREQUAL "sg2260e")
    include_directories($ENV{CROSS_TOOLCHAINS}/riscv64-linux-x86_64/python3.10)
  elseif($ENV{CHIP_ARCH} STREQUAL "bm1684x" OR $ENV{CHIP_ARCH} STREQUAL "bm1686")
    include_directories($ENV{CROSS_TOOLCHAINS}/Python-3.8.2/python_3.8.2/include/python3.8)
  else()
    message(FATAL_ERROR "chip not support: $ENV{CHIP_ARCH}")
  endif()
else()
  include_directories(${PYTHON_INCLUDE_DIR})
endif()
message(STATUS "PYTHON API ENABLE")
file(GLOB_RECURSE PYC_SRCS
  csrc/tpu/*.cpp
  csrc/utils/*.cpp
)
add_library(torch_tpu_python SHARED ${PYC_SRCS})

if ((BACKEND_SG2260 OR BACKEND_SG2260E) AND (EXISTS ${TPUv7_RUNTIME_REPO_PATH}))
  target_link_directories(torch_tpu_python PRIVATE ${TPUv7_REPO_RT_LIB_PATH}
                                                  ${TPUv7_REPO_RT_AP_LIB_PATH}
                                                  ${TPUv7_REPO_MODEL_RT_LIB_PATH}
                                                  ${TPUv7_REPO_MODEL_BMODEL_LIB_PATH})
else()
  target_link_directories(torch_tpu_python PRIVATE ${RUNTIME_LIB_PATH} ${MODELRUNTIME_LIB_PATH})
endif()

target_link_libraries(torch_tpu_python torch_tpu)

set_target_properties(torch_tpu_python PROPERTIES INSTALL_RPATH
      "$ORIGIN;${tpuv7_rpath};${tpuDNNPath};${tpurtPath};${scclPath};${RUNTIME_LIB_PATH};${MODELRUNTIME_LIB_PATH}")
install(TARGETS torch_tpu_python DESTINATION lib)

set_target_properties(torch_tpu_python PROPERTIES SUFFIX ".$ENV{CHIP_ARCH}.so")
add_custom_command(TARGET torch_tpu_python POST_BUILD
    COMMAND patchelf --set-soname libtorch_tpu_python.so $<TARGET_FILE:torch_tpu_python>
    COMMENT "Change SONAME of torch_tpu_python.so"
    DEPENDS torch_tpu_python VERBATIM)
