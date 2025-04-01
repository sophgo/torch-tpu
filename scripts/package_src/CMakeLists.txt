project(torch-tpu-python)

add_compile_options(-std=c++17 -Wno-attributes -Wall -fPIC)
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")

if (${BACKEND_SG2260} STREQUAL "ON")
add_definitions(-DBACKEND_SG2260)
elseif (${BACKEND_1684X} STREQUAL "ON")
add_definitions(-DBACKEND_1684X)
endif()

message(STATUS "PYTHON API ENABLE")
file(GLOB_RECURSE PYC_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/tpu/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
)

include_directories(${PYTORCH_INSTALL_DIR}/include
                    ${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include
                    ${PYTHON_INCLUDE_DIR}
                    $ENV{TORCH_TPU_PATH}/include
                    $ENV{SGDNN_PATH}/include
                    $ENV{TPUDNN_PATH}/include
                    $ENV{SCCL_PATH}/include
                    $ENV{TPUV7_RUNTIME_PATH}/include
                    )


set(tpuv7_rpath "/opt/tpuv7/tpuv7-current/lib")

add_library(torch_tpu_python SHARED ${PYC_SRCS})
target_link_directories(torch_tpu_python PRIVATE 
                          ${tpuv7_rpath}
                          ${PYTORCH_INSTALL_DIR}/lib
                          $ENV{TORCH_TPU_PATH}/lib
                        )
target_link_libraries(torch_tpu_python torch_tpu.$ENV{CHIP_ARCH} torch_python torch_cpu c10 tpuv7_rt tpuv7_modelrt)

set_target_properties(torch_tpu_python PROPERTIES INSTALL_RPATH 
      "$ORIGIN;${tpuv7_rpath}")
install(TARGETS torch_tpu_python DESTINATION lib)

set_target_properties(torch_tpu_python PROPERTIES SUFFIX ".$ENV{CHIP_ARCH}.so")
add_custom_command(TARGET torch_tpu_python POST_BUILD
    COMMAND patchelf --set-soname libtorch_tpu_python.so $<TARGET_FILE:torch_tpu_python>
    COMMENT "Change SONAME of torch_tpu_python.so"
    DEPENDS torch_tpu_python VERBATIM)