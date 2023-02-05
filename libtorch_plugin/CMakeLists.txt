project(LIBTORCH_PLUGIN CXX)

if(SOC_MODE)
  include(${PROJECT_ROOT}/toolchain-device.cmake)
else()
  include(${PROJECT_ROOT}/toolchain-cmodel.cmake)
endif()

add_compile_options(-std=c++14)
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(/opt/sophon/libsophon-current/include)
include_directories(${PROJECT_ROOT}/libtorch/include/torch/csrc/api/include)
include_directories(${PROJECT_ROOT}/libtorch/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tpu)
include_directories(${PROJECT_ROOT}/sgdnn/include)

link_directories(${PROJECT_ROOT}/libtorch/lib)
link_directories(/opt/sophon/libsophon-current/lib)
link_directories(${PROJECT_ROOT}/third_parth)
link_directories(${PROJECT_ROOT}/build/sgdnn)

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

if(USING_CMODEL)
  target_link_libraries(libtorch_plugin bmlib_cmodel)
else()
  target_link_libraries(libtorch_plugin bmlib)
endif()

add_executable(test test.cpp)
target_link_libraries(test libtorch_plugin)
install(TARGET test DESTINATION ./)
