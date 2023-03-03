cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(LibTorch-ResNet18-CIFAR10)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17) # To use C++ <filesystem>
include_directories(../../libtorch_plugin/tpu)
link_directories(../../libtorch_plugin/build)



add_library(cifar10 cifar10.cpp)
target_include_directories(cifar10 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cifar10 PRIVATE "${TORCH_LIBRARIES}")

add_executable(resnet-cifar-demo resnet-cifar-demo.cpp)
target_include_directories(resnet-cifar-demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(resnet-cifar-demo PRIVATE cifar10 "${TORCH_LIBRARIES}" libtorch_plugin)

