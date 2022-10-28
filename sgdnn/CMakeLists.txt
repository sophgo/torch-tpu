project(SGDNN CXX)

add_subdirectory(${PROJECT_ROOT}/third_party/include/pybind11 ${CMAKE_CURRENT_SOURCE_DIR})

include(${PROJECT_ROOT}/toolchain-cmodel.cmake)

include_directories(${PROJECT_ROOT}/common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_ROOT}/third_party/include/bmlib)
include_directories(${PROJECT_ROOT}/third_party/include/c_model)

link_directories(${PROJECT_ROOT}/third_party/lib)

pybind11_add_module(sgdnn ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn.cpp)

target_link_libraries(sgdnn PRIVATE bmlib cmodel)

INSTALL(TARGETS sgdnn LIBRARY DESTINATION /usr/lib)
