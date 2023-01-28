project(SGDNN CXX)

add_subdirectory(${PROJECT_ROOT}/third_party/include/pybind11 ${CMAKE_CURRENT_SOURCE_DIR})

if(SOC_MODE)
  include(${PROJECT_ROOT}/toolchain-device.cmake)
else()
  include(${PROJECT_ROOT}/toolchain-cmodel.cmake)
endif()

include_directories(${PROJECT_ROOT}/common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_ROOT}/third_party/include/bmlib)
if(USING_CMODEL)
  include_directories(${PROJECT_ROOT}/third_party/include/c_model)
endif()

link_directories(${PROJECT_ROOT}/third_party/lib)

add_library(sgdnn SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn.cpp)
pybind11_add_module(sgdnn_pybind11 ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn_pybind11.cpp)

if(USING_CMODEL)
  target_link_libraries(sgdnn bmlib_cmodel cmodel)
  target_link_libraries(sgdnn_pybind11 bmlib_cmodel cmodel)
else()
  target_link_libraries(sgdnn bmlib)
  target_link_libraries(sgdnn_pybind11 bmlib)
endif()
