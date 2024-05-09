project(SGDNN CXX)

include(config.cmake)

set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread -fno-strict-aliasing -D_GLIBCXX_USE_CXX11_ABI=1 ${SAFETY_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_ROOT}/common/include)
include_directories(${PROJECT_SOURCE_DIR}/../third_party/firmware/include)

## runtime
include_directories(${RUNTIME_INCLUDE_PATH})
link_directories(${RUNTIME_LIB_PATH})

## kernel_module
include_directories(${CMAKE_BINARY_DIR}/firmware_core)

if(BACKEND_1684X)
  add_library(sgdnn SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn.cpp)
  target_link_libraries(sgdnn bmlib)
elseif(BACKEND_SG2260)
  add_library(sgdnn SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/sgdnn.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/tpukernel_launcher.cpp)
  target_link_libraries(sgdnn tpuv7_rt)
endif()