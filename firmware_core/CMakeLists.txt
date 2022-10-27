project(FIRMWARE C)
if(USING_CMODEL)
  include(${PROJECT_ROOT}/toolchain-cmodel.cmake)
  include_directories(${PROJECT_ROOT}/third_party/include/c_model)
  #include_directories(${PROJECT_ROOT}/c_model/include)
else()
  include(${PROJECT_ROOT}/toolchain-device.cmake)
  include_directories(${PROJECT_ROOT}/firmware_top/include)
endif()

include_directories(${PROJECT_ROOT}/common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRCS
     ./src/atomic/*.c
     ./src/backward/*.c
)

list(APPEND SRCS ${CMAKE_SOURCE_DIR}/common/src/common.c)
add_library(firmware SHARED ${SRCS})

if(USING_CMODEL)
target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)
#link_directories(${PROJECT_ROOT}/third_party/lib)
target_link_libraries(firmware PRIVATE ${PROJECT_ROOT}/third_party/lib/libcmodel.so)
set_target_properties(firmware PROPERTIES OUTPUT_NAME firmware_cmodel)
endif()
