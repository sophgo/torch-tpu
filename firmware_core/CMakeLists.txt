project(FIRMWARE C)
if(USING_CMODEL)
  include(${PROJECT_ROOT}/toolchain-cmodel.cmake)
  include_directories(${PROJECT_ROOT}/third_party/include/c_model)
else()
  include(${PROJECT_ROOT}/toolchain-device.cmake)
  include_directories(${PROJECT_ROOT}/firmware_top/include)
endif()

include_directories(${PROJECT_ROOT}/common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRCS
     ./src/backward/*.c
     ./src/forward/*.c
)

if(USING_CMODEL)
  add_library(firmware SHARED ${SRCS})
  target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)
  target_link_libraries(firmware ${PROJECT_ROOT}/third_party/lib/libbm1684x_cmodel.so)
  set_target_properties(firmware PROPERTIES OUTPUT_NAME firmware_cmodel)
else()
  message(${PROJECT_ROOT}/tool/build_firmware.sh)
  add_custom_target(firmware
                    ALL
                    COMMAND ${PROJECT_ROOT}/tool/build_firmware.sh "${PROJECT_ROOT}/firmware_core/src/forward,${PROJECT_ROOT}/firmware_core/src/backward,${PROJECT_ROOT}/firmware_core/include" ${CMAKE_CURRENT_BINARY_DIR}/firmware
                    DEPENDS ${SRCS})
endif()
