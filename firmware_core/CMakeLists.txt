project(FIRMWARE C)

if(USING_CMODEL)
  include(${PROJECT_ROOT}/toolchain-cmodel.cmake)
else()
  include(${PROJECT_ROOT}/toolchain-device.cmake)
endif()

include_directories(${PROJECT_ROOT}/common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

aux_source_directory(src KERNEL_SRC_FILES)
if(USING_CMODEL)
  add_library(firmware SHARED ${KERNEL_SRC_FILES})
  target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)
  target_link_libraries(firmware ${PROJECT_ROOT}/third_party/libbm1684x_cmodel.so)
  set_target_properties(firmware PROPERTIES OUTPUT_NAME firmware_cmodel)
else()
  message(${PROJECT_ROOT}/scripts/build_firmware.sh)
  add_custom_target(firmware
                    ALL
                    COMMAND ${PROJECT_ROOT}/scripts/build_firmware.sh "${PROJECT_ROOT}/firmware_core/src,${PROJECT_ROOT}/firmware_core/include" ${CMAKE_CURRENT_BINARY_DIR}/firmware
                    DEPENDS ${KERNEL_SRC_FILES})
endif()
