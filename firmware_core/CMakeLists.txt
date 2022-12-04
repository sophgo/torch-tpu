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

if(USING_CMODEL)
file(GLOB_RECURSE SRCS
     ./src/backward/*.c
)

list(APPEND SRCS ${CMAKE_SOURCE_DIR}/common/src/common.c)
add_library(firmware SHARED ${SRCS})

target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)
#link_directories(${PROJECT_ROOT}/third_party/lib)
target_link_libraries(firmware PRIVATE ${PROJECT_ROOT}/third_party/lib/libcmodel.so)
set_target_properties(firmware PROPERTIES OUTPUT_NAME firmware_cmodel)
else()
message("not cmodel")
message(${PROJECT_ROOT}/tool/build_firmware.sh)
aux_source_directory(./src/backward KERNEL_SRC_FILES)
add_custom_target(firmware
                  ALL
                  COMMAND ${PROJECT_ROOT}/tool/build_firmware.sh "${PROJECT_ROOT}/firmware_core/src/backward,${PROJECT_ROOT}/firmware_core/include" ${CMAKE_CURRENT_BINARY_DIR}/firmware
                  DEPENDS ${KERNEL_SRC_FILES})

#install(DIRECTORY ${CMAKE_BINARY_DIR}/firmware DESTINATION ${CMAKE_INSTALL_PREFIX}
#                 FILES_MATCHING PATTERN bm1684x.bin_*)
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/test DESTINATION ${CMAKE_INSTALL_PREFIX})
#install(FILES $ENV{PROJECT_ROOT}/tool/load_firmware.py DESTINATION scripts)
endif()
