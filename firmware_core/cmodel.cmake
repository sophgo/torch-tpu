set(CMAKE_BUILD_TYPE "Debug")

if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
	set(FIRMWARE_CMODEL_PATH $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/libcmodel_firmware.so)
elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
	set(FIRMWARE_CMODEL_PATH $ENV{TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/libtpuv7_emulator.so)
endif()
message(STATUS "FIRMWARE_CMODEL_PATH = ${FIRMWARE_CMODEL_PATH}")

set(KERNEL_HEADER "${CMAKE_BINARY_DIR}/firmware_core/kernel_module_data.h")
add_custom_command(
	OUTPUT ${KERNEL_HEADER}
	COMMAND echo "const unsigned int kernel_module_data[] = {0}\;" > ${KERNEL_HEADER}
)
# Add a custom target that depends on the custom command
add_custom_target(kernel_module ALL DEPENDS ${KERNEL_HEADER})

aux_source_directory(src KERNEL_SRC_FILES)
add_library(firmware SHARED ${KERNEL_SRC_FILES})
target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)

target_link_libraries(firmware PRIVATE ${FIRMWARE_CMODEL_PATH} m)
set_target_properties(firmware PROPERTIES OUTPUT_NAME cmodel)