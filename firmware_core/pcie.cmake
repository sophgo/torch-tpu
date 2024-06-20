# Set the C compiler, The path needs to be modified
if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
    set(CMAKE_C_COMPILER $ENV{ARM_TOOLCHAIN}/bin/aarch64-none-linux-gnu-gcc)
elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
    set(CMAKE_C_COMPILER $ENV{RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
else()
    message(FATAL_ERROR "unsupport CHIP backend")
endif()

# Set the library directories for the shared library
if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
    link_directories(${PROJECT_SOURCE_DIR}/../third_party/firmware/$ENV{CHIP_ARCH})
elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
    link_directories(${PROJECT_SOURCE_DIR}/../third_party/tpuv7_runtime/$ENV{CHIP_ARCH}_firmware ${PROJECT_SOURCE_DIR}/../third_party/firmware/$ENV{CHIP_ARCH})
endif()

# Set the source files for the shared library
file(GLOB_RECURSE DEVICE_SRCS
    ${PROJECT_SOURCE_DIR}/src/*.c
)
# Create the shared library
set(SHARED_LIBRARY_OUTPUT_FILE "lib$ENV{CHIP_ARCH}_kernel_module")
add_library(${SHARED_LIBRARY_OUTPUT_FILE} SHARED ${DEVICE_SRCS})
if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
    target_link_libraries(${SHARED_LIBRARY_OUTPUT_FILE} -Wl,--whole-archive "lib$ENV{CHIP_ARCH}.a" -Wl,--no-whole-archive m)
elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
    target_link_libraries(${SHARED_LIBRARY_OUTPUT_FILE} -Wl,--allow-multiple-definition,--whole-archive "libfirmware_core.a" -Wl,--no-whole-archive m)
endif()
set_target_properties(${SHARED_LIBRARY_OUTPUT_FILE} PROPERTIES PREFIX "" SUFFIX ".so" COMPILE_FLAGS "-O2 -fPIC" LINK_FLAGS "-shared")

# Set the path to the input file
set(INPUT_FILE "${CMAKE_BINARY_DIR}/firmware_core/${SHARED_LIBRARY_OUTPUT_FILE}.so")
set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/firmware_core/kernel_module_data.h")
add_custom_command(
    OUTPUT ${OUTPUT_FILE}
    DEPENDS ${SHARED_LIBRARY_OUTPUT_FILE}
    COMMAND echo "const unsigned int kernel_module_data[] = {" > ${OUTPUT_FILE}
    COMMAND hexdump -v -e '1/4 \"0x%08x,\\n\"' ${INPUT_FILE} >> ${OUTPUT_FILE}
    COMMAND echo "}\;" >> ${OUTPUT_FILE}
)
# Add a custom target that depends on the custom command
add_custom_target(kernel_module DEPENDS ${OUTPUT_FILE})
