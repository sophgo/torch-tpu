cmake_minimum_required(VERSION 3.25)

project(FIRMWARE C)

include(config.cmake)

add_definitions(-Wno-address-of-packed-member
                -fno-builtin-memcpy 
                -fno-builtin-memset 
                -fno-builtin-memmove
                -fsigned-char 
                -Wsign-compare 
                -Wunused-variable 
                -Werror 
                -Wno-dev)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/firmware/include)

if (USING_CMODEL)
    message(STATUS "Building cmodel firmware for $ENV{CHIP_ARCH}")
    if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
        set(FIRMWARE_PATH $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/libcmodel_firmware.so)
    elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
        set(FIRMWARE_PATH $ENV{TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/libtpuv7_emulator.so)
    endif()
else()
    message(STATUS "Building firmware for $ENV{CHIP_ARCH}")
    include($ENV{CHIP_ARCH}-toolchain.cmake)
    if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
        set(FIRMWARE_PATH $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/libbm1684x.a)
    elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
        set(FIRMWARE_PATH $ENV{TPUTRAIN_TOP}/third_party/firmware/sg2260/libfirmware_core.a)
    endif()
endif()

message(STATUS "FIRMWARE_PATH = ${FIRMWARE_PATH}")

aux_source_directory(src KERNEL_SRC_FILES)
add_library(firmware SHARED ${KERNEL_SRC_FILES})

if (USING_CMODEL)
    target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)
    target_link_libraries(firmware PRIVATE ${FIRMWARE_PATH})
else()
    set_target_properties(firmware PROPERTIES LINK_FLAGS -s)
    target_link_options(firmware PRIVATE -Wl,-allow-multiple-definition)
    target_link_libraries(firmware PRIVATE -Wl,--whole-archive ${FIRMWARE_PATH} -Wl,--no-whole-archive)
endif()
