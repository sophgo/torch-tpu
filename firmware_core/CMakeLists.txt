cmake_minimum_required(VERSION 3.25)

project(FIRMWARE C)

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)

include(config.cmake)

if (USING_LLM_PROFILE)
    add_definitions(-DUSING_LLM_TICK_TOCK_PROFILE)
endif()

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

if (NOT USING_CMODEL)
    include($ENV{CHIP_ARCH}-toolchain.cmake)
endif()

aux_source_directory(src KERNEL_SRC_FILES)
add_library(firmware SHARED ${KERNEL_SRC_FILES})

message(STATUS "Building torch-tpu firmware for chip $ENV{CHIP_ARCH}")
string(TOUPPER "BACKEND_$ENV{CHIP_ARCH}" arch_flag)
target_compile_definitions(firmware PRIVATE -D${arch_flag})

find_package(TPU1686 REQUIRED)
if (USING_CMODEL)
    target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)
    target_link_libraries(firmware PRIVATE TPU1686::cmodel_firmware)
else()
    set_target_properties(firmware PROPERTIES LINK_FLAGS -s)
    target_link_options(firmware PRIVATE -Wl,-allow-multiple-definition) # FIXME
    target_link_libraries(firmware PRIVATE -Wl,--whole-archive TPU1686::firmware -Wl,--no-whole-archive m)
    target_link_options(firmware PRIVATE -Wl,--no-undefined)
endif()
