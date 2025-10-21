cmake_minimum_required(VERSION 3.25)

project(FIRMWARE C)

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake)

include(config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../config_common.cmake)

if (USING_LLM_PROFILE)
    add_definitions(-DUSING_LLM_TICK_TOCK_PROFILE)
endif()
if (NOT USING_CMODEL AND REMOVE_POLLS_IN_LLM)
    add_definitions(-DREMOVE_POLLS_IN_LLM)
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

if (NOT USING_CMODEL)
    include($ENV{CHIP_ARCH}-toolchain.cmake)
endif()

aux_source_directory(src KERNEL_SRC_FILES)
add_library(firmware SHARED ${KERNEL_SRC_FILES})

message(STATUS "Building torch-tpu firmware for chip $ENV{CHIP_ARCH}")
string(TOUPPER "BACKEND_$ENV{CHIP_ARCH}" arch_flag)
target_compile_definitions(firmware PRIVATE -D${arch_flag})
target_include_directories(
    firmware PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/firmware/include)

find_package(TPU1686 REQUIRED)
if (USING_CMODEL)
    target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)
    target_link_libraries(firmware PRIVATE TPU1686::cmodel_firmware)
    set_target_properties(firmware PROPERTIES BUILD_RPATH
        "$ORIGIN;")
    install(TARGETS firmware DESTINATION lib)
else()
    if(USING_RTTHREAD)
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../tpuv7-runtime/rt-thread/bsp/sophgo/sg2260_tpu/install/include)

        set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
        find_library(LIBM_STATIC_LIB c)
        if(NOT LIBM_STATIC_LIB)
            message(FATAL_ERROR "Static libc (libc.a) not found!")
        endif()
        set(CMAKE_FIND_LIBRARY_SUFFIXES .so .a)
        message(INFO 输出了 ${LIBM_STATIC_LIB})

        target_link_options(firmware PRIVATE
            "LINKER:-Bstatic"
            "LINKER:${LIBM_STATIC_LIB}"
            "LINKER:-Bdynamic"
        )

        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../tpuv7-runtime/rt-thread/bsp/sophgo/sg2260_tpu/libtp_Image_0.a")
            message(FATAL_ERROR "Error: Not found libtp_Image_0.a\n")
        endif()

        target_link_libraries(firmware PRIVATE
            # ${LIBM_STATIC_LIB}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../bm_prebuilt_toolchains/gcc-riscv-musl/lib/gcc/riscv64-unknown-linux-musl/13.2.0/libgcc.a
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tpuv7-runtime/rt-thread/bsp/sophgo/sg2260_tpu/libtp_Image_0.a
        )

        set_target_properties(firmware PROPERTIES LINK_FLAGS -s)
        target_link_options(firmware PRIVATE -Wl,-allow-multiple-definition) # FIXME

        message("LIBM_STATIC_LIB = ${LIBM_STATIC_LIB} ")
        target_link_libraries(firmware PRIVATE
            "-Wl,--verbose,--whole-archive"
            ${LIBM_STATIC_LIB}
            TPU1686::firmware
            "-Wl,--no-whole-archive"
        )
        target_link_options(firmware PRIVATE -Wl,--no-undefined)
    else()
        set_target_properties(firmware PROPERTIES LINK_FLAGS -s)
        target_link_options(firmware PRIVATE -Wl,-allow-multiple-definition) # FIXME
        target_link_libraries(firmware PRIVATE -Wl,--whole-archive TPU1686::firmware -Wl,--no-whole-archive dl m)
        target_link_options(firmware PRIVATE -Wl,--no-undefined)
    endif()
endif()

set(PPL_SUPPORTED_ARCHES sg2260e)
if ("$ENV{CHIP_ARCH}" IN_LIST PPL_SUPPORTED_ARCHES)
    include(PPLTools)
    file(GLOB pl_files LIST_DIRECTORIES false
        "${CMAKE_CURRENT_SOURCE_DIR}/../torch_tpu/csrc/ops/my_ops/*.pl"
        "${CMAKE_CURRENT_SOURCE_DIR}/../torch_tpu/csrc/ops/native_ops/*.pl")
    compile_ppl_lib(${pl_files} pl_firmware)
    target_link_libraries(firmware PRIVATE -Wl,--whole-archive pl_firmware -Wl,--no-whole-archive)
endif()
