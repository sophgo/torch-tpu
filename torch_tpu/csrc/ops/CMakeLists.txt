cmake_minimum_required(VERSION 3.25)

project(BACKEND C)

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../../../cmake)

include(config.cmake)

include_directories($ENV{PPL_PROJECT_ROOT}/runtime/customize/include/)
include_directories($ENV{PPL_PROJECT_ROOT}/runtime/${CHIP_ARCH}/TPU1686/kernel/include/)
include_directories($ENV{PPL_PROJECT_ROOT}/runtime/kernel/)

macro(compile_ppl_lib relative_path)
    set(abs_path "${CMAKE_CURRENT_SOURCE_DIR}/${relative_path}")
    file(GLOB pl_files LIST_DIRECTORIES false "${abs_path}/*.pl")

    foreach(pl_file IN LISTS pl_files)
        get_filename_component(base_name ${pl_file} NAME_WE)
        get_filename_component(dir_path ${pl_file} DIRECTORY)
        set(PPL_CMD $ENV{PPL_PROJECT_ROOT}/bin/ppl-compile ${pl_file} -o ${dir_path}/${base_name} --chip ${CHIP_ARCH} --mode 6)
        string(JOIN " " PPL_CMD_WITH_SPACES ${PPL_CMD})
        message(${PPL_CMD_WITH_SPACES})
        add_custom_command(
            OUTPUT ${dir_path}/${base_name}/device/${base_name}.c
            COMMAND ${PPL_CMD}
            DEPENDS ${pl_file}
            VERBATIM
        )
        list(APPEND SRC "${dir_path}/${base_name}/device/${base_name}.c")
    endforeach()
endmacro()

set(SRC "")
compile_ppl_lib(my_ops)
compile_ppl_lib(native_ops)

message(STATUS "Compiling Backend Src file: ${SRC}")

add_library(firmware SHARED ${SRC})
message(STATUS "Building torch-tpu Backend for chip $ENV{CHIP_ARCH}")


target_compile_definitions(firmware PRIVATE -D${arch_flag})

find_package(TPU1686 REQUIRED)
if (USING_CMODEL)
    target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)
    target_compile_options(firmware PRIVATE -Wno-unused-variable)
    target_link_libraries(firmware PRIVATE TPU1686::cmodel_firmware)
    set_target_properties(firmware PROPERTIES BUILD_RPATH
        "$ORIGIN;")
    install(TARGETS firmware DESTINATION lib)
else()
    set_target_properties(firmware PROPERTIES LINK_FLAGS -s)
    target_compile_options(firmware PRIVATE -Wno-unused-variable)
    target_link_options(firmware PRIVATE -Wl,-allow-multiple-definition) # FIXME
    target_link_libraries(firmware PRIVATE -Wl,--whole-archive TPU1686::firmware -Wl,--no-whole-archive m)
    target_link_options(firmware PRIVATE -Wl,--no-undefined)
endif()
