function (compile_ppl_lib)
    list(POP_BACK ARGV name)
    set(pl_files "${ARGV}")

    set(chip_arch $ENV{CHIP_ARCH})
    set(srcs)

    if ("${chip_arch}" STREQUAL "sg2260")
        set(ppl_arch "bm1690")
        set(ppl_mode 0)
    else()
        set(ppl_arch ${chip_arch})
        set(ppl_mode 6)
    endif()

    foreach (pl_file IN LISTS pl_files)
        get_filename_component(base_name ${pl_file} NAME_WE)
        set(output_dir ${CMAKE_BINARY_DIR}/ppl/${base_name})
        file(MAKE_DIRECTORY ${output_dir})
        set(PPL_CMD $ENV{PPL_INSTALL_PATH}/bin/ppl-compile ${pl_file} -o ${output_dir} --chip ${ppl_arch} --mode ${ppl_mode})
        set(output_c ${output_dir}/device/${base_name}.c)
        add_custom_command(
            OUTPUT ${output_c}
            COMMAND ${PPL_CMD}
            COMMAND sed -i "/TPUKERNEL_FUNC_REGISTER/d" ${output_c}
            DEPENDS ${pl_file}
            VERBATIM)
        list(APPEND srcs ${output_c})
    endforeach()

    add_library(${name} ${srcs})
    target_include_directories(
        ${name} PRIVATE
        $ENV{PPL_INSTALL_PATH}/deps/common/dev/utils/include/
        $ENV{PPL_INSTALL_PATH}/deps/common/dev/kernel/)
    target_compile_options(${name} PRIVATE
        -fPIC
        -Wno-error=sign-compare
        -Wno-error=unused-variable
        -Wno-error=unused-const-variable)

    if ("${chip_arch}" STREQUAL "sg2260e")
        target_include_directories(
            ${name} PRIVATE
            $ENV{PPL_INSTALL_PATH}/deps/chip/tpub_7_1_e/TPU1686/kernel/include/)
    elseif ("${chip_arch}" STREQUAL "sg2260")
        target_include_directories(
            ${name} PRIVATE
            $ENV{PPL_INSTALL_PATH}/deps/chip/tpub_7_1/TPU1686/kernel/include/)
    else()
        message(FATAL_ERROR "Unsupported arch ${chip_arch}")
    endif()
endfunction()
