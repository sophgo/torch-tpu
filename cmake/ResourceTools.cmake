function (compile_binary_file bin_path name)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        if($ENV{SOC_CROSS_MODE} STREQUAL "ON")
            set(bfdarch "aarch64")
            set(bfdname "elf64-littleaarch64")
        else()
            set(bfdarch "i386:x86-64")
            set(bfdname "elf64-x86-64")
        endif()
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
        set(bfdarch "riscv:rv64")
        set(bfdname "elf64-littleriscv")
    else ()
        message(FATAL_ERROR "Not implemented system processor \"${CMAKE_SYSTEM_PROCESSOR}\"")
    endif()

    add_custom_command(
        OUTPUT ${name}.o
        COMMAND ln -sf ${bin_path} ${name}
        COMMAND ${CMAKE_OBJCOPY} -I binary -B ${bfdarch} -O ${bfdname}
            --rename-section .data=.rodata ${name} ${name}.o
        DEPENDS ${bin_path}
        VERBATIM)

    configure_file(${CMAKE_SOURCE_DIR}/cmake/binary_file.h ${CMAKE_CURRENT_BINARY_DIR}/${name}.h)
    add_library(${name} OBJECT ${CMAKE_CURRENT_BINARY_DIR}/${name}.o ${CMAKE_CURRENT_BINARY_DIR}/${name}.h)
    target_include_directories(${name} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()
