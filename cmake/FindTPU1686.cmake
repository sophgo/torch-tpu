include(FindPackageHandleStandardArgs)

if (NOT DEFINED ENV{TPUTRAIN_TOP})
    message(FATAL_ERROR "Please source scripts/envsetup.sh")
endif()

set(TPU1686_PATH $ENV{TPUTRAIN_TOP}/../TPU1686 CACHE PATH "The path of TPU1686 directory")

find_path(
    tpuDNN_INCLUDE_DIR
    NAMES tpuDNN.h
    HINTS
    $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/include
    ${TPU1686_PATH}/tpuDNN/include)
find_path(
    tpurt_INCLUDE_DIR
    NAMES tpu_runtime_api.h
    HINTS
    $ENV{TPUTRAIN_TOP}/third_party/runtime_api/include
    ${TPU1686_ROOT}/runtime/)
find_path(
    SCCL_INCLUDE_DIR
    NAMES sccl.h
    HINTS
    $ENV{TPUTRAIN_TOP}/third_party/sccl/include
    ${TPU1686_PATH}/sccl/include)

if(("$ENV{SOC_CROSS_MODE}" STREQUAL "ON") OR ("$ENV{SOC_MODE}" STREQUAL "ON"))
    find_library(
        tpurt_LIBRARY
        NAMES tpurt-${PLATFORM} tpurt
        HINTS
        $ENV{TPUTRAIN_TOP}/third_party/runtime_api/lib_$ENV{CHIP_ARCH}/
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}_riscv/runtime/)

    find_library(
        SCCL_LIBRARY
        NAMES sccl-${PLATFORM} sccl
        HINTS
        $ENV{TPUTRAIN_TOP}/third_party/sccl/lib/
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}_riscv/sccl/)
else()
    find_library(
        tpurt_LIBRARY
        NAMES tpurt
        HINTS
        $ENV{TPUTRAIN_TOP}/third_party/runtime_api/lib_$ENV{CHIP_ARCH}/
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}/runtime/)

    find_library(
        SCCL_LIBRARY
        NAMES sccl
        HINTS
        $ENV{TPUTRAIN_TOP}/third_party/sccl/lib/
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}/sccl/)
endif()

set(additional_vars)

if ($ENV{CHIP_ARCH} STREQUAL "bm1684x")
    if($ENV{SOC_CROSS_MODE} STREQUAL "ON")
        find_library(
            cmodel_firmware_LIBRARY
            NAMES cmodel_firmware
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH})
        find_library(
            firmware_LIBRARY
            NAMES firmware_core bm1684x
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/arm)
    else()
        find_library(
            cmodel_firmware_LIBRARY
            NAMES cmodel_firmware
            HINTS
            #$ENV{TPUTRAIN_TOP}/../TPU1686/build_$ENV{CHIP_ARCH}/firmware_core/
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)

        find_library(
            firmware_LIBRARY
            NAMES firmware_core bm1684x
            HINTS
            #$ENV{TPUTRAIN_TOP}/../TPU1686/build_fw_$ENV{CHIP_ARCH}/firmware_core/
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)
    endif()
elseif ($ENV{CHIP_ARCH} STREQUAL "bm1686")
    if($ENV{SOC_CROSS_MODE} STREQUAL "ON")
        find_library(
            cmodel_firmware_LIBRARY
            NAMES cmodel_firmware
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH})
        find_library(
            firmware_LIBRARY
            NAMES firmware_core bm1686
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/arm)
    else()
        find_library(
            cmodel_firmware_LIBRARY
            NAMES cmodel_firmware
            HINTS
            #$ENV{TPUTRAIN_TOP}/../TPU1686/build_$ENV{CHIP_ARCH}/firmware_core/
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)

        find_library(
            firmware_LIBRARY
            NAMES firmware_core bm1686
            HINTS
            #$ENV{TPUTRAIN_TOP}/../TPU1686/build_fw_$ENV{CHIP_ARCH}/firmware_core/
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)
    endif()
elseif ($ENV{CHIP_ARCH} STREQUAL "sg2260" OR $ENV{CHIP_ARCH} STREQUAL "sg2260e")

    if ($ENV{CHIP_ARCH} STREQUAL "sg2260")
        set(emu_name tpuv7_emulator)
    else()
        set(emu_name tpuv7.1_emulator)
    endif()

    if(("$ENV{SOC_CROSS_MODE}" STREQUAL "ON") OR ("$ENV{SOC_MODE}" STREQUAL "ON"))
        find_library(
            cmodel_firmware_LIBRARY
            NAMES ${emu_name}-riscv
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/)

        find_library(
            firmware_LIBRARY
            NAMES libfirmware_core-riscv.a
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)
    else()
        find_library(
            cmodel_firmware_LIBRARY
            NAMES ${emu_name}
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/
            ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}/firmware_core/)

        find_library(
            firmware_LIBRARY
            NAMES libfirmware_core.a
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/
            ${TPU1686_PATH}/build_fw_$ENV{CHIP_ARCH}/firmware_core/)
    endif()

endif()

if(("$ENV{SOC_CROSS_MODE}" STREQUAL "ON") OR ("$ENV{SOC_MODE}" STREQUAL "ON"))
    if ($ENV{CHIP_ARCH} STREQUAL "bm1684x" OR $ENV{CHIP_ARCH} STREQUAL "bm1686")
        find_library(
            tpuDNN_LIBRARY
            NAMES tpudnn
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/$ENV{CHIP_ARCH}_lib/arm
            ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}/tpuDNN/src/)
    else()
        find_library(
            tpuDNN_LIBRARY
            NAMES tpudnn-riscv
            HINTS
            $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/$ENV{CHIP_ARCH}_lib/)
    endif()
else()
    find_library(
        tpuDNN_LIBRARY
        NAMES tpudnn
        HINTS
        $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/$ENV{CHIP_ARCH}_lib/
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}/tpuDNN/src/)
endif()

find_package_handle_standard_args(
    TPU1686
    REQUIRED_VARS tpuDNN_INCLUDE_DIR cmodel_firmware_LIBRARY firmware_LIBRARY tpuDNN_LIBRARY tpurt_INCLUDE_DIR tpurt_LIBRARY)

if (TPU1686_FOUND)
    add_library(TPU1686::tpuDNN IMPORTED INTERFACE)
    target_include_directories(TPU1686::tpuDNN INTERFACE ${tpuDNN_INCLUDE_DIR})

    add_library(TPU1686::cmodel_firmware IMPORTED SHARED)
    set_target_properties(
        TPU1686::cmodel_firmware PROPERTIES
        IMPORTED_LOCATION ${cmodel_firmware_LIBRARY})

    add_library(TPU1686::firmware IMPORTED SHARED)
    set_target_properties(
        TPU1686::firmware PROPERTIES
        IMPORTED_LOCATION ${firmware_LIBRARY})

    add_library(TPU1686::sccl IMPORTED INTERFACE)
    target_include_directories(TPU1686::sccl INTERFACE ${SCCL_INCLUDE_DIR})

    add_library(TPU1686::tpurt IMPORTED INTERFACE)
    target_include_directories(TPU1686::tpurt INTERFACE ${tpurt_INCLUDE_DIR})
endif()
