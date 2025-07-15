include(FindPackageHandleStandardArgs)

if (NOT DEFINED ENV{TPUTRAIN_TOP})
    message(FATAL_ERROR "Please source scripts/envsetup.sh")
endif()

set(TPU1686_PATH $ENV{TPUTRAIN_TOP}/../TPU1686 CACHE PATH "The path of TPU1686 directory")

find_path(
    tpuDNN_INCLUDE_DIR
    NAMES tpuDNN.h
    HINTS
    ${TPU1686_PATH}/tpuDNN/include
    $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/include)
find_path(
    tpurt_INCLUDE_DIR
    NAMES tpu_runtime_api.h
    HINTS
    ${TPU1686_ROOT}/runtime/
    $ENV{TPUTRAIN_TOP}/third_party/runtime_api/include)
find_path(
    SCCL_INCLUDE_DIR
    NAMES sccl.h
    HINTS
    ${TPU1686_PATH}/sccl/include
    $ENV{TPUTRAIN_TOP}/third_party/sccl/include)

if($ENV{SOC_CROSS_MODE} STREQUAL "ON")
    find_library(
        tpurt_LIBRARY
        NAMES tpurt-${PLATFORM} tpurt
        HINTS
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}_riscv/runtime/
        $ENV{TPUTRAIN_TOP}/third_party/runtime_api/lib_$ENV{CHIP_ARCH}/)

    find_library(
        SCCL_LIBRARY
        NAMES sccl-${PLATFORM} sccl
        HINTS
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}_riscv/sccl/
        $ENV{TPUTRAIN_TOP}/third_party/sccl/lib/)
else()
    find_library(
        tpurt_LIBRARY
        NAMES tpurt
        HINTS
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}/runtime/
        $ENV{TPUTRAIN_TOP}/third_party/runtime_api/lib_$ENV{CHIP_ARCH}/)

    find_library(
        SCCL_LIBRARY
        NAMES sccl
        HINTS
        ${TPU1686_PATH}/build_$ENV{CHIP_ARCH}/sccl/
        $ENV{TPUTRAIN_TOP}/third_party/sccl/lib/)
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
elseif ($ENV{CHIP_ARCH} STREQUAL "sg2260")
    if($ENV{SOC_CROSS_MODE} STREQUAL "ON")
        find_library(
            cmodel_firmware_LIBRARY
            NAMES tpuv7_emulator-riscv
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
            NAMES cmodel_firmware tpuv7_emulator
            HINTS
            $ENV{TPUTRAIN_TOP}/../TPU1686/build_$ENV{CHIP_ARCH}/firmware_core/
            $ENV{TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/)

        find_library(
            firmware_LIBRARY
            NAMES libfirmware_core.a
            HINTS
            $ENV{TPUTRAIN_TOP}/../TPU1686/build_fw_$ENV{CHIP_ARCH}/firmware_core/
            $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)
    endif()

endif()

if($ENV{SOC_CROSS_MODE} STREQUAL "ON")
    if ($ENV{CHIP_ARCH} STREQUAL "bm1684x")
        find_library(
            tpuDNN_LIBRARY
            NAMES tpudnn
            HINTS
            $ENV{TPUTRAIN_TOP}/../TPU1686/build_$ENV{CHIP_ARCH}/tpuDNN/src/
            $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/$ENV{CHIP_ARCH}_lib/arm)
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
        $ENV{TPUTRAIN_TOP}/../TPU1686/build_$ENV{CHIP_ARCH}/tpuDNN/src/
        $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/$ENV{CHIP_ARCH}_lib/)
endif()

find_package_handle_standard_args(
    TPU1686
    REQUIRED_VARS tpuDNN_INCLUDE_DIR cmodel_firmware_LIBRARY firmware_LIBRARY tpuDNN_LIBRARY tpurt_INCLUDE_DIR tpurt_LIBRARY)

if (TPU1686_FOUND)
    add_library(TPU1686::tpuDNN IMPORTED SHARED)
    set_target_properties(
        TPU1686::tpuDNN PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${tpuDNN_INCLUDE_DIR}
        IMPORTED_LOCATION ${tpuDNN_LIBRARY})

    add_library(TPU1686::cmodel_firmware IMPORTED SHARED)
    set_target_properties(
        TPU1686::cmodel_firmware PROPERTIES
        IMPORTED_LOCATION ${cmodel_firmware_LIBRARY})

    add_library(TPU1686::firmware IMPORTED SHARED)
    set_target_properties(
        TPU1686::firmware PROPERTIES
        IMPORTED_LOCATION ${firmware_LIBRARY})

    add_library(TPU1686::sccl IMPORTED SHARED)
    set_target_properties(TPU1686::sccl PROPERTIES
        IMPORTED_LOCATION ${SCCL_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${SCCL_INCLUDE_DIR})

    add_library(TPU1686::tpurt SHARED IMPORTED)
    set_target_properties(
        TPU1686::tpurt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${tpurt_INCLUDE_DIR}
        IMPORTED_LOCATION ${tpurt_LIBRARY})
endif()
