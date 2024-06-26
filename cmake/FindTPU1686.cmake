include(FindPackageHandleStandardArgs)

if (NOT DEFINED ENV{TPUTRAIN_TOP})
    message(FATAL_ERROR "Please source scripts/envsetup.sh")
endif()

find_path(
    tpuDNN_INCLUDE_DIR
    NAMES tpuDNN.h
    HINTS
    $ENV{TPUTRAIN_TOP}/../TPU1686/tpuDNN/include
    $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/include)

if ($ENV{CHIP_ARCH} STREQUAL "bm1684x")
    find_library(
        cmodel_firmware_LIBRARY
        NAMES cmodel_firmware
        HINTS
        $ENV{TPUTRAIN_TOP}/../TPU1686/build_cmodel_fw/firmware_core/
        $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)

    find_library(
        firmware_LIBRARY
        NAMES firmware_core bm1684x
        HINTS
        $ENV{TPUTRAIN_TOP}/../TPU1686/build_fw/firmware_core/
        $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)
elseif ($ENV{CHIP_ARCH} STREQUAL "sg2260")
    find_library(
        cmodel_firmware_LIBRARY
        NAMES cmodel_firmware tpuv7_emulator
        HINTS
        $ENV{TPUTRAIN_TOP}/../TPU1686/build_cmodel_fw/firmware_core/
        $ENV{TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/)

    find_library(
        firmware_LIBRARY
        NAMES firmware_core
        HINTS
        $ENV{TPUTRAIN_TOP}/../TPU1686/build_fw/firmware_core/
        $ENV{TPUTRAIN_TOP}/third_party/firmware/$ENV{CHIP_ARCH}/)
endif()

find_library(
    tpuDNN_LIBRARY
    NAMES tpudnn
    HINTS
    $ENV{TPUTRAIN_TOP}/../TPU1686/build_fw/tpuDNN/src/
    $ENV{TPUTRAIN_TOP}/third_party/tpuDNN/$ENV{CHIP_ARCH}_lib/)

find_package_handle_standard_args(
    TPU1686
    REQUIRED_VARS tpuDNN_INCLUDE_DIR cmodel_firmware_LIBRARY firmware_LIBRARY tpuDNN_LIBRARY)

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
endif()
