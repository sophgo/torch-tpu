if (TARGET tpudnn)
    add_library(tpudnn::tpudnn ALIAS tpudnn)
    return()
endif()

include(FindPackageHandleStandardArgs)

find_path(
    tpudnn_INCLUDE_DIR
    NAMES tpuDNN.h
    HINTS $ENV{TPUDNN_PATH}/include)

find_library(
    tpudnn_LIBRARY
    NAMES tpudnn
    HINTS $ENV{TPUDNN_PATH}/$ENV{CHIP_ARCH}_lib)

find_package_handle_standard_args(
    tpudnn
    REQUIRED_VARS tpudnn_INCLUDE_DIR tpudnn_LIBRARY)

if (tpudnn_FOUND)
    add_library(tpudnn::tpudnn IMPORTED SHARED)
    set_target_properties(
        tpudnn::tpudnn PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${tpudnn_INCLUDE_DIR}
        IMPORTED_LOCATION ${tpudnn_LIBRARY})
endif()
