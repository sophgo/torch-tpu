if (TARGET tpurt)
    add_library(tpurt::tpurt ALIAS tpurt)
    return()
endif()

include(FindPackageHandleStandardArgs)

find_path(
    tpurt_INCLUDE_DIR
    NAMES tpuv7_rt.h
    HINTS $ENV{TPURT_TOP}/cdmlib/host/cdm_runtime/include)
find_library(
    tpurt_LIBRARY
    NAMES tpuv7_rt
    HINTS $ENV{TPURT_TOP}/build/emulator/cdmlib/host/cdm_runtime)

find_package_handle_standard_args(
    tpurt
    REQUIRED_VARS tpurt_INCLUDE_DIR tpurt_LIBRARY)

if (tpurt_FOUND)
    add_library(tpurt::tpurt IMPORTED SHARED)
    set_target_properties(
        tpurt::tpurt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${tpurt_INCLUDE_DIR}
        IMPORTED_LOCATION ${tpurt_LIBRARY})
endif()
