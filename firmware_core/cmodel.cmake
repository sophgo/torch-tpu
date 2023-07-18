set(CMAKE_BUILD_TYPE "Debug")
if($ENV{CHIP_ARCH} STREQUAL "bm1684x")
if($ENV{LIBSOPHON_PATTERN} MATCHES $ENV{LIBSOPHON_STABLE})
	find_package(libsophon REQUIRED)
elseif($ENV{LIBSOPHON_PATTERN} MATCHES $ENV{LIBSOPHON_LATEST})
	include_directories($ENV{LIBSOPHON_TOP}/bmlib/include)
	include_directories($ENV{LIBSOPHON_TOP}/bmlib/src)
	link_directories($ENV{LIBSOPHON_TOP}/build/bmlib)
endif()
include_directories(${LIBSOPHON_INCLUDE_DIRS})
include_directories(include)
include_directories($ENV{TPUTRAIN_TOP}/include)
include_directories(${CMAKE_BINARY_DIR})
link_directories($ENV{TPUTRAIN_TOP}/lib)
link_directories(${CMAKE_BINARY_DIR})

set(KERNEL_HEADER "${CMAKE_BINARY_DIR}/firmware_core/kernel_module_data.h")
add_custom_command(
	OUTPUT ${KERNEL_HEADER}
	COMMAND echo "const unsigned int kernel_module_data[] = {0}\;" > ${KERNEL_HEADER}
)
# Add a custom target that depends on the custom command
add_custom_target(kernel_module ALL DEPENDS ${KERNEL_HEADER})

aux_source_directory(src KERNEL_SRC_FILES)
add_library(firmware SHARED ${KERNEL_SRC_FILES})
target_include_directories(firmware PRIVATE
	include
	$ENV{TPUTRAIN_TOP}/common/include
	$ENV{TPUTRAIN_TOP}/third_party/include
)
target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)

target_link_libraries(firmware PRIVATE $ENV{BMLIB_CMODEL_PATH} m)
set_target_properties(firmware PROPERTIES OUTPUT_NAME cmodel)

elseif($ENV{CHIP_ARCH} STREQUAL "sg2260")
	include_directories($ENV{TPUTRAIN_TOP}/common/include)
	include_directories($ENV{TPUTRAIN_TOP}/third_party/include)

	aux_source_directory(src KERNEL_SRC_FILES)

	add_library(firmware SHARED ${KERNEL_SRC_FILES})
	target_link_libraries(firmware PRIVATE $ENV{BMLIB_PATH} m)
	set_target_properties(firmware PROPERTIES OUTPUT_NAME cmodel)

endif()
