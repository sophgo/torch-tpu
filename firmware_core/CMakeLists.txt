project(FIRMWARE C)

include(config.cmake)

set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS}")

add_definitions(-Wno-address-of-packed-member
                -fno-builtin-memcpy 
                -fno-builtin-memset 
                -fno-builtin-memmove
                -fsigned-char 
                -Wsign-compare 
                -Wunused-variable 
                -Werror 
                -Wno-dev)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/firmware/include)

if (USING_CMODEL)
    include(cmodel.cmake)
elseif (PCIE_MODE)
    include(pcie.cmake)
elseif (SOC_MODE)
    include(soc.cmake)
endif()
