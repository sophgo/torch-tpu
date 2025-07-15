option(DEBUG "option for debug" OFF)

if(DEBUG OR "$ENV{TPUTRAIN_BUILD_TYPE}" STREQUAL "ON")
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-DDEBUG)
  if(USING_CMODEL)
    add_definitions(-rdynamic)
  endif()
else()
  set(CMAKE_BUILD_TYPE "Release")
  add_definitions(-O3)
endif()

add_definitions(-Wno-address-of-packed-member
                -fno-builtin-memcpy
                -fno-builtin-memset
                -fno-builtin-memmove
                -fsigned-char
                -Wsign-compare
                -Wunused-variable
                -Werror
                -Wno-dev)

if (NOT USING_CMODEL)
  set(CMAKE_C_COMPILER $ENV{RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
  set(CMAKE_ASM_COMPILER $ENV{RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER $ENV{RISCV_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)
endif()