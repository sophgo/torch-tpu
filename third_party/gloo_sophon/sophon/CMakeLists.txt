# SOPHON_SRCS is the list of source files that we need to build libsophon.
set(SOPHON_SRCS)

set(CMAKE_CXX_STANDARD 17)

# SOPHON_HDRS is the list of header files that we need to install.
set(SOPHON_HDRS)

# Compiled sources in root directory
list(APPEND SOPHON_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/algorithm.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/types.cc"
  )

list(APPEND SOPHON_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/algorithm.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/context.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/math.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/pairwise_exchange.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/types.h"
  )

# if(USE_CUDA)
#   file(GLOB SOPHON_CUDA_SRCS
#     "${CMAKE_CURRENT_SOURCE_DIR}/cuda*.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/cuda*.cu"
#     )

#   file(GLOB SOPHON_CUDA_HDRS
#     "${CMAKE_CURRENT_SOURCE_DIR}/cuda*.h"
#     )
# endif()

# if(USE_SOPHGO)
#   file(GLOB SOPHON_SOPHGO_SRCS
#     "${CMAKE_CURRENT_SOURCE_DIR}/sophon*.cc"
#     "${CMAKE_CURRENT_SOURCE_DIR}/sophon*.cu"
#   )
#   file(GLOB SOPHON_SOPHGO_HDRS
#     "${CMAKE_CURRENT_SOURCE_DIR}/sophon*.h"
#     )
# endif()

if(USE_ROCM)
  file(GLOB SOPHON_HIP_SRCS
    "${HIPIFY_OUTPUT_DIR}/hip*.cc"
    "${HIPIFY_OUTPUT_DIR}/hip*.hip"
    )

  file(GLOB SOPHON_HIP_HDRS
    "${HIPIFY_OUTPUT_DIR}/hip*.h"
    )
endif()

# Compile TCP transport only on Linux (it uses epoll(2))
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(SOPHON_HAVE_TRANSPORT_TCP 1)
  if(${USE_TCP_OPENSSL_LINK} OR ${USE_TCP_OPENSSL_LOAD})
    find_package(OpenSSL 1.1 REQUIRED EXACT)
    set(SOPHON_HAVE_TRANSPORT_TCP_TLS 1)
  else()
    set(SOPHON_HAVE_TRANSPORT_TCP_TLS 0)
  endif()
else()
  set(SOPHON_HAVE_TRANSPORT_TCP 0)
  set(SOPHON_HAVE_TRANSPORT_TCP_TLS 0)
endif()

# Compile ibverbs transport if libibverbs is available
if(USE_IBVERBS)
  set(SOPHON_HAVE_TRANSPORT_IBVERBS 1)
else()
  set(SOPHON_HAVE_TRANSPORT_IBVERBS 0)
endif()

# Compile uv transport if libuv is available
if(USE_LIBUV)
  set(SOPHON_HAVE_TRANSPORT_UV 1)
else()
  set(SOPHON_HAVE_TRANSPORT_UV 0)
endif()

# if(USE_SOPHGO)
#   set(SOPHON_HAVE_TRANSPORT_SOPHGO 1)
# else()
#   set(SOPHON_HAVE_TRANSPORT_SOPHGO 0)
# endif()

add_subdirectory(common)
# add_subdirectory(mpi)
# if(USE_CUDA AND USE_NCCL)
#   add_subdirectory(nccl)
# endif()
# if(USE_ROCM AND USE_RCCL)
#   add_subdirectory(nccl)
# endif()
add_subdirectory(rendezvous)
add_subdirectory(transport)
# add_subdirectory(sophgo)

# transport里拿到SOPHON_SRCS和SOPHON_HDRS列表

# Depend on pthreads for transport device threads
if(NOT MSVC)
  list(APPEND sophon_DEPENDENCY_LIBS pthread)
endif()

# Enable the following to get a list of source files
if(FALSE)
  message(STATUS "Sources: ")
  foreach(tmp ${SOPHON_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# Write configuration header.
# Set variables so macros have SOPHON_ prefix.
# set(SOPHON_USE_CUDA ${USE_CUDA})
set(SOPHON_USE_NCCL ${USE_NCCL})
set(SOPHON_USE_ROCM ${USE_ROCM})
set(SOPHON_USE_RCCL ${USE_RCCL})
set(SOPHON_USE_REDIS ${USE_REDIS})
set(SOPHON_USE_IBVERBS ${USE_IBVERBS})
set(SOPHON_USE_MPI ${USE_MPI})
set(SOPHON_USE_AVX ${USE_AVX})
set(SOPHON_USE_LIBUV ${USE_LIBUV})
# set(SOPHON_USE_SOPHGO ${USE_SOPHGO})
configure_file(config.h.in config.h)

# link sgdnn
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR} CACHE PATH "default project root")
add_library(sophon ${SOPHON_STATIC_OR_SHARED} ${SOPHON_SRCS})
include_directories($ENV{PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
include_directories($ENV{PYTORCH_INSTALL_DIR}/include)
link_directories($ENV{PYTORCH_INSTALL_DIR}/lib)
include_directories(${PROJECT_ROOT}/third_party/include)
include_directories(${PROJECT_ROOT}/torch_tpu/csrc/core)
include_directories(${PROJECT_ROOT})
find_package(TPU1686 REQUIRED)
target_link_libraries(sophon PRIVATE TPU1686::tpuDNN)
target_include_directories(sophon INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gloo
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR})

# if(USE_CUDA)
#   if(SOPHON_USE_CUDA_TOOLKIT)
#     enable_language(CUDA)
#     add_library(sophon_cuda ${SOPHON_STATIC_OR_SHARED} ${SOPHON_CUDA_SRCS})
#   else()
#     cuda_add_library(sophon_cuda ${SOPHON_CUDA_SRCS} ${SOPHON_STATIC_OR_SHARED})
#   endif()
#   target_link_libraries(sophon_cuda sophon ${sophon_cuda_DEPENDENCY_LIBS})
# endif()

# if(USE_SOPHGO)
#   add_library(sophon_sophgo ${SOPHON_STATIC_OR_SHARED} ${SOPHON_SOPHGO_SRCS})
#   target_link_libraries(sophon_sophgo sophon)
# endif()

if(USE_ROCM)
  sophon_hip_add_library(sophon_hip ${SOPHON_HIP_SRCS})
  target_link_libraries(sophon_hip sophon)
endif()
if(USE_LIBUV)
  target_link_libraries(sophon PRIVATE uv_a)
endif()

if(USE_TCP_OPENSSL_LINK)
  target_compile_definitions(sophon PRIVATE USE_TCP_OPENSSL_LINK)
  target_link_libraries(sophon PRIVATE OpenSSL::SSL OpenSSL::Crypto)
elseif(USE_TCP_OPENSSL_LOAD)
  target_compile_definitions(sophon PRIVATE USE_TCP_OPENSSL_LOAD)
  # Explicitly include the OPENSSL_INCLUDE_DIR, just in case it's not already in our path
  target_include_directories(sophon PRIVATE ${OPENSSL_INCLUDE_DIR})
  target_link_libraries(sophon PRIVATE dl)
endif()

target_link_libraries(sophon PRIVATE ${sophon_DEPENDENCY_LIBS})

# Add Interface include directories that are relocatable.
target_include_directories(sophon INTERFACE $<INSTALL_INTERFACE:include>)
# if(USE_CUDA)
#   target_include_directories(sophon_cuda INTERFACE $<INSTALL_INTERFACE:include>)
# endif()
if(USE_ROCM)
  target_include_directories(sophon_hip INTERFACE $<INSTALL_INTERFACE:include>)
endif()

# Install if necessary.
# If the Sophon build is included from another project's build, it may
# want to statically link with Sophon and not install any artifacts.
if(SOPHON_INSTALL)
  install(TARGETS sophon EXPORT SophonTargets
      DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  # if(USE_CUDA)
  #   install(TARGETS sophon_cuda EXPORT SophonTargets
  #       DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  # endif()
  if(USE_ROCM)
    install(TARGETS sophon_hip EXPORT SophonTargets
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  endif()
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/sophon)
  foreach(HEADER ${SOPHON_HDRS})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "sophon" DIR ${DIR})
    install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
  endforeach()
  # if(USE_CUDA)
  #   foreach(HEADER ${SOPHON_CUDA_HDRS})
  #     string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
  #     string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "sophon" DIR ${DIR})
  #     install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
  #   endforeach()
  # endif()
  if(USE_SOPHGO)
    foreach(HEADER ${SOPHON_SOPHGO_HDRS})
      string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
      string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "sophon" DIR ${DIR})
      install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
    endforeach()
  endif()
  if(USE_ROCM)
    foreach(HEADER ${SOPHON_HIP_HDRS})
      string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
      string(REGEX REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "sophon" DIR ${DIR})
      install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${DIR})
    endforeach()
  endif()
endif()

if(MSVC)
  add_definitions(-DNOMINMAX=1)
endif()

if(MSVC AND USE_LIBUV)
  add_custom_command(TARGET sophon POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${libuv_DLL_PATH}"
          $<TARGET_FILE_DIR:sophon>)
endif()

if(BUILD_TEST)
  add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
