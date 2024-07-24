if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  message(WARNING "Sophon TCP-TLS transport is supported only on Linux(uses epoll API)")
else ()
  list(APPEND SOPHON_TRANSPORT_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/device.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/openssl.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair.cc"
    )
  list(APPEND SOPHON_TRANSPORT_HDRS
    "${CMAKE_CURRENT_SOURCE_DIR}/context.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/device.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/openssl.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair.h"
    )
  if (USE_TCP_OPENSSL_LOAD)
    list(APPEND SOPHON_TRANSPORT_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/dynamic_library.cc"
      )
    list(APPEND SOPHON_TRANSPORT_HDRS
      "${CMAKE_CURRENT_SOURCE_DIR}/dynamic_library.h"
      )
  endif ()
endif ()

set(SOPHON_TRANSPORT_SRCS ${SOPHON_TRANSPORT_SRCS} PARENT_SCOPE)
set(SOPHON_TRANSPORT_HDRS ${SOPHON_TRANSPORT_HDRS} PARENT_SCOPE)
