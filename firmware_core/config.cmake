option(DEBUG "option for debug" ON)

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