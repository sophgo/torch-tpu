option(DEBUG "option for debug" OFF)

if(DEBUG OR "$ENV{TPUTRAIN_BUILD_TYPE}" STREQUAL "ON")
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-DDEBUG)
else()
  set(CMAKE_BUILD_TYPE "Release")
  add_definitions(-O3)
endif()