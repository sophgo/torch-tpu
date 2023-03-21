#include_directories({PROJECT_SOURCE_DIR}/third_party/flatbuffers/include)
#link_directories({PROJECT_SOURCE_DIR}/third_party/flatbuffers/lib)
include_directories(${FLATBUFFERS_PATH}/include)
link_directories(${FLATBUFFERS_PATH}/lib)

add_subdirectory(include)
add_subdirectory(src)
