add_library(cnpy STATIC cnpy.cpp)
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(ZLIB REQUIRED)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

set_target_properties(cnpy PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(cnpy PRIVATE -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-conversion -Wno-error)
