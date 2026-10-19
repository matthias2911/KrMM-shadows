# utils_core
add_library(utils_core STATIC 
    utils.cpp
)

target_include_directories(utils_core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(utils_core PUBLIC
    cnpy
    stdc++fs
) 

add_subdirectory(shadows)
