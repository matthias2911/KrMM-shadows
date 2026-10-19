# shadows_core
add_library(shadows_core STATIC 
    shadows.cpp
    metrics.cpp
    moments/running/MatMat.cpp
    moments/batched/MatMat.cpp
    QFI/running/MatMat.cpp
    QFI/batched/MatMat.cpp
)

target_include_directories(shadows_core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(shadows_core PRIVATE 
    BLAS::BLAS
)

target_link_libraries(shadows_core PRIVATE
    OpenMP::OpenMP_CXX 
) 

target_link_libraries(shadows_core PUBLIC 
    MPI::MPI_CXX
)

target_sources(utils_core PRIVATE
    utils.cpp
)
