add_executable(bench_moments bench_moments.cpp)

target_link_libraries(bench_moments PRIVATE shadows_core)
target_link_libraries(bench_moments PRIVATE utils_core)

add_executable(bench_QFI bench_QFI.cpp)

target_link_libraries(bench_QFI PRIVATE shadows_core)
target_link_libraries(bench_QFI PRIVATE utils_core)
