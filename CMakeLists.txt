cmake_minimum_required(VERSION 3.20)
project(shadows-cpp 
  LANGUAGES CXX
  HOMEPAGE_URL "https://github.com/matthias2911/KrMM-shadows.git"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------
# remove environment/toolchain injected flags
# ------------------------------------------------------------
set(CMAKE_CXX_FLAGS "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_TESTING "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_BENCHMARK "" CACHE STRING "" FORCE)

# ------------------------------------------------------------
# Custom build types
# ------------------------------------------------------------
set(allowedBuildTypes Testing Benchmark)
set(defaultBuildType Testing)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, defaulting to '${defaultBuildType}'")
  set(CMAKE_BUILD_TYPE "${defaultBuildType}" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CONFIGURATION_TYPES "${allowedBuildTypes}" CACHE STRING "" FORCE)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_BENCHMARK TRUE) # applies -flto for Benchmark
# ------------------------------------------------------------

# ------------------------------------------------------------
# global compiler flags
# ------------------------------------------------------------
# avx: -mavx2 -mfma
# sse: -msse4.2
# -DNDEBUG: careful, deactives assert statements
# -Wall;-Werror;-Wextra;-Wpedantic;-Wshadow;-Wconversion
add_compile_options(
    "$<$<CONFIG:Testing>:-DDEBUG;-Og;-g;-march=native;-fsanitize=address;-fsanitize=undefined>"
    "$<$<CONFIG:Benchmark>:-DBENCH;-O3;-march=native;-ffast-math;-funroll-loops;-ffp-contract=fast>"
)

add_link_options(
    "$<$<CONFIG:Testing>:-fsanitize=address;-fsanitize=undefined>"
)

# ------------------------------------------------------------


add_subdirectory(src)
add_subdirectory(external/cnpy)

add_subdirectory(experiments/main)
