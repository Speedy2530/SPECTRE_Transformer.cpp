cmake_minimum_required(VERSION 3.10)
project(transformer_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/third-party
    ${CMAKE_SOURCE_DIR}/third-party/xtensor/include
    ${CMAKE_SOURCE_DIR}/third-party/xtensor-blas/include
    ${CMAKE_SOURCE_DIR}/incl
)

# Add the main tensor source
file(GLOB_RECURSE CORE_SRCS src/core/*.cpp)

# Add test executable
add_executable(test_tensor test/test_tensor.cpp ${CORE_SRCS})

# Link doctest (header-only, include it in test file)
