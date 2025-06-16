cmake_minimum_required(VERSION 3.10)
project(TransformerTests)

set(CMAKE_CXX_STANDARD 17)

include_directories(incl)
include_directories(incl/third_party)

add_executable(test_tensor tests/test_tensor.cpp src/core/Tensor.cpp)

