cmake_minimum_required(VERSION 3.22)
project(round LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(DCMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -g -O0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")

set(SRC_LIST round.cu)
add_library(test SHARED ${SRC_LIST})
target_compile_features(test PRIVATE cxx_std_14)
