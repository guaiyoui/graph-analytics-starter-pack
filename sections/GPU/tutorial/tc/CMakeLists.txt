cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca")
# set nvcc path and architecture
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)

project(tc LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3") # Added -O3 optimization for CUDA./



find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Fetch additional dependencies
include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

add_subdirectory(src)

add_executable(tc main.cu)

target_link_libraries(tc src argparse)

set_target_properties(tc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )

