cmake_minimum_required(VERSION 3.18)
project(localize CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Release build with full optimisation by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Jetson Orin Nano — Ampere SM87
set(CMAKE_CUDA_ARCHITECTURES 87)

# OpenCV — system-installed on Jetson (JetPack)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# CUDA / cuBLAS
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA version:   ${CUDAToolkit_VERSION}")

# PopSift — GPU SIFT (built from source, installed locally)
list(APPEND CMAKE_PREFIX_PATH "/home/scott/claude-project/popsift/install")
find_package(PopSift REQUIRED)
message(STATUS "PopSift found: ${PopSift_VERSION}")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(localize localize.cpp cuda_bf_matcher.cu popsift_sift.cpp)
set_target_properties(localize PROPERTIES
    BUILD_RPATH "/home/scott/claude-project/popsift/install/lib"
    INSTALL_RPATH "/home/scott/claude-project/popsift/install/lib")
target_include_directories(localize PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(localize PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cublas
    PopSift::popsift)
