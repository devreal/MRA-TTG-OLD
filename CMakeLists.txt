cmake_minimum_required(VERSION 3.19)
project(MRA-TTG LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules/")

set(BUILD_TESTING OFF CACHE BOOL "Whether to build tests in this project and its subprojects")

# needed for atomicAdd_block
set(CMAKE_CUDA_ARCHITECTURES 70)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif(CMAKE_CUDA_COMPILER)
set(HAVE_CUDA ${CMAKE_CUDA_COMPILER} CACHE BOOL "True if we can compile .cu files")

include(FindOrFetchTTG)

find_package(OpenMP)

add_subdirectory(mra)
