cmake_minimum_required(VERSION 2.8)
project(PDFlow)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA REQUIRED)

include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})

set(cuda_src src/pdflow_cudalib.cu)
set(cpp_src src/scene_flow.cpp)

set(CUDA_ARCH_BIN "20 30 32 35 37 50 52 61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

include(CudaComputeTargetFlags.cmake)
append_target_arch_flags()

list(APPEND CUDA_NVCC_FLAGS "-O3;-Xcompiler;-fPIC;-std=c++11")

cuda_compile(cuda_objs ${cuda_src})

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O3 -msse2 -msse3 -Wall -std=c++11")

add_library(scene_flow SHARED ${cuda_src} ${cuda_objs} ${cpp_src})
target_link_libraries(scene_flow ${CUDA_LIBRARIES})
