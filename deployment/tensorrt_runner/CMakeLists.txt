cmake_minimum_required(VERSION 3.18)
project(care_asd_trt_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include /usr/include/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/libraries not found. Install JetPack TensorRT packages first.")
endif()

add_executable(care_asd_trt_runner src/main.cpp)
target_include_directories(care_asd_trt_runner PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(care_asd_trt_runner PRIVATE ${TENSORRT_LIBRARY} CUDA::cudart)
target_compile_options(care_asd_trt_runner PRIVATE -Wall -Wextra -Wpedantic)
