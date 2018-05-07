project(histogram_test)
cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(MAIN_PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)
message(${MAIN_PROJECT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${MAIN_PROJECT_SOURCE_DIR}/bin)

set(CPU_HISTOGRAM_INCLUDE_DIRS ${MAIN_PROJECT_SOURCE_DIR}/src/cpu_histogram)
set(GPU_HISTOGRAM_INCLUDE_DIRS ${MAIN_PROJECT_SOURCE_DIR}/src/gpu_histogram)
include_directories(
${OpenCV_INCLUDE_DIRS} 
${CPU_HISTOGRAM_INCLUDE_DIRS}
${GPU_HISTOGRAM_INCLUDE_DIRS}
)

add_executable(cpu_hist_equal_test cpu_hist_equal_test.cpp)
add_executable(cpu_hist_match_test cpu_hist_match_test.cpp)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
cuda_add_executable(gpu_hist_equal_test gpu_hist_equal_test.cu)

set(CPU_HISTOGRAM_LIBRARIES_DIRS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/cpu_histogram)
set(GPU_HISTOGRAM_LIBRARIES_DIRS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/gpu_histogram)
set(CPU_HISTOGRAM_LIBS ${CPU_HISTOGRAM_LIBRARIES_DIRS}/libcpu_histogram.so)
set(GPU_HISTOGRAM_LIBS ${GPU_HISTOGRAM_LIBRARIES_DIRS}/libgpu_histogram.a)

target_link_libraries(cpu_hist_equal_test ${OpenCV_LIBS} ${CPU_HISTOGRAM_LIBS})
target_link_libraries(cpu_hist_match_test ${OpenCV_LIBS} ${CPU_HISTOGRAM_LIBS})
target_link_libraries(gpu_hist_equal_test ${OpenCV_LIBS} ${GPU_HISTOGRAM_LIBS})
