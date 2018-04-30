project(histogram_test)
cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)

set(MAIN_PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)
message(${MAIN_PROJECT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${MAIN_PROJECT_SOURCE_DIR}/bin)

set(CPU_HISTOGRAM_INCLUDE_DIRS ${MAIN_PROJECT_SOURCE_DIR}/src/cpu_histogram)
include_directories(
${OpenCV_INCLUDE_DIRS} 
${CPU_HISTOGRAM_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARIES_DIRS})

add_executable(cpu_hist_equal_test cpu_hist_equal_test.cpp)
add_executable(cpu_hist_match_test cpu_hist_match_test.cpp)

set(CPU_HISTOGRAM_LIBRARIES_DIRS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib)
set(CPU_HISTOGRAM_LIBS ${CPU_HISTOGRAM_LIBRARIES_DIRS}/libcpu_histogram.so)
target_link_libraries(cpu_hist_equal_test ${OpenCV_LIBS} ${CPU_HISTOGRAM_LIBS})
target_link_libraries(cpu_hist_match_test ${OpenCV_LIBS} ${CPU_HISTOGRAM_LIBS})
