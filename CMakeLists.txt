project(histogram)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARIES_DIRS})

add_executable(hist_equal src/cpu_hist_equal.cpp)
add_executable(hist_match src/cpu_hist_match.cpp)

target_link_libraries(hist_equal ${OpenCV_LIBS})
target_link_libraries(hist_match ${OpenCV_LIBS})
