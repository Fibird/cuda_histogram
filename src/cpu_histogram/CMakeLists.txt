find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARIES_DIRS})

add_library(cpu_histogram SHARED cpu_histogram.cpp)

target_link_libraries(cpu_histogram ${OpenCV_LIBS})

