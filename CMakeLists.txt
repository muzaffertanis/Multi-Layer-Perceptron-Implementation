cmake_minimum_required(VERSION 3.20)
project(mlp_implementation)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/mlp/*)

add_executable(mlp_implementation main.cpp ${SRC_FILES})

target_link_libraries(mlp_implementation ${OpenCV_LIBS})