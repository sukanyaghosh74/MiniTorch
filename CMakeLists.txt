cmake_minimum_required(VERSION 3.10)
project(minitorch)
set(CMAKE_CXX_STANDARD 14)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/cpp/include)

add_library(minitorch SHARED cpp/src/tensor.cpp)
add_library(minitorch_bindings MODULE cpp/src/bindings.cpp)

target_link_libraries(minitorch_bindings PRIVATE minitorch pybind11::module)
set_target_properties(minitorch_bindings PROPERTIES OUTPUT_NAME "_C" PREFIX "")
