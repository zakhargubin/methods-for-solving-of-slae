cmake_minimum_required(VERSION 3.10)
project(solution LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(solution main.cpp)
target_include_directories(solution PRIVATE ${CMAKE_SOURCE_DIR})
