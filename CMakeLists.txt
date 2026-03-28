cmake_minimum_required(VERSION 3.23)
project(methods_for_solving_of_slae LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WITH_TESTS "Build tests" ON)
option(WITH_BENCHMARK "Build benchmark" ON)

add_subdirectory(src)

if(WITH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(WITH_BENCHMARK)
    add_subdirectory(benchmark)
endif()
