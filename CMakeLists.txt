cmake_minimum_required(VERSION 3.16)

project(KnightBot VERSION 0.5.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(KnightBot
    src/main.cpp
    src/chess.cpp
    src/evaluation.cpp
    src/search.cpp
)

target_include_directories(KnightBot PRIVATE src)

if (MSVC)
    target_compile_options(KnightBot PRIVATE /W4)
else()
    target_compile_options(KnightBot PRIVATE -Wall -Wextra -Wpedantic)
endif()