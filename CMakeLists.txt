cmake_minimum_required(VERSION 3.20)

project(
    KnightBot
    VERSION 0.7.0
    DESCRIPTION "KnightBot Chess Engine"
    LANGUAGES CXX
)

# ============================================================
# C++ SETTINGS
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ============================================================
# KNIGHTBOT EXECUTABLE
# ============================================================

add_executable(
    KnightBot

    src/main.cpp
    src/chess.cpp
    src/evaluation.cpp
    src/search.cpp
)


# ============================================================
# INCLUDE DIRECTORIES
# ============================================================

target_include_directories(
    KnightBot
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


# ============================================================
# VERSION INFORMATION
# ============================================================

target_compile_definitions(
    KnightBot
    PRIVATE
        KNIGHTBOT_VERSION="${PROJECT_VERSION}"
        KNIGHTBOT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        KNIGHTBOT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        KNIGHTBOT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)


# ============================================================
# COMPILER WARNINGS
# ============================================================

if(MSVC)

    target_compile_options(
        KnightBot
        PRIVATE
            /W4
            /permissive-
    )

else()

    target_compile_options(
        KnightBot
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

endif()


# ============================================================
# RELEASE OPTIMIZATION
# ============================================================

if(MSVC)

    target_compile_options(
        KnightBot
        PRIVATE
            $<$<CONFIG:Release>:/O2>
    )

else()

    target_compile_options(
        KnightBot
        PRIVATE
            $<$<CONFIG:Release>:-O3>
    )

endif()