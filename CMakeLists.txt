cmake_minimum_required(VERSION 3.10)

# Set the macOS SDK for Apple platforms.
if(APPLE)
    set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)
endif()

project(MLForge VERSION 1.0.0 LANGUAGES CXX)

# Option to build a shared library (ON) or a static library (OFF).
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Set the C++ standard and required features.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position-independent code for static libraries if needed.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set common compiler warnings.
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

# Define project directories.
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})
set(INCLUDE_DIR ${PROJECT_ROOT}/include)
set(SRC_DIR ${PROJECT_ROOT}/src)
set(TEST_DIR ${PROJECT_ROOT}/tests)
set(EXAMPLES_DIR ${PROJECT_ROOT}/examples)
set(BENCHMARKS_DIR ${PROJECT_ROOT}/benchmarks)
set(PROFILING_DIR ${PROJECT_ROOT}/profiling)
set(THIRD_PARTY_DIR ${PROJECT_ROOT}/third_party)

# Add third-party include directories.
include_directories(${THIRD_PARTY_DIR}/cpp-httplib)

# Optionally generate compile_commands.json for clang tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Collect source files for the core library.
# -----------------------------------------------------------------------------
file(GLOB_RECURSE MLFORGE_SOURCES "${SRC_DIR}/*.cpp")

# Create the main MLForge library target.
add_library(MLForgeLib ${MLFORGE_SOURCES})
target_include_directories(MLForgeLib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add specific compile definitions or options for MLForgeLib:
target_compile_definitions(MLForgeLib PRIVATE $<$<CONFIG:Debug>:DEBUG_ENABLED>)

# -----------------------------------------------------------------------------
# Executable Targets for Examples.
# -----------------------------------------------------------------------------
file(GLOB_RECURSE EXAMPLE_SOURCES "${EXAMPLES_DIR}/*.cpp")
foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    add_executable(${example_name} ${example_src})
    target_link_libraries(${example_name} PRIVATE MLForgeLib)
    target_include_directories(${example_name} PRIVATE ${INCLUDE_DIR})
endforeach()

# -----------------------------------------------------------------------------
# Executable Targets for Tests.
# -----------------------------------------------------------------------------
enable_testing()  # Enable CTest.
file(GLOB_RECURSE TEST_SOURCES "${TEST_DIR}/*.cpp")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE MLForgeLib)
    target_include_directories(${test_name} PRIVATE ${INCLUDE_DIR})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# -----------------------------------------------------------------------------
# Executable Targets for Benchmarks.
# -----------------------------------------------------------------------------
file(GLOB_RECURSE BENCHMARK_SOURCES "${BENCHMARKS_DIR}/*.cpp")
foreach(bench_src ${BENCHMARK_SOURCES})
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} PRIVATE MLForgeLib)
endforeach()

# -----------------------------------------------------------------------------
# Executable Targets for Profiling.
# -----------------------------------------------------------------------------
file(GLOB_RECURSE PROFILING_SOURCES "${PROFILING_DIR}/*.cpp")
foreach(profile_src ${PROFILING_SOURCES})
    get_filename_component(profile_name ${profile_src} NAME_WE)
    add_executable(${profile_name} ${profile_src})
    target_link_libraries(${profile_name} PRIVATE MLForgeLib)
endforeach()

# -----------------------------------------------------------------------------
# Installation Rules (optional)
# -----------------------------------------------------------------------------
install(TARGETS MLForgeLib
    EXPORT MLForgeTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY ${INCLUDE_DIR}/ml/ DESTINATION include/ml)

# -----------------------------------------------------------------------------
# Export the project for use with CMake's find_package.
# -----------------------------------------------------------------------------
export(EXPORT MLForgeTargets FILE "${PROJECT_BINARY_DIR}/MLForgeTargets.cmake")
install(EXPORT MLForgeTargets FILE MLForgeTargets.cmake DESTINATION lib/cmake/MLForge)

# -----------------------------------------------------------------------------
# Additional Options / Custom Targets.
# -----------------------------------------------------------------------------
# Add targets for documentation generation (e.g., using Doxygen) if desired.
# Add custom target for packaging, code formatting, etc.
