cmake_minimum_required(VERSION 3.14...3.21)

project(FabSoften LANGUAGES CXX)

include(cmake/utils.cmake)
include(cmake/options.cmake)

find_package(OpenCV 4 REQUIRED)

add_library(FabSoften SHARED)

add_subdirectory(src)
add_subdirectory(include)

target_include_directories(FabSoften PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_compile_features(FabSoften PRIVATE cxx_std_17)

# install
include(cmake/install_config.cmake)

# test
if(BUILD_TESTS)
    add_subdirectory(external/Catch2)
    add_subdirectory(tests)
endif()

# tool
if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()