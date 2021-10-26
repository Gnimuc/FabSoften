cmake_minimum_required(VERSION 3.19...3.21)

project(FabSoften LANGUAGES CXX)

include(cmake/utils.cmake)
include(cmake/options.cmake)

# fetch dependencies
add_subdirectory(external)
# dlib
add_subdirectory(external/dlib)

# OpenCV
find_package(OpenCV 4 REQUIRED)

# config library
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
if(FABSOFTEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(external/Catch2)    
    add_subdirectory(tests)
endif()

# tool
if(FABSOFTEN_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# examples
if(FABSOFTEN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()