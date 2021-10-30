cmake_minimum_required(VERSION 3.21)

project(FabSoften LANGUAGES CXX)

# Modules
include(cmake/utils.cmake)
include(cmake/options.cmake)

# External Dependencies
add_subdirectory(external)
add_subdirectory(external/dlib)
add_subdirectory(external/tinyspline)
find_package(OpenCV 4 REQUIRED)

# Core Library
add_library(FabSoften SHARED)

add_subdirectory(src)
add_subdirectory(include)

target_include_directories(FabSoften PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

target_compile_features(FabSoften PRIVATE cxx_std_20)

set_property(TARGET FabSoften PROPERTY ASSETS_DIR ${PROJECT_SOURCE_DIR}/assets)
set_property(TARGET FabSoften PROPERTY MODELS_DIR ${PROJECT_SOURCE_DIR}/models)

include(cmake/install_config.cmake)

if(FABSOFTEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(external/Catch2)
    add_subdirectory(tests)
endif()

if(FABSOFTEN_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(FABSOFTEN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (FABSOFTEN_BUILD_DOCS)
    add_subdirectory(docs)
endif()

