include(FetchContent)

FetchContent_Declare(
    tinyspline
    GIT_REPOSITORY https://github.com/msteinbeck/tinyspline.git
    GIT_TAG        v0.3.0
)

message(STATUS "Checking tinyspline...")

option(TINYSPLINE_BUILD_EXAMPLES "Build TinySpline examples." ON)
option(TINYSPLINE_BUILD_TESTS "Build TinySpline tests." ON)
option(TINYSPLINE_BUILD_DOCS "Build TinySpline documentation." ON)

set(TINYSPLINE_BUILD_EXAMPLES OFF)
set(TINYSPLINE_BUILD_TESTS OFF)
set(TINYSPLINE_BUILD_DOCS OFF)

FetchContent_MakeAvailable(tinyspline)

