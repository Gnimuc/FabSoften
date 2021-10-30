include(FetchContent)

FetchContent_Declare(
    tinyspline
    GIT_REPOSITORY https://github.com/msteinbeck/tinyspline.git
    GIT_TAG        v0.3.0
)

message(STATUS "Checking tinyspline...")

FetchContent_MakeAvailable(tinyspline)