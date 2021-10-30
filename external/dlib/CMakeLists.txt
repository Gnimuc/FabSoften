include(FetchContent)

FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.22
)

message(STATUS "Checking dlib...")

FetchContent_MakeAvailable(dlib)