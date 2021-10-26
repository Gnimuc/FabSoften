add_executable(SanityCheck sanity_check.cpp sanity_check.h)
target_link_libraries(SanityCheck PRIVATE FabSoften Catch2::Catch2WithMain)

include(CTest)
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(SanityCheck)