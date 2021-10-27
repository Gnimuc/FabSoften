add_executable(SanityCheck SanityCheck.cpp SanityCheck.h)
target_link_libraries(SanityCheck PRIVATE FabSoften Catch2::Catch2WithMain)

add_custom_command(TARGET SanityCheck POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:SanityCheck> $<TARGET_FILE_DIR:SanityCheck>
    COMMAND_EXPAND_LISTS
)

include(CTest)
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(SanityCheck)