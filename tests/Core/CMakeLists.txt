add_executable(Core_UNITTEST Core.cpp Core.h)
target_link_libraries(Core_UNITTEST PRIVATE FabSoften Catch2::Catch2WithMain ${OpenCV_LIBS} dlib::dlib)

target_compile_definitions(Core_UNITTEST PRIVATE UNITTEST_PROJECT_DIR="${PROJECT_SOURCE_DIR}")
target_compile_features(Core_UNITTEST PRIVATE cxx_std_17)

if (WIN32)
    add_custom_command(TARGET Core_UNITTEST POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:Core_UNITTEST> $<TARGET_FILE_DIR:Core_UNITTEST>
        COMMAND_EXPAND_LISTS
    )
endif()

include(CTest)
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(Core_UNITTEST)