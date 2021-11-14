add_executable(ADF_UNITTEST ADF.cpp ADF.h)
target_link_libraries(ADF_UNITTEST PRIVATE FabSoften Catch2::Catch2WithMain ${OpenCV_LIBS})

if (WIN32)
    add_custom_command(TARGET ADF_UNITTEST POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:ADF_UNITTEST> $<TARGET_FILE_DIR:ADF_UNITTEST>
        COMMAND_EXPAND_LISTS
    )
endif()

include(CTest)
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
catch_discover_tests(ADF_UNITTEST)