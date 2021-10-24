# public headers
set(pub_hdrs 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fabsoften/libfabsoften.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fabsoften/platform.h
)
set_target_properties(FabSoften PROPERTIES PUBLIC_HEADER "${pub_hdrs}")

# install compilation database
set_target_properties(FabSoften PROPERTIES EXPORT_COMPILE_COMMANDS true)
set(ccmds_json ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json)
if(EXISTS ${ccmds_json})
    message(STATUS "Found CompilationDatabase File: " ${ccmds_json})
    install(FILES ${ccmds_json} DESTINATION share)
endif()

# install targets
install(TARGETS FabSoften
        EXPORT FabSoftenTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        INCLUDES DESTINATION include/fabsoften
        PUBLIC_HEADER DESTINATION include/fabsoften)

install(EXPORT FabSoftenTargets
        NAMESPACE FabSoften::
        FILE FabSoften-config.cmake
        DESTINATION lib/cmake/FabSoften)