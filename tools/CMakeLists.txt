add_executable(Soften Soften.cpp)
target_link_libraries(Soften PRIVATE ${OpenCV_LIBS} FabSoften dlib::dlib tinysplinecxx)

target_compile_options(Soften PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W3>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

set_target_properties(Soften PROPERTIES 
    VS_DEBUGGER_COMMAND_ARGUMENTS "-images_dir=${PROJECT_SOURCE_DIR}/assets -models_dir=${PROJECT_SOURCE_DIR}/models pexels-aadil-2598024.jpg shape_predictor_68_face_landmarks.dat"
)

if (WIN32)
    add_custom_command(TARGET Soften POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:Soften> $<TARGET_FILE_DIR:Soften>
        COMMAND_EXPAND_LISTS
    )
endif()

# install(TARGETS Soften RUNTIME DESTINATION bin)