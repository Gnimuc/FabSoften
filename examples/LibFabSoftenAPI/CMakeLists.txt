add_executable(LibFabSoftenAPI LibFabSoftenAPI.cpp)
target_link_libraries(LibFabSoftenAPI PRIVATE ${OpenCV_LIBS} FabSoften)
target_compile_features(LibFabSoftenAPI PRIVATE $<IF:$<PLATFORM_ID:Windows>,cxx_std_23,cxx_std_20>) # to enable `/std:c++latest` 

target_compile_options(LibFabSoftenAPI PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W3>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

set_target_properties(LibFabSoftenAPI PROPERTIES 
    VS_DEBUGGER_COMMAND_ARGUMENTS "-images_dir=${PROJECT_SOURCE_DIR}/assets -models_dir=${PROJECT_SOURCE_DIR}/models pexels-aadil-2598024.jpg shape_predictor_68_face_landmarks.dat"
)

if (WIN32)
    add_custom_command(TARGET LibFabSoftenAPI POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:LibFabSoftenAPI> $<TARGET_FILE_DIR:LibFabSoftenAPI>
        COMMAND_EXPAND_LISTS
    )
endif()