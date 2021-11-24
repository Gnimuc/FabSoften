add_executable(Demo Demo.cpp)
target_link_libraries(Demo PRIVATE ${OpenCV_LIBS} FabSoften dlib::dlib tinysplinecxx)
target_compile_features(Demo PRIVATE $<IF:$<PLATFORM_ID:Windows>,cxx_std_23,cxx_std_20>) # to enable `/std:c++latest` 

target_compile_options(Demo PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W3>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

set_target_properties(Demo PROPERTIES 
    VS_DEBUGGER_COMMAND_ARGUMENTS "-images_dir=${PROJECT_SOURCE_DIR}/assets -models_dir=${PROJECT_SOURCE_DIR}/models pexels-aadil-2598024.jpg shape_predictor_68_face_landmarks.dat"
)

if (WIN32)
    add_custom_command(TARGET Demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:Demo> $<TARGET_FILE_DIR:Demo>
        COMMAND_EXPAND_LISTS
    )
endif()