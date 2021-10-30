add_executable(FaceLandmarkDetection FaceLandmarkDetection.cpp)
target_link_libraries(FaceLandmarkDetection PRIVATE dlib::dlib ${OpenCV_LIBS})
target_compile_features(FaceLandmarkDetection PRIVATE cxx_std_23) # to enable `/std:c++latest` 

target_compile_options(FaceLandmarkDetection PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W3>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

set_target_properties(FaceLandmarkDetection PROPERTIES 
    VS_DEBUGGER_COMMAND_ARGUMENTS "-images_dir=${PROJECT_SOURCE_DIR}/assets -models_dir=${PROJECT_SOURCE_DIR}/models pexels-aadil-2598024.jpg shape_predictor_68_face_landmarks.dat"
)

