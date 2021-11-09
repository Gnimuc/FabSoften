add_executable(SkinMapGeneration SkinMapGeneration.cpp)
target_link_libraries(SkinMapGeneration PRIVATE dlib::dlib ${OpenCV_LIBS} tinysplinecxx)
target_compile_features(SkinMapGeneration PRIVATE $<IF:$<PLATFORM_ID:Windows>,cxx_std_23,cxx_std_20>) # to enable `/std:c++latest` 

target_compile_options(SkinMapGeneration PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W3>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

set_target_properties(SkinMapGeneration PROPERTIES 
    VS_DEBUGGER_COMMAND_ARGUMENTS "-images_dir=${PROJECT_SOURCE_DIR}/assets -models_dir=${PROJECT_SOURCE_DIR}/models pexels-aadil-2598024.jpg shape_predictor_68_face_landmarks.dat"
)

