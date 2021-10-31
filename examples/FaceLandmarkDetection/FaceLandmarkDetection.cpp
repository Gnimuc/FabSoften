/**
 * @file FaceLandmarkDetection.cpp
 * @brief An example of how to do face landmark detection with OpenCV and dlib.
 * 
 */

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>

/// @brief Command line keys for command line parsing
static constexpr auto cmdKeys =
    "{help h usage ?   |       | print this message            }"
    "{@image           |<none> | input image                   }"
    "{@landmark_model  |<none> | face landmark detection model }"
    "{images_dir       |       | search path for images        }"
    "{models_dir       |       | search path for models        }";

/// @brief Face Landmark Detection Window
static constexpr auto landmarkWin = "FaceLandmarkDetection";

/// @brief Face landmark detection example
///
/// Usage: FaceLandmarkDetection.exe [params] image landmark_model
int main(int argc, char **argv) {
  // Handle command line arguments
  cv::CommandLineParser parser(argc, argv, cmdKeys);

  parser.about("Face landmark detection example");

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  if (parser.has("images_dir"))
    cv::samples::addSamplesDataSearchPath(parser.get<cv::String>("images_dir"));

  if (parser.has("models_dir"))
    cv::samples::addSamplesDataSearchPath(parser.get<cv::String>("models_dir"));

  // Load image
  auto imgArg = parser.get<cv::String>("@image");
  if (!parser.check()) {
    parser.printErrors();
    parser.printMessage();
    return -1;
  }
  // Set `required=false` to prevent `findFile` from throwing an exception.
  // Instead, we check whether the image is valid via the `empty` method.
  auto inputImg =
      cv::imread(cv::samples::findFile(imgArg, /*required=*/false, /*silentMode=*/true));
  if (inputImg.empty()) {
    std::cout << "Could not open or find the image: " << imgArg << "\n"
              << "The image should be located in `images_dir`.\n"
              << std::endl;
    parser.printMessage();
    return -1;
  }

  // Load dlib's face landmark detection model
  auto landmarkModelArg = parser.get<cv::String>("@landmark_model");
  if (!parser.check()) {
    parser.printErrors();
    parser.printMessage();
    return -1;
  }
  auto landmarkModelPath = cv::samples::findFile(landmarkModelArg, /*required=*/false);
  if (landmarkModelPath.empty()) {
    std::cout << "Could not find the landmark model file: " << landmarkModelArg << "\n"
              << "The model should be located in `models_dir`.\n"
              << std::endl;
    parser.printMessage();
    return -1;
  }
  
  dlib::shape_predictor landmarkDetector;
  dlib::deserialize(landmarkModelPath) >> landmarkDetector;

  // Detect faces
  // Need to use `dlib::cv_image` to bridge OpenCV and dlib.
  const auto dlibImg = dlib::cv_image<dlib::bgr_pixel>(inputImg);
  auto faceDetector = dlib::get_frontal_face_detector();
  auto faces = faceDetector(dlibImg);  // FIXME: see issue #1

  // Detect and draw landmarks
  const auto detect = [&](const auto &face) { return landmarkDetector(dlibImg, face); };
  for (const auto &shape : faces | std::views::transform(detect)) {
    for (unsigned long i : std::views::iota(0) | std::views::take(shape.num_parts())) {
      const auto point = shape.part(i);
      const auto center = cv::Point(point.x(), point.y());
      constexpr auto radius = 15;
      const auto color = cv::Scalar(0, 255, 255);
      constexpr auto thickness = 5;
      cv::circle(inputImg, center, radius, color, thickness);
    }
  }

  // Fit image to the screen and show image
  cv::namedWindow(landmarkWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(landmarkWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  auto [x, y, resW, resH] = cv::getWindowImageRect(landmarkWin);
  auto [imgW, imgH] = inputImg.size();
  const auto scale = 40;
  cv::resizeWindow(landmarkWin, scale * resW / 100, scale * imgH * resW / (imgW * 100));
  cv::imshow(landmarkWin, inputImg);
  cv::waitKey();
  cv::destroyAllWindows();

  return 0;
}
