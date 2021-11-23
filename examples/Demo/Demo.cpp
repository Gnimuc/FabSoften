/// \file Demo.cpp
/// \brief A demo shows how to use the high-level C++ API.
///

#include "fabsoften\Beautifier.h"
#include <iostream>
#include <opencv2/highgui.hpp>

/// \brief Command line keys for command line parsing
static constexpr auto cmdKeys =
    "{help h usage ?   |       | print this message            }"
    "{@image           |<none> | input image                   }"
    "{@landmark_model  |<none> | face landmark detection model }"
    "{images_dir       |       | search path for images        }"
    "{models_dir       |       | search path for models        }";

/// \brief Show input image
static constexpr auto imageWin = "Input Image";

/// \brief Show landmarks
static constexpr auto landmarkWin = "Landmarks";

/// \brief Show binary mask
static constexpr auto maskWin = "Mask";

/// \brief Show Canny edge detection results
static constexpr auto cannyWin = "Canny Edges";

/// \brief Show preprocessed image
static constexpr auto processedWin = "Preprocessed Image";

/// \brief C++ API demo
///
/// Usage: Demo.exe [params] image landmark_model
int main(int argc, char **argv) {
  // Handle command line arguments
  cv::CommandLineParser parser(argc, argv, cmdKeys);
  parser.about("C++ API demo");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  if (parser.has("images_dir"))
    cv::samples::addSamplesDataSearchPath(parser.get<cv::String>("images_dir"));
  if (parser.has("models_dir"))
    cv::samples::addSamplesDataSearchPath(parser.get<cv::String>("models_dir"));

  // Load image
  const auto imgArg = parser.get<cv::String>("@image");
  if (!parser.check()) {
    parser.printErrors();
    parser.printMessage();
    return -1;
  }
  // Set `required=false` to prevent `findFile` from throwing an exception.
  // Instead, we check whether the image is valid via the `empty` method.
  const auto inputImgPath =
      cv::samples::findFile(imgArg, /*required=*/false, /*silentMode=*/true);
  if (inputImgPath.empty()) {
    std::cout << "Could not find the image: " << imgArg << "\n"
              << "The image should be located in `images_dir`.\n";
    parser.printMessage();
    return -1;
  }

  // Load dlib's face landmark detection model
  const auto landmarkModelArg = parser.get<cv::String>("@landmark_model");
  if (!parser.check()) {
    parser.printErrors();
    parser.printMessage();
    return -1;
  }
  auto landmarkModelPath = cv::samples::findFile(landmarkModelArg, /*required=*/false);
  if (landmarkModelPath.empty()) {
    std::cout << "Could not find the landmark model file: " << landmarkModelArg << "\n"
              << "The model should be located in `models_dir`.\n";
    parser.printMessage();
    return -1;
  }

  // Create helper class
  fabsoften::Beautifier bf(inputImgPath, landmarkModelPath);

  // Detect and draw landmarks
  bf.createFaceLandmarkDetector();
  bf.createFace();
  bf.interpolateLandmarks();

  const auto &workImg = bf.getWorkImage();
  cv::Mat landmarkImg = workImg.clone();
  bf.drawLandmarks(landmarkImg);

  // Create a simple binary mask
  cv::Mat maskImg = cv::Mat::zeros(workImg.size(), CV_8UC1);
  bf.drawBinaryMask(maskImg);

  const auto &inputImg = bf.getInputImage();
  // Fit image to the screen and show image
  cv::namedWindow(imageWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(imageWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  const auto [x, y, resW, resH] = cv::getWindowImageRect(imageWin);
  const auto [imgW, imgH] = inputImg.size();
  const auto scaleFactor = 20;
  const auto scaledW = scaleFactor * resW / 100;
  const auto scaledH = scaleFactor * imgH * resW / (imgW * 100);
  cv::resizeWindow(imageWin, scaledW, scaledH);
  cv::imshow(imageWin, inputImg);

  // Landmarks Window
  cv::namedWindow(landmarkWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(landmarkWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(landmarkWin, scaledW, scaledH);
  cv::moveWindow(landmarkWin, scaledW, 0);
  cv::imshow(landmarkWin, landmarkImg);

  // Mask Window
  cv::namedWindow(maskWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(maskWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(maskWin, scaledW, scaledH);
  cv::moveWindow(maskWin, 2 * scaledW, 0);
  cv::Mat maskChannels[3] = {maskImg, maskImg, maskImg};
  cv::Mat maskImg3C;
  cv::merge(maskChannels, 3, maskImg3C);
  cv::addWeighted(maskImg3C, 0.35, workImg, 1, 0, maskImg3C);
  cv::imshow(maskWin, maskImg3C);

  cv::waitKey();
  cv::destroyAllWindows();

  return 0;
}
