/// \file LibFabSoftenAPI.cpp
/// \brief A demo shows how to use the C API.
///

#include "fabsoften\LibFabSoften.h"
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

/// \brief Show preprocessed image
static constexpr auto processedWin = "Preprocessed Image";

/// \brief C API demo
///
/// Usage: LibFabSoftenAPI.exe [params] image landmark_model
int main(int argc, char **argv) {
  // Handle command line arguments
  cv::CommandLineParser parser(argc, argv, cmdKeys);
  parser.about("C API demo");
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

  // C API
  fabsoften_err err = fabsoften_success;
  fabsoften_context ctx =
      fabsoften_create_context(inputImgPath.c_str(), landmarkModelPath.c_str(), &err);
  assert(*err == fabsoften_success);

  fabsoften_beautify(ctx);

  fabsoften_encode(ctx);

  size_t n = fabsoften_get_buffer_size(ctx);

  std::vector<uchar> img(n);

  fabsoften_get_data(ctx, img.data());

  fabsoften_dispose(ctx);

  cv::Mat outputImg = cv::imdecode(img, cv::IMREAD_COLOR);

  const auto inputImg = cv::imread(inputImgPath);

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

  // Processed Window
  cv::namedWindow(processedWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(processedWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(processedWin, scaledW, scaledH);
  cv::moveWindow(processedWin, scaledW, 0);
  cv::imshow(processedWin, outputImg);

  cv::waitKey();
  cv::destroyAllWindows();

  return 0;
}
