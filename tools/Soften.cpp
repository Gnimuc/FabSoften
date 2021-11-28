/// \file Soften.cpp
/// \brief A handy command line tool.
///
/// Soften.exe [params] image landmark_model

#include "fabsoften\LibFabSoften.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>

/// \brief Command line keys for command line parsing
static constexpr auto cmdKeys =
    "{help h usage ?   |       | print this message            }"
    "{@image           |<none> | input image                   }"
    "{@landmark_model  |<none> | face landmark detection model }"
    "{images_dir       |       | search path for images        }"
    "{models_dir       |       | search path for models        }"
    "{output           |a.jpg  | output path                   }";

int main(int argc, char **argv) {
  // Handle command line arguments
  cv::CommandLineParser parser(argc, argv, cmdKeys);
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

  cv::imwrite(parser.get<cv::String>("output"), outputImg);

  return 0;
}
