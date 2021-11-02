/**
 * @file SpotConcealment.cpp
 * @brief An example demonstrates how to detect and conceal large blemishes with Canny Edge
 * detector.
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
#include <tinysplinecxx.h>

/// @brief Command line keys for command line parsing
static constexpr auto cmdKeys =
    "{help h usage ?   |       | print this message            }"
    "{@image           |<none> | input image                   }"
    "{@landmark_model  |<none> | face landmark detection model }"
    "{images_dir       |       | search path for images        }"
    "{models_dir       |       | search path for models        }";

/// @brief Face Landmark Detection Window
static constexpr auto landmarkWin = "FaceLandmarkDetection";

/// @brief Binary Skin Mask Window
static constexpr auto skinMaskWin = "BinarySkinMask";

/// @brief Canny Edge Detection Window
static constexpr auto edgeWin = "CannyEdgeDetection";

/// @brief Bianry skin mask example
///
/// Usage: SpotConcealment.exe [params] image landmark_model
int main(int argc, char **argv) {
  // Handle command line arguments
  cv::CommandLineParser parser(argc, argv, cmdKeys);
  parser.about("Bianry skin mask example");
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

  // Leave the original image untouched
  cv::Mat workImg = inputImg.clone();
  //cv::pyrDown(inputImg, workImg);
  //cv::pyrDown(workImg, workImg);
  //cv::pyrDown(workImg, workImg);
  // Make a copy for drawing landmarks
  cv::Mat landmarkImg = workImg.clone();
  // Make a copy for drawing binary mask
  cv::Mat maskImg = cv::Mat::zeros(workImg.size(), CV_8UC1);

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
  const auto dlibImg = dlib::cv_image<dlib::bgr_pixel>(workImg);
  auto faceDetector = dlib::get_frontal_face_detector();
  auto faces = faceDetector(dlibImg);

  // Draw landmark on the input image
  const auto drawLandmark = [&](const auto x, const auto y) {
    const auto radius = workImg.rows / 300;
    const auto color = cv::Scalar(0, 255, 255);
    const auto thickness = radius / 2;
    const auto center = cv::Point(x, y);
    cv::circle(landmarkImg, center, radius, color, thickness);
  };

  // clang-format off
  // Get outer contour of facial features
  // The 68 facial landmark from the iBUG 300-W dataset(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
  // Jaw:              0-16 (lower face boundary)
  // Right eyebrow:   17-21
  // Left eyebrow:    22-26 
  // Nose:            27-35
  // Right eye:       36-41
  // Left eye:        42-47
  // Mouth:           48-67 (boundary:48-59)
  // clang-format on
  const auto jaw = [](const auto i) { return i >= 0 && i <= 16; };
  const auto rightEye = [](const auto i) { return i >= 36 && i <= 41; };
  const auto leftEye = [](const auto i) { return i >= 42 && i <= 47; };
  const auto mouthBoundary = [](const auto i) { return i >= 48 && i <= 59; };

  const auto detect = [&](const auto &face) { return landmarkDetector(dlibImg, face); };
  for (const auto &shape : faces | std::views::transform(detect)) {
    std::vector<tinyspline::real> knots;
    // Join the landmark points on the boundary of facial features using cubic curve
    const auto getCurve = [&]<typename T>(T predicate, const auto n) {
      for (auto i :
           std::views::iota(0) | std::views::filter(predicate) | std::views::take(n)) {
        const auto point = shape.part(i);
        knots.push_back(point.x());
        knots.push_back(point.y());
      }
      // Make a closed curve
      knots.push_back(knots[0]);
      knots.push_back(knots[1]);
      // Interpolate the curve
      auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);
      return spline;
    };

    // Right eye cubic curve
    const auto rightEyeCurve = getCurve(rightEye, 6);
    // Sample landmark points from the curve
    constexpr auto eyePointNum = 25;
    std::array<cv::Point, eyePointNum> rightEyePts;
    for (const auto i : std::views::iota(0, eyePointNum)) {
      auto net = rightEyeCurve(1.0 / eyePointNum * i);
      auto result = net.result();
      auto x = result[0], y = result[1];
      drawLandmark(x, y);
      rightEyePts[i] = cv::Point(x, y);
    }
    // Draw binary mask
    cv::fillConvexPoly(maskImg, rightEyePts, cv::Scalar(255), cv::LINE_AA);
    knots.clear();

    // Left eye cubic curve
    const auto leftEyeCurve = getCurve(leftEye, 6);
    std::array<cv::Point, eyePointNum> leftEyePts;
    // Sample landmark points from the curve
    for (const auto i : std::views::iota(0, eyePointNum)) {
      auto net = leftEyeCurve(1.0 / eyePointNum * i);
      auto result = net.result();
      auto x = result[0], y = result[1];
      drawLandmark(x, y);
      leftEyePts[i] = cv::Point(x, y);
    }
    // Draw binary mask
    cv::fillConvexPoly(maskImg, leftEyePts, cv::Scalar(255), cv::LINE_AA);
    knots.clear();

    // Mouth cubic curve
    const auto mouthCurve = getCurve(mouthBoundary, 12);
    constexpr auto mouthPointNum = 40;
    std::array<cv::Point, mouthPointNum> mouthPts;
    // Sample landmark points from the curve
    for (const auto i : std::views::iota(0, mouthPointNum)) {
      auto net = mouthCurve(1.0 / mouthPointNum * i);
      auto result = net.result();
      auto x = result[0], y = result[1];
      drawLandmark(x, y);
      mouthPts[i] = cv::Point(x, y);
    }
    // Draw binary mask
    cv::fillPoly(maskImg, mouthPts, cv::Scalar(255), cv::LINE_AA);

    // Estimate an ellipse that can complete the upper face region
    std::vector<cv::Point> lowerFacePts;
    for (auto i : std::views::iota(0) | std::views::filter(jaw) | std::views::take(17)) {
      const auto point = shape.part(i);
      auto x = point.x(), y = point.y();
      drawLandmark(x, y);
      lowerFacePts.push_back(cv::Point(x, y));
    }
    // Guess a point located in the upper face region
    // Pb: 8 (bottom of jaw)
    // Pt: 27 (top of nose
    const auto Pb = shape.part(8);
    const auto Pt = shape.part(27);
    auto x = Pb.x();
    auto y = Pt.y() - 0.85 * abs(Pb.y() - Pt.y());
    drawLandmark(x, y);
    lowerFacePts.push_back(cv::Point(x, y));
    // Fit ellipse
    auto box = cv::fitEllipseDirect(lowerFacePts);
    cv::Mat maskTmp = cv::Mat(maskImg.size(), CV_8UC1, cv::Scalar(255));
    cv::ellipse(maskTmp, box, cv::Scalar(0), -1, cv::FILLED);
    cv::bitwise_or(maskTmp, maskImg, maskImg);
    cv::bitwise_not(maskImg, maskImg);
  }

  // Spot Concealment
  // Convert the RGB image to an intensity one
  cv::Mat grayImg;
  cv::cvtColor(workImg, grayImg, cv::COLOR_BGR2GRAY);

  // Compute the DoG to detect edges
  cv::Mat blurImg1, blurImg2, dogImg;
  const auto sigmaY = grayImg.cols / 200.0;
  const auto sigmaX = grayImg.rows / 200.0;
  cv::GaussianBlur(grayImg, blurImg1, cv::Size(3, 3), 0);
  cv::GaussianBlur(grayImg, blurImg2, cv::Size(0, 0), sigmaX, sigmaY);
  cv::absdiff(blurImg1, blurImg2, dogImg); 
  
  // Apply binary mask to the image
  cv::bitwise_and(maskImg, dogImg, dogImg);
  constexpr auto rejectT = 17.5;
  cv::threshold(dogImg, dogImg, rejectT, 255, cv::THRESH_BINARY);

  // Apply Canny Edge Detection
  cv::Mat edgeImg;
  cv::GaussianBlur(dogImg, dogImg, cv::Size(0, 0), 1);
  cv::Canny(dogImg, edgeImg, 10, 150); 
  

  //cv::normalize(dogImg, dogImg, 0, 255, cv::NORM_MINMAX);
  //cv::threshold(dogImg, dogImg, 30, 0, cv::THRESH_TOZERO);
  // cv::bitwise_not(edgeImg, edgeImg);
  // cv::bitwise_and(edgeImg, blurImg, blurImg);
  //const auto upperT = cv::threshold(dogImg, edgeImg, 10, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
  //const auto lowT = 0.5 * upperT;
  // std::cout << lowT << "|" << upperT << std::endl; 
  // 
  //try {
  //  cv::GaussianBlur(cannyImg, cannyImg, cv::Size(0, 0), 1);
  //  cv::Canny(cannyImg, edgeImg, 0.3, 0.5);
  //} catch (cv::Exception &e) {
  //  const char *err_msg = e.what();
  //  std::cout << "exception caught: " << err_msg << std::endl;
  //}

  // Fit image to the screen and show image
  cv::namedWindow(landmarkWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(landmarkWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  auto [x, y, resW, resH] = cv::getWindowImageRect(landmarkWin);
  auto [imgW, imgH] = landmarkImg.size();
  const auto scaleFactor = 30;
  const auto scaledW = scaleFactor * resW / 100;
  const auto scaledH = scaleFactor * imgH * resW / (imgW * 100);
  cv::resizeWindow(landmarkWin, scaledW, scaledH);
  cv::imshow(landmarkWin, landmarkImg);

  // Show binary skin mask
  cv::namedWindow(skinMaskWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(skinMaskWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(skinMaskWin, scaledW, scaledH);
  cv::moveWindow(skinMaskWin, scaledW, 0);
   cv::normalize(dogImg, dogImg, 0, 255, cv::NORM_MINMAX);
  cv::imshow(skinMaskWin, dogImg);

  // Show edges
  cv::namedWindow(edgeWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(edgeWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(edgeWin, scaledW, scaledH);
  cv::moveWindow(edgeWin, 2 * scaledW, 0);
  cv::normalize(edgeImg, edgeImg, 0, 255, cv::NORM_MINMAX);
  cv::imshow(edgeWin, edgeImg);
  cv::waitKey();
  cv::destroyAllWindows();

  return 0;
}
