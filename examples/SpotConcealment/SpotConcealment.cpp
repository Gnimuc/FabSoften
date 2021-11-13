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

/// @brief Show input image
static constexpr auto imageWin = "Input Image";

/// @brief Show Canny edge detection results
static constexpr auto cannyWin = "Canny Edges";

/// @brief Show preprocessed image
static constexpr auto processedWin = "Preprocessed Image";

/// @brief blemish concealment example
///
/// Usage: SpotConcealment.exe [params] image landmark_model
int main(int argc, char **argv) {
  // Handle command line arguments
  cv::CommandLineParser parser(argc, argv, cmdKeys);
  parser.about("Blemish Concealment example");
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
  const auto inputImg =
      cv::imread(cv::samples::findFile(imgArg, /*required=*/false, /*silentMode=*/true));
  if (inputImg.empty()) {
    std::cout << "Could not open or find the image: " << imgArg << "\n"
              << "The image should be located in `images_dir`.\n"
              << std::endl;
    parser.printMessage();
    return -1;
  }

  // Leave the original input image untouched
  cv::Mat workImg = inputImg.clone();

  // Downsampling
  // cv::pyrDown(inputImg, workImg);
  // cv::pyrDown(workImg, workImg);
  // cv::pyrDown(workImg, workImg);

  // Make a copy for drawing landmarks
  cv::Mat landmarkImg = workImg.clone();

  // Make a copy for drawing binary mask
  cv::Mat maskImg = cv::Mat::zeros(workImg.size(), CV_8UC1);

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
  const auto rightEyeBrow = [](const auto i) { return i >= 17 && i <= 21; };
  const auto leftEyeBrow = [](const auto i) { return i >= 22 && i <= 26; };

  const auto detect = [&](const auto &face) { return landmarkDetector(dlibImg, face); };
  for (const auto &shape : faces | std::views::transform(detect)) {
    std::vector<tinyspline::real> knots;
    // Join the landmark points on the boundary of facial features using cubic curve
    const auto getCurve = [&]<typename T>(T predicate, const auto n, bool isClosed = true) {
      knots.clear();
      for (const auto i :
           std::views::iota(0) | std::views::filter(predicate) | std::views::take(n)) {
        const auto &point = shape.part(i);
        knots.push_back(point.x());
        knots.push_back(point.y());
      }
      // Make a closed curve
      if (isClosed) {
        knots.push_back(knots[0]);
        knots.push_back(knots[1]);
      }
      // Interpolate the curve
      auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);
      return spline;
    };

    // Right eye cubic curve
    constexpr auto nEyeCurve = 6;
    const auto rightEyeCurve = getCurve(rightEye, nEyeCurve);
    // Sample landmark points from the curve
    constexpr auto eyePointNum = 25;
    std::array<cv::Point, eyePointNum> rightEyePts;
    for (const auto i : std::views::iota(0, eyePointNum)) {
      const auto net = rightEyeCurve(1.0 / eyePointNum * i);
      const auto result = net.result();
      const auto x = result[0], y = result[1];
      drawLandmark(x, y);
      rightEyePts[i] = cv::Point(x, y);
    }
    // Draw binary mask
    cv::fillConvexPoly(maskImg, rightEyePts, cv::Scalar(255), cv::LINE_AA);

    // Left eye cubic curve
    const auto leftEyeCurve = getCurve(leftEye, nEyeCurve);
    std::array<cv::Point, eyePointNum> leftEyePts;
    // Sample landmark points from the curve
    for (const auto i : std::views::iota(0, eyePointNum)) {
      const auto net = leftEyeCurve(1.0 / eyePointNum * i);
      const auto result = net.result();
      const auto x = result[0], y = result[1];
      drawLandmark(x, y);
      leftEyePts[i] = cv::Point(x, y);
    }
    // Draw binary mask
    cv::fillConvexPoly(maskImg, leftEyePts, cv::Scalar(255), cv::LINE_AA);

    // Mouth cubic curve
    constexpr auto nMouthCurve = 12;
    const auto mouthCurve = getCurve(mouthBoundary, nMouthCurve);
    constexpr auto mouthPointNum = 40;
    std::array<cv::Point, mouthPointNum> mouthPts;
    // Sample landmark points from the curve
    for (const auto i : std::views::iota(0, mouthPointNum)) {
      const auto net = mouthCurve(1.0 / mouthPointNum * i);
      const auto result = net.result();
      const auto x = result[0], y = result[1];
      drawLandmark(x, y);
      mouthPts[i] = cv::Point(x, y);
    }
    // Draw binary mask
    cv::fillPoly(maskImg, mouthPts, cv::Scalar(255), cv::LINE_AA);

    // Estimate an ellipse that can complete the upper face region
    constexpr auto nJaw = 17;
    std::vector<cv::Point> lowerFacePts;
    for (auto i : std::views::iota(0) | std::views::filter(jaw) | std::views::take(nJaw)) {
      const auto &point = shape.part(i);
      const auto x = point.x(), y = point.y();
      drawLandmark(x, y);
      lowerFacePts.push_back(cv::Point(x, y));
    }
    // Guess a point located in the upper face region
    // Pb: 8 (bottom of jaw)
    // Pt: 27 (top of nose
    const auto &Pb = shape.part(8);
    const auto &Pt = shape.part(27);
    const auto x = Pb.x();
    const auto y = Pt.y() - 0.85 * abs(Pb.y() - Pt.y());
    drawLandmark(x, y);
    lowerFacePts.push_back(cv::Point(x, y));
    // Fit ellipse
    const auto box = cv::fitEllipseDirect(lowerFacePts);
    cv::Mat maskTmp = cv::Mat(maskImg.size(), CV_8UC1, cv::Scalar(255));
    cv::ellipse(maskTmp, box, cv::Scalar(0), /*thickness=*/-1, cv::FILLED);
    cv::bitwise_or(maskTmp, maskImg, maskImg);
    cv::bitwise_not(maskImg, maskImg);

    // Also add simple eye brow masks
    constexpr auto nEyeBrow = 5;
    constexpr auto eyeBrowPointNum = 50;
    const auto offset = workImg.cols / 100;
    std::array<cv::Point, eyeBrowPointNum> eyeBrowPts;
    const auto leftEyeBrowCurve = getCurve(leftEyeBrow, nEyeBrow, false);
    for (const auto i : std::views::iota(0, eyeBrowPointNum)) {
      const auto net = leftEyeBrowCurve(1.0 / eyeBrowPointNum * i);
      const auto result = net.result();
      const auto x = result[0], y = result[1];
      eyeBrowPts[i] = cv::Point(x, y + offset);
    }
    const auto color = cv::Scalar(0, 0, 0);
    const auto thickness = workImg.rows / 50;
    const auto center = cv::Point(x, y);
    cv::polylines(maskImg, eyeBrowPts, /*isClosed=*/false, color, thickness, cv::LINE_AA);

    const auto rightEyeBrowCurve = getCurve(rightEyeBrow, nEyeBrow, false);
    for (const auto i : std::views::iota(0, eyeBrowPointNum)) {
      const auto net = rightEyeBrowCurve(1.0 / eyeBrowPointNum * i);
      const auto result = net.result();
      const auto x = result[0], y = result[1];
      eyeBrowPts[i] = cv::Point(x, y + offset);
    }
    cv::polylines(maskImg, eyeBrowPts, /*isClosed=*/false, color, thickness, cv::LINE_AA);
  }

  // Expand the mask a bit
  cv::Mat maskEx;
  cv::Mat maskElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(71, 71));
  cv::morphologyEx(maskImg, maskEx, cv::MORPH_ERODE, maskElement);
  cv::Mat maskExs[3] = {maskEx, maskEx, maskEx};
  cv::Mat maskEx3C;
  cv::merge(maskExs, 3, maskEx3C);

  // Make a preserved image for future use
  cv::Mat preservedImg, maskPres;
  cv::bitwise_not(maskEx3C, maskPres);
  cv::bitwise_and(workImg, maskPres, preservedImg);

  // Spot Concealment
  // Convert the RGB image to a single channel gray image
  cv::Mat grayImg;
  cv::cvtColor(workImg, grayImg, cv::COLOR_BGR2GRAY);

  // Compute the DoG to detect edges
  cv::Mat blurImg1, blurImg2, dogImg;
  const auto sigmaY = grayImg.cols / 200.0;
  const auto sigmaX = grayImg.rows / 200.0;
  cv::GaussianBlur(grayImg, blurImg1, cv::Size(3, 3), /*sigma=*/0);
  cv::GaussianBlur(grayImg, blurImg2, cv::Size(0, 0), sigmaX, sigmaY);
  cv::subtract(blurImg2, blurImg1, dogImg);

  // Apply binary mask to the image
  cv::bitwise_and(maskImg, dogImg, dogImg);

  // Discard uniform skin regions
  cv::Mat threshImg;
  const int sizeAT = 2 * (std::min(dogImg.cols, dogImg.rows) / 50) + 1;
  cv::adaptiveThreshold(dogImg, threshImg, /*maxValue=*/255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, /*blockSize=*/sizeAT, 0);
  // Eroding
  cv::Mat elErode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
  cv::morphologyEx(threshImg, threshImg, cv::MORPH_ERODE, elErode);

  // Apply Canny Edge Detection
  cv::GaussianBlur(threshImg, threshImg, cv::Size(0, 0), /*sigma=*/3);
  cv::Mat edgeImg;
  cv::Canny(threshImg, edgeImg, /*threshold1=*/0, /*threshold2=*/10000, /*apertureSize=*/7,
            /*L2gradient=*/false);

  // Dilate detected edges so the extreme outer contours can cover those blemishes
  cv::Mat elDilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
  cv::morphologyEx(edgeImg, edgeImg, cv::MORPH_DILATE, elDilate);

  // Find contours from those detected edges
  std::vector<cv::Vec4i> hierarchy;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edgeImg.clone(), contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Perform a depth-first traversal to find blemish boundaries
  constexpr auto ignoreThreshold = 30;
  constexpr auto traversalDepth = 10 * ignoreThreshold;
  const auto isBlemish = [](const auto &contour) {
    const auto len = cv::arcLength(contour, /*closed=*/true);
    return len < traversalDepth && len > ignoreThreshold;
  };
  auto preprocssedImg = workImg.clone();
  for (const auto &contour : contours | std::views::filter(isBlemish)) {
    float b = 0.0, g = 0.0, r = 0.0;
    for (const auto &pt : contour) {
      const auto &bgr = workImg.at<cv::Vec3b>(pt);
      b += bgr[0], g += bgr[1], r += bgr[2];
    }
    const auto len = contour.size();
    b /= len, g /= len, r /= len;
    auto color = cv::Scalar(static_cast<int>(b), static_cast<int>(g), static_cast<int>(r));
    cv::fillPoly(preprocssedImg, contour, color);
  }

  // Undo blemish concealment in the preserved facial zone
  cv::bitwise_and(preprocssedImg, maskEx3C, preprocssedImg);
  cv::add(preprocssedImg, preservedImg, preprocssedImg);

  // Fit image to the screen and show image
  cv::namedWindow(imageWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(imageWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  const auto [x, y, resW, resH] = cv::getWindowImageRect(imageWin);
  const auto [imgW, imgH] = landmarkImg.size();
  const auto scaleFactor = 30;
  const auto scaledW = scaleFactor * resW / 100;
  const auto scaledH = scaleFactor * imgH * resW / (imgW * 100);
  cv::resizeWindow(imageWin, scaledW, scaledH);
  cv::imshow(imageWin, inputImg);

  // Show Canny edge detection result
  cv::namedWindow(cannyWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(cannyWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(cannyWin, scaledW, scaledH);
  cv::moveWindow(cannyWin, scaledW, 0);
  cv::imshow(cannyWin, edgeImg);

  // Show preprocessed image
  cv::namedWindow(processedWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(processedWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(processedWin, scaledW, scaledH);
  cv::moveWindow(processedWin, 2 * scaledW, 0);
  cv::imshow(processedWin, preprocssedImg);

  cv::waitKey();
  cv::destroyAllWindows();

  return 0;
}
