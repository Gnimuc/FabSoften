/**
 * @file SkinMapGeneration.cpp
 * @brief An example demonstrates how to generate a skin segmentaiton map using GMM.
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
#include <opencv2/ml.hpp>
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

/// @brief Show skin map
static constexpr auto mapWin = "Skin Map";

/// @brief Show preprocessed image
static constexpr auto processedWin = "preprocessed Image";

/// @brief blemish concealment example
///
/// Usage: SkinMapGeneration.exe [params] image landmark_model
int main(int argc, char **argv) {
  // Handle command line arguments
  cv::CommandLineParser parser(argc, argv, cmdKeys);
  parser.about("Skin map generation example");
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

  // Leave the original input image untouched
  cv::Mat workImg = inputImg.clone();

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
  const auto rightEyeBrow = [](const auto i) { return i >= 17 && i <= 21; };
  const auto leftEyeBrow = [](const auto i) { return i >= 22 && i <= 26; };

  const auto detect = [&](const auto &face) { return landmarkDetector(dlibImg, face); };
  for (const auto &shape : faces | std::views::transform(detect)) {
    std::vector<tinyspline::real> knots;
    // Join the landmark points on the boundary of facial features using cubic curve
    const auto getCurve = [&]<typename T>(T predicate, const auto n, bool isClosed = true) {
      knots.clear();
      for (auto i :
           std::views::iota(0) | std::views::filter(predicate) | std::views::take(n)) {
        const auto point = shape.part(i);
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

    // Also add simple eye brow masks
    constexpr auto eyeBrowPointNum = 50;
    const auto offset = workImg.cols / 100;
    std::array<cv::Point, eyeBrowPointNum> eyeBrowPts;
    const auto leftEyeBrowCurve = getCurve(leftEyeBrow, 5, false);
    for (const auto i : std::views::iota(0, eyeBrowPointNum)) {
      auto net = leftEyeBrowCurve(1.0 / eyeBrowPointNum * i);
      auto result = net.result();
      auto x = result[0], y = result[1];
      eyeBrowPts[i] = cv::Point(x, y + offset);
    }
    const auto color = cv::Scalar(0, 0, 0);
    const auto thickness = workImg.rows / 50;
    const auto center = cv::Point(x, y);
    cv::polylines(maskImg, eyeBrowPts, false, color, thickness, cv::LINE_AA);

    const auto rightEyeBrowCurve = getCurve(rightEyeBrow, 5, false);
    for (const auto i : std::views::iota(0, eyeBrowPointNum)) {
      auto net = rightEyeBrowCurve(1.0 / eyeBrowPointNum * i);
      auto result = net.result();
      auto x = result[0], y = result[1];
      eyeBrowPts[i] = cv::Point(x, y + offset);
    }
    cv::polylines(maskImg, eyeBrowPts, false, color, thickness, cv::LINE_AA);
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
  cv::GaussianBlur(grayImg, blurImg1, cv::Size(3, 3), 0);
  cv::GaussianBlur(grayImg, blurImg2, cv::Size(0, 0), sigmaX, sigmaY);
  cv::subtract(blurImg2, blurImg1, dogImg);

  // Apply binary mask to the image
  cv::bitwise_and(maskImg, dogImg, dogImg);

  // Discard uniform skin regions
  cv::Mat threshImg;
  const int sizeAT = 2 * (std::min(dogImg.cols, dogImg.rows) / 50) + 1;
  cv::adaptiveThreshold(dogImg, threshImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, sizeAT, 0);
  // Eroding
  cv::Mat elErode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
  cv::morphologyEx(threshImg, threshImg, cv::MORPH_ERODE, elErode);

  // Apply Canny Edge Detection
  cv::GaussianBlur(threshImg, threshImg, cv::Size(0, 0), 3);
  cv::Mat edgeImg;
  cv::Canny(threshImg, edgeImg, 0, 10000, 7, false);

  // Dilate detected edges so the extreme outer contours can cover those blemishes
  cv::Mat elDilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
  cv::morphologyEx(edgeImg, edgeImg, cv::MORPH_DILATE, elDilate);

  // Find contours from those detected edges
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edgeImg.clone(), contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Perform a depth-first traversal to find blemish boundaries
  constexpr auto ignoreThreshold = 30;
  constexpr auto traversalDepth = 10 * ignoreThreshold;
  const auto isBlemish = [](const auto &contour) {
    const auto len = cv::arcLength(contour, true);
    return len < traversalDepth && len > ignoreThreshold;
  };
  auto preprocessedImg = workImg.clone();
  for (const auto &contour : contours | std::views::filter(isBlemish)) {
    float b = 0.0, g = 0.0, r = 0.0;
    for (const auto &pt : contour) {
      auto color = workImg.at<cv::Vec3b>(pt);
      b += color[0], g += color[1], r += color[2];
    }
    auto len = contour.size();
    b /= len, g /= len, r /= len;
    auto color = cv::Scalar(static_cast<int>(b), static_cast<int>(g), static_cast<int>(r));
    cv::fillPoly(preprocessedImg, contour, color);
  }

  // Undo blemish concealment in the preserved facial zone
  cv::bitwise_and(preprocessedImg, maskEx3C, preprocessedImg);
  cv::add(preprocessedImg, preservedImg, preprocessedImg);

  // Skin Mask Generation
  // Generate skin and non-skin mask
  // clang-format off
  // The 68 facial landmark from the iBUG 300-W dataset(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
  // Right face:        1  2  3  4 31 32 33 30 29 28 39 40 41 36  0
  // Left face:        15 14 13 12 35 34 33 30 29 28 42 47 46 45 16
  const size_t rightFaceIdxs[15] = {1, 2, 3, 4, 31, 32, 33, 30, 29, 28, 39, 40, 41, 36, 1};
  const size_t leftFaceIdxs[15] = {15, 14, 13, 12, 35, 34, 33, 30, 29, 28, 42, 47, 46, 45, 15};
  // clang-format on
  cv::Mat skinMask = cv::Mat::zeros(maskImg.size(), CV_8UC1);
  for (const auto &shape : faces | std::views::transform(detect)) {
    std::vector<cv::Point> landmarks;
    for (auto i : std::views::iota(0) | std::views::take(shape.num_parts())) {
      const auto point = shape.part(i);
      landmarks.push_back(cv::Point(point.x(), point.y()));
    }
    std::vector<tinyspline::real> knots;
    // Right face zone
    for (auto idx : rightFaceIdxs) {
      const auto pt = landmarks[idx];
      knots.push_back(pt.x);
      knots.push_back(pt.y);
    }
    auto rightFaceSpline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);
    constexpr auto sampleNum = 50;
    std::array<cv::Point, sampleNum> rightFacePts;
    for (const auto i : std::views::iota(0, sampleNum)) {
      auto net = rightFaceSpline(1.0 / sampleNum * i);
      auto result = net.result();
      auto x = result[0], y = result[1];
      rightFacePts[i] = cv::Point(x, y);
    }
    cv::fillConvexPoly(skinMask, rightFacePts, cv::Scalar(255), cv::LINE_AA);
    knots.clear();
    // Left face zone
    for (auto idx : leftFaceIdxs) {
      const auto pt = landmarks[idx];
      knots.push_back(pt.x);
      knots.push_back(pt.y);
    }
    auto leftFaceSpline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);
    std::array<cv::Point, sampleNum> leftFacePts;
    for (const auto i : std::views::iota(0, sampleNum)) {
      auto net = leftFaceSpline(1.0 / sampleNum * i);
      auto result = net.result();
      auto x = result[0], y = result[1];
      leftFacePts[i] = cv::Point(x, y);
    }
    cv::fillConvexPoly(skinMask, leftFacePts, cv::Scalar(255), cv::LINE_AA);
  };
  // skin mask
  cv::bitwise_and(skinMask, maskEx, skinMask);
  cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, maskElement);
  cv::Mat skinMaskExs[3] = {skinMask, skinMask, skinMask};
  cv::Mat skinMaskEx3C;
  cv::merge(skinMaskExs, 3, skinMaskEx3C);
  cv::Mat skinPre = preprocessedImg.clone();
  cv::bitwise_and(skinMaskEx3C, skinPre, skinPre);
  // non-skin mask
  cv::Mat nonSkinMask;
  cv::bitwise_not(maskEx, nonSkinMask);
  cv::Mat nonSkinMaskExs[3] = {nonSkinMask, nonSkinMask, nonSkinMask};
  cv::Mat nonSkinMaskEx3C;
  cv::merge(nonSkinMaskExs, 3, nonSkinMaskEx3C);
  cv::Mat nonSkinPre = preprocessedImg.clone();
  cv::bitwise_and(nonSkinMaskEx3C, nonSkinPre, nonSkinPre);

  // Generate a skin mask cluster
  // Downsampling
  cv::Mat skinPreDn;
  cv::pyrDown(skinPre, skinPreDn);
  cv::pyrDown(skinPreDn, skinPreDn);
  // cv::pyrDown(skinPreDn, skinPreDn);
  skinPreDn.convertTo(skinPreDn, CV_64F);
  cv::randShuffle(skinPreDn);
  constexpr auto nsamples = 1000;
  cv::Mat skinSamples;
  const auto getSkinColor = [&](auto i) { return skinPreDn.at<cv::Vec3d>(i); };
  const auto isNotBlack = [](auto x) { return x != cv::Vec3d(0, 0, 0); };
  for (const auto v : std::views::iota(0) | std::views::transform(getSkinColor) |
                          std::views::filter(isNotBlack) | std::views::take(nsamples))
    skinSamples.push_back(v);
  skinSamples = skinSamples.reshape(1, 0);
  auto skinModel = cv::ml::EM::create();
  constexpr auto K = 6;
  skinModel->setClustersNumber(K);
  skinModel->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
  skinModel->setTermCriteria(
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 300, 0.1));
  skinModel->trainEM(skinSamples);

  cv::RNG rng(1);
  std::array<cv::Scalar, K> colors;
  for (const auto i : std::views::iota(0, K))
    colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

  // Generate a non-skin mask cluster
  // cv::Mat nonSkinPreDn;
  // cv::pyrDown(nonSkinPre, nonSkinPreDn);
  // cv::pyrDown(nonSkinPreDn, nonSkinPreDn);
  //// cv::pyrDown(nonSkinPreDn, nonSkinPreDn);
  // nonSkinPreDn.convertTo(nonSkinPreDn, CV_64F);
  // cv::randShuffle(nonSkinPreDn);
  // cv::Mat nonSkinSamples;
  // const auto getNonSkinColor = [&](auto i) { return nonSkinPreDn.at<cv::Vec3d>(i); };
  // for (const auto v : std::views::iota(0) | std::views::transform(getNonSkinColor) |
  //                        std::views::filter(isNotBlack) | std::views::take(nsamples))
  //  nonSkinSamples.push_back(v);
  // nonSkinSamples = nonSkinSamples.reshape(1, 0);
  // auto nonSkinModel = cv::ml::EM::create();
  // nonSkinModel->setClustersNumber(8);
  // nonSkinModel->trainEM(nonSkinSamples);

  // Generate a skin probability mask
  // Downsampling
  cv::Mat predictImgDn;
  cv::pyrDown(workImg, predictImgDn);
  cv::pyrDown(predictImgDn, predictImgDn);
  // cv::pyrDown(predictImgDn, predictImgDn);
  cv::Mat sample(1, 3, CV_64FC1);
  cv::Mat probs;
  cv::Mat skinProbMask = cv::Mat::zeros(skinPreDn.size(), CV_8UC3);
  for (const auto i : std::views::iota(0) | std::views::take(predictImgDn.total())) {
    auto v = predictImgDn.at<cv::Vec3b>(i);
    sample.at<double>(0) = v[0];
    sample.at<double>(1) = v[1];
    sample.at<double>(2) = v[2];
    auto result = skinModel->predict2(sample, probs);
    int idx = cvRound(result[1]);
    auto c = colors[idx];
    // if (probs.at<double>(idx) > 0.9)
    skinProbMask.at<cv::Vec3b>(i) = cv::Vec3b(c[0], c[1], c[2]);
  }

  // Fit image to the screen and show image
  cv::namedWindow(imageWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(imageWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  auto [x, y, resW, resH] = cv::getWindowImageRect(imageWin);
  auto [imgW, imgH] = inputImg.size();
  const auto scaleFactor = 30;
  const auto scaledW = scaleFactor * resW / 100;
  const auto scaledH = scaleFactor * imgH * resW / (imgW * 100);
  cv::resizeWindow(imageWin, scaledW, scaledH);
  cv::imshow(imageWin, inputImg);

  // Show generated skin map
  cv::namedWindow(mapWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(mapWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(mapWin, scaledW, scaledH);
  cv::moveWindow(mapWin, scaledW, 0);
  cv::addWeighted(skinMaskEx3C, 0.3, workImg, 1, 0, skinMaskEx3C);
  // predictImgDn.convertTo(predictImgDn, CV_8U);
  cv::imshow(mapWin, skinProbMask);

  // Show preprocessed image
  cv::namedWindow(processedWin, cv::WINDOW_NORMAL);
  cv::setWindowProperty(processedWin, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
  cv::resizeWindow(processedWin, scaledW, scaledH);
  cv::moveWindow(processedWin, 2 * scaledW, 0);
  cv::Mat skinProbMaskUp;
  cv::resize(skinProbMask, skinProbMaskUp, workImg.size(), cv::INTER_LINEAR);
  cv::addWeighted(skinProbMaskUp, 0.3, workImg, 1, 0, skinProbMaskUp);
  cv::imshow(processedWin, skinProbMaskUp);
  cv::waitKey();
  cv::destroyAllWindows();

  return 0;
}
