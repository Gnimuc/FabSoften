/// \file BlemishRemover.cpp
/// \brief BlemishRemover Implmentation
///

#include "fabsoften/BlemishRemover.h"
#include <ranges>

using namespace fabsoften;

void BlemishRemover::computeDoG(const cv::Mat &src, const cv::Mat &mask) {
  // Convert the RGB image to a single channel gray image
  cv::cvtColor(src, grayImg, cv::COLOR_BGR2GRAY);

  // Compute the DoG to detect edges
  const auto sigmaY = grayImg.cols / 200.0;
  const auto sigmaX = grayImg.rows / 200.0;
  cv::GaussianBlur(grayImg, workImg, cv::Size(3, 3), /*sigma=*/0);
  cv::GaussianBlur(grayImg, workImg2, cv::Size(0, 0), sigmaX, sigmaY);
  cv::subtract(workImg2, workImg, workImg);

  // Apply binary mask to the image
  cv::bitwise_and(mask, workImg, workImg2);

  // Discard uniform skin regions
  const int N = 2 * (std::min(workImg.cols, workImg.rows) / 50) + 1;
  cv::adaptiveThreshold(workImg2, workImg, /*maxValue=*/255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY, /*blockSize=*/N, 0);
  // Eroding
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
  cv::morphologyEx(workImg, workImg, cv::MORPH_ERODE, element);
}

void BlemishRemover::runCannyEdgeDetection() {
  // Apply Canny Edge Detection
  cv::GaussianBlur(workImg, workImg, cv::Size(0, 0), /*sigma=*/3);
  cv::Canny(workImg, workImg2, /*threshold1=*/0, /*threshold2=*/10000,
            /*apertureSize=*/7,
            /*L2gradient=*/false);

  // Dilate detected edges so the extreme outer contours can cover those blemishes
  cv::Mat elDilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
  cv::morphologyEx(workImg2, workImg2, cv::MORPH_DILATE, elDilate);

  // Store results in `workImg` as `workImg2` will be modified when calling `findContours`
  workImg2.copyTo(workImg);

  // Find contours from those detected edges
  cv::findContours(workImg2, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
}

void BlemishRemover::removeBlemishes(const cv::Mat &src, cv::Mat &dst) {
  constexpr auto ignoreThreshold = 30;
  constexpr auto traversalDepth = 10 * ignoreThreshold;
  const auto isBlemish = [](const auto &contour) {
    const auto len = cv::arcLength(contour, /*closed=*/true);
    return len < traversalDepth && len > ignoreThreshold;
  };
  for (const auto &contour : contours | std::views::filter(isBlemish)) {
    float b = 0.0, g = 0.0, r = 0.0;
    for (const auto &pt : contour) {
      const auto &bgr = src.at<cv::Vec3b>(pt);
      b += bgr[0], g += bgr[1], r += bgr[2];
    }
    const auto len = contour.size();
    b /= len, g /= len, r /= len;
    auto color = cv::Scalar(static_cast<int>(b), static_cast<int>(g), static_cast<int>(r));
    cv::fillPoly(dst, contour, color);
  }
}

void BlemishRemover::concealBlemish(const cv::Mat &src, cv::Mat &dst, const cv::Mat &mask) {
  computeDoG(src, mask);
  runCannyEdgeDetection();
  removeBlemishes(src, dst);
}
