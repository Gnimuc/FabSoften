#include "ADF.h"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/imgproc.hpp>

TEST_CASE("ADF", "[integral image]") {
  cv::Mat img(100, 100, CV_8UC3);
  cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

  cv::Mat integralImg;
  cv::integral(img, integralImg);

  SECTION("dimension") {
    REQUIRE(integralImg.rows == img.rows + 1);
    REQUIRE(integralImg.cols == img.cols + 1);
  }

  SECTION("summed-area table") {
    for (size_t x = 0; x < img.rows; ++x)
      for (size_t y = 0; y < img.cols; ++y) {
        const auto pA = cv::Point(y, x);
        const auto pB = cv::Point(y + 1, x);
        const auto pC = cv::Point(y, x + 1);
        const auto pD = cv::Point(y + 1, x + 1);
        const auto &sA = integralImg.at<cv::Vec3i>(pA);
        const auto &sB = integralImg.at<cv::Vec3i>(pB);
        const auto &sC = integralImg.at<cv::Vec3i>(pC);
        const auto &sD = integralImg.at<cv::Vec3i>(pD);
        const auto sum = sD + sA - sB - sC;
        const auto &val = img.at<cv::Vec3b>(x, y);
        REQUIRE(cv::Vec3i(val) == sum);
      }
  }

  constexpr auto r = 2; // radius
  const auto blockSize = cv::Size(2 * r + 1, 2 * r + 1);
  const auto anchor = cv::Point(-1, -1);
  cv::Mat boxImg;
  cv::boxFilter(img, boxImg, /*ddepth=*/-1, blockSize, anchor, /*normalize=*/true,
                cv::BORDER_ISOLATED);

  const auto nRow = integralImg.rows, nCol = integralImg.cols;
  SECTION("boxfilter") {
    cv::Mat meanImg = cv::Mat::zeros(img.size(), CV_8UC3);
    for (size_t x = 0; x < img.rows; ++x)
      for (size_t y = 0; y < img.cols; ++y) {
        const auto xA = std::max<int>(x - r, 0);
        const auto yA = std::max<int>(y - r, 0);
        const auto &sA = integralImg.at<cv::Vec3i>(xA, yA);
        const auto xB = std::max<int>(x - r, 0);
        const auto yB = std::min<int>(y + r + 1, nCol - 1);
        const auto &sB = integralImg.at<cv::Vec3i>(xB, yB);
        const auto xC = std::min<int>(x + r + 1, nRow - 1);
        const auto yC = std::max<int>(y - r, 0);
        const auto &sC = integralImg.at<cv::Vec3i>(xC, yC);
        const auto xD = std::min<int>(x + r + 1, nRow - 1);
        const auto yD = std::min<int>(y + r + 1, nCol - 1);
        const auto &sD = integralImg.at<cv::Vec3i>(xD, yD);
        meanImg.at<cv::Vec3b>(x, y) = (sD + sA - sB - sC) / ((2 * r + 1) * (2 * r + 1));
      }

    for (size_t x = 0; x < img.rows; ++x)
      for (size_t y = 0; y < img.cols; ++y)
        REQUIRE(boxImg.at<cv::Vec3b>(x, y) == meanImg.at<cv::Vec3b>(x, y));
  }
}
