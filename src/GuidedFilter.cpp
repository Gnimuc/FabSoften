/// \file GuidedFilter.cpp
/// \brief GuidedFilter Implmentation
///

#include "fabsoften/GuidedFilter.h"

using namespace fabsoften;

void GuidedFilter::dynamicMeanFilter(const cv::Mat &src, cv::Mat &dst,
                                     const cv::Mat &radius) {
  CV_Assert(src.depth() == CV_32F && radius.depth() == CV_32F);
  cv::integral(src, workImg);
  const auto nRow = workImg.rows, nCol = workImg.cols;
  for (size_t x = 0; x < src.rows; ++x)
    for (size_t y = 0; y < src.cols; ++y) {
      const auto r = radius.at<float>(x, y);
      const auto xA = std::max<int>(x - r, 0);
      const auto yA = std::max<int>(y - r, 0);
      const auto &sA = workImg.at<double>(xA, yA);
      const auto xB = std::max<int>(x - r, 0);
      const auto yB = std::min<int>(y + r + 1, nCol - 1);
      const auto &sB = workImg.at<double>(xB, yB);
      const auto xC = std::min<int>(x + r + 1, nRow - 1);
      const auto yC = std::max<int>(y - r, 0);
      const auto &sC = workImg.at<double>(xC, yC);
      const auto xD = std::min<int>(x + r + 1, nRow - 1);
      const auto yD = std::min<int>(y + r + 1, nCol - 1);
      const auto &sD = workImg.at<double>(xD, yD);
      dst.at<float>(x, y) = (sD + sA - sB - sC) / ((2 * r + 1) * (2 * r + 1));
    }
}

// TODO: fuse all of these separated loops
void GuidedFilter::dynamicGuidedFilter(const cv::Mat &src, const cv::Mat &guidance,
                                       cv::Mat &dst, const cv::Mat &radius,
                                       const double eps) {
  checkAndInit(src, guidance);
  // Split color image to channels
  cv::split(guideImg, IChannels);
  // Mean of I & P in each local patch
  dynamicMeanFilter(inputImg, meanP, radius);
  dynamicMeanFilter(IChannels[0], meanIChannels[0], radius);
  dynamicMeanFilter(IChannels[1], meanIChannels[1], radius);
  dynamicMeanFilter(IChannels[2], meanIChannels[2], radius);
  // Covariance of I & P in each local patch
  dynamicMeanFilter(IChannels[0].mul(inputImg), meanIpChannels[0], radius);
  dynamicMeanFilter(IChannels[1].mul(inputImg), meanIpChannels[1], radius);
  dynamicMeanFilter(IChannels[2].mul(inputImg), meanIpChannels[2], radius);

  covIpChannels[0] = meanIpChannels[0] - meanIChannels[0].mul(meanP);
  covIpChannels[1] = meanIpChannels[1] - meanIChannels[1].mul(meanP);
  covIpChannels[2] = meanIpChannels[2] - meanIChannels[2].mul(meanP);

  // Variance of I in each local patch
  dynamicMeanFilter(IChannels[0].mul(IChannels[0]), var00, radius);
  dynamicMeanFilter(IChannels[0].mul(IChannels[1]), var01, radius);
  dynamicMeanFilter(IChannels[0].mul(IChannels[2]), var02, radius);
  dynamicMeanFilter(IChannels[1].mul(IChannels[1]), var11, radius);
  dynamicMeanFilter(IChannels[1].mul(IChannels[2]), var12, radius);
  dynamicMeanFilter(IChannels[2].mul(IChannels[2]), var22, radius);
  var00 -= meanIChannels[0].mul(meanIChannels[0]);
  var01 -= meanIChannels[0].mul(meanIChannels[1]);
  var02 -= meanIChannels[0].mul(meanIChannels[2]);
  var11 -= meanIChannels[1].mul(meanIChannels[1]);
  var12 -= meanIChannels[1].mul(meanIChannels[2]);
  var22 -= meanIChannels[2].mul(meanIChannels[2]);
  // Compute a
  for (size_t x = 0; x < inputImg.rows; ++x)
    for (size_t y = 0; y < inputImg.cols; ++y) {
      // Unroll the 3x3 matrix algebra
      // Sigma Matrix:
      // s00 s01 s02
      // s01 s11 s12
      // s02 s12 s22
      double s00 = var00.at<float>(x, y);
      double s01 = var01.at<float>(x, y);
      double s02 = var02.at<float>(x, y);
      double s11 = var11.at<float>(x, y);
      double s12 = var12.at<float>(x, y);
      double s22 = var22.at<float>(x, y);
      // Sigma = Sigma + eps * I
      s00 += eps;
      s11 += eps;
      s22 += eps;
      // Determinant should not be 0
      double det = s00 * (s22 * s11 - s12 * s12) - s01 * (s22 * s01 - s12 * s02) +
                   s02 * (s12 * s01 - s11 * s02);
      CV_Assert(det != 0);
      // Compute the inverse matrix of Sigma
      double inv00 = s22 * s11 - s12 * s12;
      double inv01 = s02 * s12 - s22 * s01;
      double inv02 = s01 * s12 - s02 * s11;
      double inv11 = s22 * s00 - s02 * s02;
      double inv12 = s01 * s02 - s00 * s12;
      double inv22 = s00 * s11 - s01 * s01;
      const double detNew = (s00 * inv00) + (s01 * inv01) + (s02 * inv02);
      inv00 /= detNew;
      inv01 /= detNew;
      inv02 /= detNew;
      inv11 /= detNew;
      inv12 /= detNew;
      inv22 /= detNew;
      // Compute a
      const double cov0 = covIpChannels[0].at<float>(x, y);
      const double cov1 = covIpChannels[1].at<float>(x, y);
      const double cov2 = covIpChannels[2].at<float>(x, y);
      double a0 = cov0 * inv00 + cov1 * inv01 + cov2 * inv02;
      double a1 = cov0 * inv01 + cov1 * inv11 + cov2 * inv12;
      double a2 = cov0 * inv02 + cov1 * inv12 + cov2 * inv22;
      aChannels[0].at<float>(x, y) = a0;
      aChannels[1].at<float>(x, y) = a1;
      aChannels[2].at<float>(x, y) = a2;
    }

  // Compute b (resue `varXX` to reduce allocations)
  var22 = meanP - aChannels[0].mul(meanIChannels[0]) - aChannels[1].mul(meanIChannels[1]) -
          aChannels[2].mul(meanIChannels[2]);
  // Compute the final result
  dynamicMeanFilter(aChannels[0], /*meanAChannels[0]=*/var00, radius);
  dynamicMeanFilter(aChannels[1], /*meanAChannels[1]=*/var01, radius);
  dynamicMeanFilter(aChannels[2], /*meanAChannels[2]=*/var02, radius);
  dynamicMeanFilter(/*b=*/var22, /*meanB=*/var11, radius);
  dst = var00.mul(IChannels[0]) + var01.mul(IChannels[1]) + var02.mul(IChannels[2]) + var11;
}

void GuidedFilter::checkAndInit(const cv::Mat &src, const cv::Mat &guidance) {
  CV_Assert(guidance.channels() == 3 && src.channels() == 1);

  src.copyTo(inputImg);
  if (inputImg.depth() != CV_32F)
    inputImg.convertTo(inputImg, CV_32F);

  guidance.copyTo(guideImg);
  if (guideImg.depth() != CV_32F)
    guideImg.convertTo(guideImg, CV_32F);

  // Re-init work arrays if the input image size got changed.
  const auto sizeSrc = src.size();

  if (meanP.size() != sizeSrc)
    meanP = cv::Mat::zeros(sizeSrc, CV_32FC1);

  if (meanIChannels[0].size() != sizeSrc) {
    meanIChannels[0] = cv::Mat::zeros(sizeSrc, CV_32FC1);
    meanIChannels[1] = cv::Mat::zeros(sizeSrc, CV_32FC1);
    meanIChannels[2] = cv::Mat::zeros(sizeSrc, CV_32FC1);
  }

  if (meanIpChannels[0].size() != sizeSrc) {
    meanIpChannels[0] = cv::Mat::zeros(sizeSrc, CV_32FC1);
    meanIpChannels[1] = cv::Mat::zeros(sizeSrc, CV_32FC1);
    meanIpChannels[2] = cv::Mat::zeros(sizeSrc, CV_32FC1);
  }

  if (var00.size() != sizeSrc) {
    var00 = cv::Mat::zeros(sizeSrc, CV_32FC1);
    var01 = cv::Mat::zeros(sizeSrc, CV_32FC1);
    var02 = cv::Mat::zeros(sizeSrc, CV_32FC1);
    var11 = cv::Mat::zeros(sizeSrc, CV_32FC1);
    var12 = cv::Mat::zeros(sizeSrc, CV_32FC1);
    var22 = cv::Mat::zeros(sizeSrc, CV_32FC1);
  }

  if (aChannels[0].size() != sizeSrc) {
    aChannels[0] = cv::Mat::zeros(sizeSrc, CV_32FC1);
    aChannels[1] = cv::Mat::zeros(sizeSrc, CV_32FC1);
    aChannels[2] = cv::Mat::zeros(sizeSrc, CV_32FC1);
  }
}

void GuidedFilter::runADF(const cv::Mat &mask, const cv::Mat &guidance, const cv::Mat &src,
                          cv::Mat &dst) {
  // constexpr auto alphaRadius = 10;
  // constexpr auto betaRadius = 10;

  // const double nSpots = std::ranges::count_if(contours, isBlemish);
  // const double spotFactor = nSpots / 60;
  // const double alphaEps = 5 * spotFactor;
  // const double betaEps = 100 * spotFactor;
  // const double eps = alphaEps * nSpots + betaEps;

  const double eps = opts.eps;
  if (radiusImg.size() != src.size())
    radiusImg = cv::Mat::ones(src.size(), CV_8UC1);

  cv::bitwise_and(radiusImg, mask, radiusImg);
  radiusImg.convertTo(radiusImg, CV_32F);
  radiusImg *= opts.radius4skin;

  CV_Assert(src.channels() == 3);
  cv::split(src, inputChannels);
  dynamicGuidedFilter(inputChannels[0], guidance, outputChannels[0], radiusImg, eps);
  dynamicGuidedFilter(inputChannels[1], guidance, outputChannels[1], radiusImg, eps);
  dynamicGuidedFilter(inputChannels[2], guidance, outputChannels[2], radiusImg, eps);
  cv::merge(outputChannels, dst);
}
