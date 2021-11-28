#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace fabsoften {

/// GFOptions - Options for Guided Filtering.
class GFOptions {
public:
  /// The epsilon parameter in Guided Filtering.
  float eps;

  /// The radius parameter in Guided Filtering for skin regions.
  float radius4skin;

public:
  GFOptions() : eps(300), radius4skin(20) {}
};

/// \brief Class for Attribute-aware Dynamic Guided Filter.
class GuidedFilter {
public:
  GFOptions opts;

public:
  explicit GuidedFilter(GFOptions op = GFOptions()) : opts(op) {}

  // TODO: support more CV types

  /// \brief Blurs a single channel image with dynamic window size(radius).
  /// \param src [in] Input image(CV_32FC1).
  /// \param dst [out] Output image of the same size and type as src(CV_32FC1).
  /// \param radius [in] Float-valued matrix contains radius info for each pixel.
  void dynamicMeanFilter(const cv::Mat &src, cv::Mat &dst, const cv::Mat &radius);

  /// \brief Blurs a single channel image with guided filtering.
  /// \param src [in] Input image.
  /// \param guidance [in] Guidance color image.
  /// \param dst [out] Output image of the same size and type as src.
  /// \param radius [in] Float-valued matrix contains radius info for each pixel.
  void dynamicGuidedFilter(const cv::Mat &src, const cv::Mat &guidance, cv::Mat &dst,
                           const cv::Mat &radius, const double eps);

  void checkAndInit(const cv::Mat &src, const cv::Mat &guidance);

  /// \brief Blurs a color image with guided filtering.
  /// \param mask [in] Binary mask(CV_8UC1).
  /// \param guidance [in] Guidance Color Image.
  /// \param src [in] Input color image.
  /// \param dst [out] Output image of the same size and type as src.
  void applyADF(const cv::Mat &mask, const cv::Mat &guidance, const cv::Mat &src,
                cv::Mat &dst);

private:
  cv::Mat inputImg;
  cv::Mat guideImg;
  cv::Mat workImg;
  cv::Mat radiusImg;
  std::array<cv::Mat, 3> inputChannels;
  std::array<cv::Mat, 3> outputChannels;
  cv::Mat meanP;
  std::array<cv::Mat, 3> IChannels;
  std::array<cv::Mat, 3> meanIChannels;
  std::array<cv::Mat, 3> meanIpChannels;
  std::array<cv::Mat, 3> covIpChannels;
  cv::Mat var00, var01, var02, var11, var12, var22;
  std::array<cv::Mat, 3> aChannels;
};

} // namespace fabsoften

#endif
