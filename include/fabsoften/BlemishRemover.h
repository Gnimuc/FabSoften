#ifndef BLEMISH_REMOVER_H
#define BLEMISH_REMOVER_H

#include <opencv2/imgproc.hpp>

namespace fabsoften {

/// BlemishRemoverOptions - Options for controlling the effect of blemish concealment.
class BlemishRemoverOptions {
public:
  /// Enable the lower and upper face region
  unsigned EnableFace : 1;

public:
  BlemishRemoverOptions() : EnableFace(true) {}
};

/// \brief Class for removing blemishes.
class BlemishRemover {
public:
  BlemishRemoverOptions opts;

public:
  explicit BlemishRemover(BlemishRemoverOptions op = BlemishRemoverOptions()) : opts(op) {}

  /// \brief Compute the Difference of Gaussian for the intensity channel of the image.
  ///
  /// \param src Input image. e.g. the `workImg` of \ref Beautifier.
  /// \param mask Binary mask with eltype `CV_8UC1`.
  void computeDoG(const cv::Mat &src, const cv::Mat &mask);

  void runCannyEdgeDetection();

  /// \brief
  /// \param src
  /// \param dst
  void removeBlemishes(const cv::Mat &src, cv::Mat &dst);

  /// \brief Blemish Concealment.
  /// \param src [in] Input image. e.g. the `workImg` of \ref Beautifier.
  /// \param dst [out] Output image.
  /// \param mask [in] Binary mask with eltype `CV_8UC1`.
  void concealBlemish(const cv::Mat &src, cv::Mat &dst, const cv::Mat &mask);

private:
  cv::Mat grayImg;
  cv::Mat workImg;
  cv::Mat workImg2;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
};

} // namespace fabsoften

#endif
