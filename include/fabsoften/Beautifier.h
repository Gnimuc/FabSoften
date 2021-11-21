#ifndef BEAUTIFIER_H
#define BEAUTIFIER_H

#include "fabsoften/FaceLandmarkDetector.h"

namespace fabsoften {

/**
 * @brief
 *
 */
class Beautifier {
public:
  explicit Beautifier(const std::string inputImgPath, const std::string landmarkModelPath);

  Beautifier(const Beautifier &) = delete;
  Beautifier &operator=(const Beautifier &) = delete;

  /**
   * @brief Run the FabSoften pipeline.
   *
   */
  void soften();

  /// Downsampling \ref workImg
  void downsampling();

  /// Create and init the face landmark detector.
  void createFaceLandmarkDetector();

  bool hasFaceLandmarkDetector() const { return detector != nullptr; }

  FaceLandmarkDetector &getFaceLandmarkDetector() const {
    assert(detector && "Beautifier has no FaceLandmarkDetector!");
    return *detector;
  }

  /// Remove current \ref detector and give the ownership to the caller
  std::unique_ptr<FaceLandmarkDetector> takeFaceLandmarkDetector() {
    return std::move(detector);
  }

  /// Set current \ref detector to \p and take ownership of \p D.
  void setFaceLandmarkDetector(std::unique_ptr<FaceLandmarkDetector> newDetector);

  static void drawLandmark(cv::Mat &img, cv::Point pt);

private:
  /// Path to the input image
  std::string imgPath;

  /// Path to facial landmark detector model
  std::string modelPath;

  /// Face Landmark Detector
  std::unique_ptr<FaceLandmarkDetector> detector;

  /// Maintain a copy of the input image
  cv::Mat inputImg;

  /// Work image
  cv::Mat workImg;
};

} // namespace fabsoften

#endif