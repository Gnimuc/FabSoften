#ifndef BEAUTIFIER_H
#define BEAUTIFIER_H

#include "fabsoften/FaceLandmarkDetector.h"
#include "fabsoften/SkinMaskGenerator.h"

namespace fabsoften {

/// \brief Helper class for managing modules of FabSoften.
class Beautifier {
public:
  explicit Beautifier(const std::string inputImgPath, const std::string landmarkModelPath);

  /// Run the FabSoften pipeline.
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

  /// Set current \ref detector to \p newDetector and take ownership of \p newDetector.
  void setFaceLandmarkDetector(std::unique_ptr<FaceLandmarkDetector> newDetector);

  /// Create the Face object with the landmarks from \ref detector.
  void createFace();

  bool hasFace() const { return theFace != nullptr; }

  Face &getFace() const {
    assert(theFace && "Beautifier has no Face object!");
    return *theFace;
  }

  CurveFittingOptions &getCurveFittingOpts() { return curveFitVis->opts; }
  const CurveFittingOptions &getCurveFittingOpts() const { return curveFitVis->opts; }

  bool hasSkinMaskGenerator() const { return maskGen != nullptr; }

  SkinMaskGenerator &getSkinMaskGenerator() const {
    assert(maskGen && "Beautifier has no SkinMaskGenerator!");
    return *maskGen;
  }

  void drawLandmarks();

  static void drawLandmark(cv::Mat &img, cv::Point pt);

private:
  /// Path to the input image
  std::string imgPath;

  /// Path to facial landmark detector model
  std::string modelPath;

  /// Face Landmark Detector
  std::unique_ptr<FaceLandmarkDetector> detector;

  /// Face Object
  std::unique_ptr<Face> theFace;

  /// Curve Fitting Visitor
  std::unique_ptr<CurveFittingVisitor> curveFitVis;

  /// Mask Generator
  std::unique_ptr<SkinMaskGenerator> maskGen;

  /// Blemish Remover
  // std::unique_ptr<BlemishRemover> blemishRM;

  /// Attribute-aware Dynamic Guided Filter
  // std::unique_ptr<DynamicGuidedFilter> dynamicGF;

  /// Maintain a copy of the input image
  cv::Mat inputImg;

  /// Work image
  cv::Mat workImg;
};

} // namespace fabsoften

#endif
