#ifndef SKIN_MASK_GENERATOR_H
#define SKIN_MASK_GENERATOR_H

#include "fabsoften/FaceRegion.h"

namespace fabsoften {

/// SkinMaskOptions - Options for controlling the behavior of the skin mask generator.
class SkinMaskOptions {
public:
  /// Enable the lower and upper face region
  unsigned EnableFace : 1;

  /// Enable the mouth region
  unsigned EnableMouth : 1;

  /// Enable the eye region
  unsigned EnableEye : 1;

  /// Enable the eye brow region
  unsigned EnableEyeBrow : 1;

  /// Enable the cheek zone
  unsigned EnableCheek : 1;

  /// Control the size of face region
  float faceScaleRate;

public:
  SkinMaskOptions()
      : EnableFace(true), EnableMouth(true), EnableEye(true), EnableEyeBrow(true),
        EnableCheek(false), faceScaleRate(0.85f) {}
};

/// \brief Class for generating skin masks.
class SkinMaskGenerator {
public:
  SkinMaskOptions opts;

public:
  explicit SkinMaskGenerator(SkinMaskOptions op = SkinMaskOptions()) : opts(op) {}

  /// \brief Estimate an ellipse mask that can cover both the upper and lower face region
  ///
  /// \param face The face object.
  /// \param size Mask size.
  void generateEllipseFaceMask(Face &face, cv::Size size);

  /// \brief Generate a binary mask without refinement
  ///
  /// This method will call \ref generateEllipseFaceMask firstly to draw a face region, then
  /// mask those not needed regions according to \ref SkinMaskOptions.
  ///
  /// \param [in] face The face object.
  /// \param [out] dstMask A single channel mask of eltype `CV_8UC1`.
  void generateBinaryMask(Face &face, cv::Mat dstMask);

  void copyCurrentMaskTo(cv::Mat &mask) const { maskCur.copyTo(mask); }

private:
  cv::Mat maskCur;
};

} // namespace fabsoften

#endif
