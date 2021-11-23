#ifndef FACE_LANDMARK_DETECTOR_H
#define FACE_LANDMARK_DETECTOR_H

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

namespace fabsoften {

/// Class to detect and generate facial landmarks.
class FaceLandmarkDetector {
  using PointVec = std::vector<cv::Point>;

public:
  explicit FaceLandmarkDetector(const std::string landmarkModelPath, const cv::Mat cvImg);

  /// Run the facial detector and store detected landmarks for the first detected face.
  void detectSingleFace();

  // TODO: support multiple faces

  /// Return a shared pointer to a vector of landmark positions.
  std::shared_ptr<PointVec> getLandmarks() const { return landmarks; }

private:
  /// Work image
  dlib::cv_image<dlib::bgr_pixel> img;

  /// Face detector
  dlib::shape_predictor shapePredictor;

  /// A vector of detected facial landmarks
  std::shared_ptr<PointVec> landmarks;
};

} // namespace fabsoften

#endif
