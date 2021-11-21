#ifndef FACE_LANDMARK_DETECTOR_H
#define FACE_LANDMARK_DETECTOR_H

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

namespace fabsoften {

/**
 * @brief
 *
 */
class FaceLandmarkDetector {
public:
  explicit FaceLandmarkDetector(const std::string landmarkModelPath, const cv::Mat cvImg);
  FaceLandmarkDetector(const FaceLandmarkDetector &) = delete;
  FaceLandmarkDetector &operator=(const FaceLandmarkDetector &) = delete;

  /**
   * @brief Run the facial detector and store detected landmarks for the first detected face
   *
   */
  void detectSingleFace();

  // TODO: support multiple faces

  /**
   * @brief Return a vector of landmark positions.
   *
   */
  const std::vector<cv::Point> &getLandmarks() const { return landmarks; }

private:
  /// Work image
  dlib::cv_image<dlib::bgr_pixel> img;

  /// Face detector
  dlib::shape_predictor shapePredictor;

  /// A vector of detected facial landmarks
  std::vector<cv::Point> landmarks;
};

} // namespace fabsoften

#endif
