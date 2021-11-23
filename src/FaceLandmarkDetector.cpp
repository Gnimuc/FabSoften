/// \file FaceLandmarkDetector.cpp
/// \brief FaceLandmarkDetector Implmentation
///
/// TODO: Add implementation for multiple facial landmark detection.

#include "fabsoften/FaceLandmarkDetector.h"
#include <ranges>

using namespace fabsoften;

FaceLandmarkDetector::FaceLandmarkDetector(const std::string landmarkModelPath,
                                           const cv::Mat cvImg)
    : landmarks(std::make_shared<PointVec>()) {
  // Load model
  dlib::deserialize(landmarkModelPath) >> shapePredictor;
  // Bridge OpenCV and dlib
  img = dlib::cv_image<dlib::bgr_pixel>(cvImg);
}

void FaceLandmarkDetector::detectSingleFace() {
  landmarks->clear();
  auto faceDetector = dlib::get_frontal_face_detector();
  auto faces = faceDetector(img);
  assert(!faces.empty() && "Failed to detect faces!");
  // Only check the first detected face
  const auto shape = shapePredictor(img, faces[0]);
  for (const auto i : std::views::iota(0) | std::views::take(shape.num_parts())) {
    const auto &pt = shape.part(i);
    landmarks->push_back(cv::Point(pt.x(), pt.y()));
  }
}
