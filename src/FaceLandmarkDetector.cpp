/**
 * @file FaceLandmarkDetector.cpp
 * @brief FaceLandmarkDetector Implmentation
 */

#include "fabsoften/FaceLandmarkDetector.h"
#include <ranges>

using namespace fabsoften;

FaceLandmarkDetector::FaceLandmarkDetector(const std::string landmarkModelPath,
                                           const cv::Mat cvImg) {
  // Load model
  dlib::deserialize(landmarkModelPath) >> shapePredictor;
  // Bridge OpenCV and dlib
  img = dlib::cv_image<dlib::bgr_pixel>(cvImg);
  // Allocate a vector for storing landmarks
  landmarks = std::make_shared<PointVec>();
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
