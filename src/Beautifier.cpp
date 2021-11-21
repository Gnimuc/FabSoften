/**
 * @file Beautifier.cpp
 * @brief Beautifier Implmentation
 */

#include "fabsoften/Beautifier.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>

using namespace fabsoften;

Beautifier::Beautifier(const std::string inputImgPath, const std::string landmarkModelPath)
    : imgPath(inputImgPath), modelPath(landmarkModelPath) {
  inputImg = cv::imread(inputImgPath);
  assert(!inputImg.empty() && "Could not load image!");
  workImg = inputImg.clone();
}

void Beautifier::soften() {}

void Beautifier::downsampling() { cv::pyrDown(workImg, workImg); }

void Beautifier::createFaceLandmarkDetector() {
  detector = std::make_unique<FaceLandmarkDetector>(modelPath, workImg);
}

void Beautifier::setFaceLandmarkDetector(
    std::unique_ptr<FaceLandmarkDetector> newDetector) {
  detector = std::move(newDetector);
}

void Beautifier::drawLandmark(cv::Mat &img, cv::Point pt) {
  const auto color = cv::Scalar(0, 255, 255);
  const auto radius = img.rows / 300;
  const auto thickness = radius / 2;
  cv::circle(img, pt, radius, color, thickness);
}
