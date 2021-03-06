/// \file Beautifier.cpp
/// \brief Beautifier Implmentation
///

#include "fabsoften/Beautifier.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>

using namespace fabsoften;

Beautifier::Beautifier(const std::string inputImgPath, const std::string landmarkModelPath)
    : imgPath(inputImgPath), modelPath(landmarkModelPath),
      curveFitVis(std::make_unique<CurveFittingVisitor>()),
      maskGen(std::make_unique<SkinMaskGenerator>()),
      blemishRM(std::make_unique<BlemishRemover>()), gf(std::make_unique<GuidedFilter>()) {
  inputImg = cv::imread(inputImgPath);
  assert(!inputImg.empty() && "Could not load image!");
  workImg = inputImg.clone();
}

void Beautifier::soften() {
  if (!hasFaceLandmarkDetector())
    createFaceLandmarkDetector();

  if (!hasFace())
    createFace();

  interpolateLandmarks();

  if (maskImg.size() != workImg.size())
    maskImg = cv::Mat::zeros(workImg.size(), CV_8UC1);

  drawBinaryMask(maskImg);

  cv::Mat maskChannels[3] = {maskImg, maskImg, maskImg};
  cv::merge(maskChannels, 3, maskImg3C);

  cv::bitwise_not(maskImg3C, /*mask=*/tmpImg);
  cv::bitwise_and(workImg, /*mask=*/tmpImg, /*background=*/tmpImg);
  workImg.copyTo(tmpImg2);

  concealBlemish(maskImg);

  cv::bitwise_and(workImg, maskImg3C, workImg);
  cv::add(workImg, tmpImg, workImg);

  applyADF(maskImg, workImg, /*original image=*/tmpImg2, tmpImg);
  tmpImg.convertTo(outputImg, CV_8U);
}

void Beautifier::downsampling() { cv::pyrDown(workImg, workImg); }

void Beautifier::createFaceLandmarkDetector() {
  detector = std::make_unique<FaceLandmarkDetector>(modelPath, workImg);
}

void Beautifier::setFaceLandmarkDetector(
    std::unique_ptr<FaceLandmarkDetector> newDetector) {
  detector = std::move(newDetector);
}

void Beautifier::createFace() {
  assert(hasFaceLandmarkDetector() && "No FaceLandmarkDetector found in the Beautifier!");

  // Run detector if there is no available results
  if (detector->getLandmarks()->size() == 0)
    detector->detectSingleFace();

  theFace = std::make_unique<Face>(detector->getLandmarks());
}

void Beautifier::encode() {
  std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 100};
  cv::imencode(".jpg", outputImg, buf, params);
}

void Beautifier::drawLandmarks(cv::Mat &img, bool interpolated) {
  if (interpolated) {
    const auto curves = theFace->getCurves();
    for (const auto &map = *(curves); auto &[key, pts] : map)
      if (key != "leftCheek" && key != "rightCheek")
        for (const auto &pt : pts)
          Beautifier::drawLandmark(img, pt);
  } else {
    for (const auto landmarks = theFace->getLandmarks(); auto &pt : *landmarks)
      Beautifier::drawLandmark(img, pt);
  }
}

void Beautifier::drawLandmark(cv::Mat &img, const cv::Point pt) {
  const auto color = cv::Scalar(0, 255, 255);
  const auto radius = img.rows / 300;
  const auto thickness = radius / 2;
  cv::circle(img, pt, radius, color, thickness);
}
