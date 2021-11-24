/// \file SkinMaskGenerator.cpp
/// \brief SkinMaskGenerator Implmentation
///

#include "fabsoften/SkinMaskGenerator.h"

using namespace fabsoften;

void SkinMaskGenerator::generateEllipseFaceMask(Face &face, cv::Size size) {
  std::array<cv::Point2f, nJaw + 1> ellipsePts;
  const auto &curves = face.getCurves();
  const auto &jawPts = (*curves)["jaw"];
  for (size_t i = 0; const auto &pt : jawPts) {
    ellipsePts[i] = pt;
    i++;
  }

  // Guess a point located in the upper face region
  // Pb: 8 (bottom of jaw)
  // Pt: 27 (top of nose)
  const auto &landmarks = face.getLandmarks();
  const auto &Pb = (*landmarks)[8];
  const auto &Pt = (*landmarks)[27];
  const auto xUp = Pb.x;
  const auto yUp = Pt.y - opts.faceScaleRate * abs(Pb.y - Pt.y);
  ellipsePts[nJaw] = cv::Point(xUp, yUp);

  // Fit ellipse
  auto box = cv::fitEllipseDirect(ellipsePts);

  // Create a mask with black color in all region out side of the ellipse
  maskCur = cv::Mat::zeros(size, CV_8UC1);
  cv::ellipse(maskCur, box, cv::Scalar(255), /*thickness=*/-1, cv::FILLED);
}

void SkinMaskGenerator::generateBinaryMask(Face &face, cv::Mat dstMask) {
  generateEllipseFaceMask(face, dstMask.size());

  // TODO: make sure those curves are available in `face`
  const auto &curves = face.getCurves();

  if (opts.EnableEye) {
    cv::fillConvexPoly(maskCur, (*curves)["leftEye"], cv::Scalar(0), cv::LINE_AA);
    cv::fillConvexPoly(maskCur, (*curves)["rightEye"], cv::Scalar(0), cv::LINE_AA);
  }

  if (opts.EnableEyeBrow) {
    const auto offset = dstMask.cols * opts.BrowOffsetRate;
    const auto thickness = dstMask.rows * opts.BrowThicknessRate;

    std::vector<cv::Point> pts;
    for (auto &pt : (*curves)["leftEyeBrow"])
      pts.push_back(cv::Point(pt.x, pt.y + offset));

    cv::polylines(maskCur, pts, /*isClosed=*/false, cv::Scalar(0), thickness, cv::LINE_AA);

    pts.clear();
    for (auto &pt : (*curves)["rightEyeBrow"])
      pts.push_back(cv::Point(pt.x, pt.y + offset));

    cv::polylines(maskCur, pts, /*isClosed=*/false, cv::Scalar(0), thickness, cv::LINE_AA);
  }

  if (opts.EnableMouth) {
    cv::fillPoly(maskCur, (*curves)["mouth"], cv::Scalar(0), cv::LINE_AA);
  }

  if (opts.EnableCheek) {
    cv::fillConvexPoly(maskCur, (*curves)["leftCheek"], cv::Scalar(255), cv::LINE_AA);
    cv::fillConvexPoly(maskCur, (*curves)["rightCheek"], cv::Scalar(255), cv::LINE_AA);
  }

  const auto N = opts.ErodingSize;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(N, N));
  cv::morphologyEx(maskCur, maskCur, cv::MORPH_ERODE, element);

  copyCurrentMaskTo(dstMask);
}
