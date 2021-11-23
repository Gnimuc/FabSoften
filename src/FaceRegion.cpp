/// \file FaceRegion.cpp
/// \brief FaceRegion Implmentation

#include "fabsoften/FaceRegion.h"
#include "ranges"

using namespace fabsoften;

Face::Face(std::shared_ptr<PointVec> landmarks)
    : landmarks(std::move(landmarks)), regions(std::make_shared<FaceRegionMap>()),
      curves(std::make_shared<CurveMap>()) {
  (*regions)["Jaw"] = std::make_unique<Jaw>();
  (*regions)["LeftEye"] = std::make_unique<LeftEye>();
  (*regions)["RightEye"] = std::make_unique<RightEye>();
  (*regions)["LeftEyeBrow"] = std::make_unique<LeftEyeBrow>();
  (*regions)["RightEyeBrow"] = std::make_unique<RightEyeBrow>();
  (*regions)["LeftCheek"] = std::make_unique<LeftCheek>();
  (*regions)["RightCheek"] = std::make_unique<RightCheek>();
}

void CurveFittingVisitor::handleRegion(Jaw &jaw) const {
  if (opts.nJaw <= 0)
    return;

  // TODO: add impl
}

void CurveFittingVisitor::handleRegion(Eye &eye) const {
  if (opts.nEye <= 0)
    return;

  knots.clear();

  // Make a curve
  for (const auto &idx : eye.getIdxs()) {
    const auto &point = (*landmarks)[idx];
    knots.push_back(point.x);
    knots.push_back(point.y);
  }

  // Make a closed curve
  knots.push_back(knots[0]);
  knots.push_back(knots[1]);

  // Interpolate the curve
  auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);

  // Sampling
  const auto num = opts.nEye;
  const auto name = eye.getKind() == Eye::EyeKind::Left ? "leftEye" : "rightEye";
  for (const auto i : std::views::iota(0, num)) {
    const auto net = spline(1.0 / num * i);
    const auto result = net.result();
    const auto x = result[0], y = result[1];
    (*curves)[name].push_back(cv::Point(x, y));
  }
}

void CurveFittingVisitor::handleRegion(EyeBrow &brow) const {
  if (opts.nEyeBrow <= 0)
    return;

  knots.clear();

  // Make a curve
  for (const auto &idx : brow.getIdxs()) {
    const auto &point = (*landmarks)[idx];
    knots.push_back(point.x);
    knots.push_back(point.y);
  }

  // Interpolate the curve
  auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);

  // Sampling
  const auto num = opts.nEyeBrow;
  auto name = brow.getKind() == EyeBrow::EyeBrowKind::Left ? "leftEyeBrow" : "rightEyeBrow";
  for (const auto i : std::views::iota(0, num)) {
    const auto net = spline(1.0 / num * i);
    const auto result = net.result();
    const auto x = result[0], y = result[1];
    (*curves)[name].push_back(cv::Point(x, y));
  }
}

void CurveFittingVisitor::handleRegion(Nose &nose) const {
  if (opts.nNose <= 0)
    return;

  // TODO: add impl
}

void CurveFittingVisitor::handleRegion(Mouth &mouth) const {
  if (opts.nMouth <= 0)
    return;

  knots.clear();

  // Make a curve
  for (const auto &idx : mouth.getIdxs()) {
    const auto &point = (*landmarks)[idx];
    knots.push_back(point.x);
    knots.push_back(point.y);
  }

  // Make a closed curve
  knots.push_back(knots[0]);
  knots.push_back(knots[1]);

  // Interpolate the curve
  auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);

  // Sampling
  const auto num = opts.nMouth;
  for (const auto i : std::views::iota(0, num)) {
    const auto net = spline(1.0 / num * i);
    const auto result = net.result();
    const auto x = result[0], y = result[1];
    (*curves)["mouth"].push_back(cv::Point(x, y));
  }
}

void CurveFittingVisitor::handleRegion(Cheek &cheek) const {
  if (opts.nCheek <= 0)
    return;

  knots.clear();

  // Make a curve
  for (const auto &idx : cheek.getIdxs()) {
    const auto &point = (*landmarks)[idx];
    knots.push_back(point.x);
    knots.push_back(point.y);
  }

  // Interpolate the curve
  auto spline = tinyspline::BSpline::interpolateCubicNatural(knots, 2);

  // Sampling
  const auto num = opts.nCheek;
  const auto name = cheek.getKind() == Cheek::CheekKind::Left ? "leftCheek" : "rightCheek";
  for (const auto i : std::views::iota(0, num)) {
    const auto net = spline(1.0 / num * i);
    const auto result = net.result();
    const auto x = result[0], y = result[1];
    (*curves)[name].push_back(cv::Point(x, y));
  }
}
