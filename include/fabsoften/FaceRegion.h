#ifndef FACE_REGION_H
#define FACE_REGION_H

#include <array>
#include <map>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <tinysplinecxx.h>
#include <utility>
#include <vector>

namespace fabsoften {
// TODO: find a generic way to support other models

// clang-format off
// The 68 facial landmark from the iBUG 300-W dataset(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
// Jaw:              0-16 (lower face boundary)
// Right eyebrow:   17-21
// Left eyebrow:    22-26 
// Nose:            27-35
// Right eye:       36-41
// Left eye:        42-47
// Mouth:           48-59 (inner boundary: 60-67)
// Right cheek:      1  2  3  4 31 32 33 30 29 28 39 40 41 36  1
// Left cheek:      15 14 13 12 35 34 33 30 29 28 42 47 46 45 15
// clang-format on
constexpr auto nJaw = 17;

template <int... Idxs>
static constexpr std::array<int, sizeof...(Idxs)>
ExpandToArray(std::integer_sequence<int, Idxs...>) {
  return {(static_cast<int>(Idxs))...};
}

static constexpr auto jawIdxs = ExpandToArray(std::make_integer_sequence<int, nJaw>{});

constexpr auto nEye = 6;
static constexpr std::array<int, nEye> leftEyeIdxs{42, 43, 44, 45, 46, 47};
static constexpr std::array<int, nEye> rightEyeIdxs{36, 37, 38, 39, 40, 41};

constexpr auto nEyeBrow = 5;
static constexpr std::array<int, nEyeBrow> leftEyeBrowIdxs{22, 23, 24, 25, 26};
static constexpr std::array<int, nEyeBrow> rightEyeBrowIdxs{17, 18, 19, 20, 21};

constexpr auto nNose = 9;
static constexpr std::array<int, nNose> noseIdxs{27, 28, 29, 30, 31, 32, 33, 34, 35};

constexpr auto nMouth = 12;
consteval auto getMouthIdxs() {
  std::array<int, nMouth> mouthIdxs{};
  for (const auto i : std::views::iota(0) | std::views::take(nMouth))
    mouthIdxs[i] = i;
  return mouthIdxs;
}
static constexpr auto mouthIdxs = getMouthIdxs();

constexpr auto nCheek = 15;
static constexpr std::array<int, nCheek> rightCheekIdxs{1,  2,  3,  4,  31, 32, 33, 30,
                                                        29, 28, 39, 40, 41, 36, 1};
static constexpr std::array<int, nCheek> leftCheekIdxs{15, 14, 13, 12, 35, 34, 33, 30,
                                                       29, 28, 42, 47, 46, 45, 15};

class Jaw;
class Eye;
class EyeBrow;
class Nose;
class Mouth;
class Cheek;

struct FaceRegionVisitor {
  virtual void handleRegion(Jaw &jaw) const = 0;
  virtual void handleRegion(Eye &eye) const = 0;
  virtual void handleRegion(EyeBrow &brow) const = 0;
  virtual void handleRegion(Nose &nose) const = 0;
  virtual void handleRegion(Mouth &mouth) const = 0;
  virtual void handleRegion(Cheek &cheek) const = 0;
};

struct FaceRegion {
  virtual void dispatch(const FaceRegionVisitor &visitor) = 0;
};

class Jaw : public FaceRegion {
public:
  explicit Jaw(std::array<int, nJaw> x = jawIdxs) : idxs(x) {}

  const std::array<int, nJaw> &getIdxs() const { return idxs; };

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

private:
  std::array<int, nJaw> idxs;
};

class Eye : public FaceRegion {
public:
  enum class EyeKind { Left, Right };
  explicit Eye(std::array<int, nEye> x) : idxs(x) {}

  virtual const EyeKind getKind() const = 0;

  const std::array<int, nEye> &getIdxs() const { return idxs; };

private:
  std::array<int, nEye> idxs;
};

class LeftEye : public Eye {
public:
  explicit LeftEye(std::array<int, nEye> x = leftEyeIdxs) : Eye(x), kind(EyeKind::Left) {}

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

  const EyeKind getKind() const override { return kind; }

private:
  EyeKind kind;
};

class RightEye : public Eye {
public:
  explicit RightEye(std::array<int, nEye> x = rightEyeIdxs)
      : Eye(x), kind(EyeKind::Right) {}

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

  const EyeKind getKind() const override { return kind; }

private:
  EyeKind kind;
};

class EyeBrow : public FaceRegion {
public:
  enum class EyeBrowKind { Left, Right };
  explicit EyeBrow(std::array<int, nEyeBrow> x) : idxs(x) {}

  virtual const EyeBrowKind getKind() const = 0;

  const std::array<int, nEyeBrow> &getIdxs() const { return idxs; };

private:
  std::array<int, nEyeBrow> idxs;
};

class LeftEyeBrow : public EyeBrow {
public:
  explicit LeftEyeBrow(std::array<int, nEyeBrow> x = leftEyeBrowIdxs)
      : EyeBrow(x), kind(EyeBrowKind::Left) {}

  const EyeBrowKind getKind() const override { return kind; }

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

private:
  EyeBrowKind kind;
};

class RightEyeBrow : public EyeBrow {
public:
  explicit RightEyeBrow(std::array<int, nEyeBrow> x = rightEyeBrowIdxs)
      : EyeBrow(x), kind(EyeBrowKind::Right) {}

  const EyeBrowKind getKind() const override { return kind; }

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

private:
  EyeBrowKind kind;
};

class Nose : public FaceRegion {
public:
  explicit Nose(std::array<int, nNose> x = noseIdxs) : idxs(x) {}

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

  const std::array<int, nNose> &getIdxs() const { return idxs; };

private:
  std::array<int, nNose> idxs;
};

class Mouth : public FaceRegion {
public:
  explicit Mouth(std::array<int, nMouth> x = mouthIdxs) : idxs(x) {}

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

  const std::array<int, nMouth> &getIdxs() const { return idxs; };

private:
  std::array<int, nMouth> idxs;
};

class Cheek : public FaceRegion {
public:
  enum class CheekKind { Left, Right };
  explicit Cheek(std::array<int, nCheek> x) : idxs(x) {}

  virtual const CheekKind getKind() const = 0;

  const std::array<int, nCheek> &getIdxs() const { return idxs; };

private:
  std::array<int, nCheek> idxs;
};

class LeftCheek : public Cheek {
public:
  explicit LeftCheek(std::array<int, nCheek> x = leftCheekIdxs)
      : Cheek(x), kind(CheekKind::Left) {}

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

  const CheekKind getKind() const override { return kind; }

private:
  CheekKind kind;
};

class RightCheek : public Cheek {
public:
  explicit RightCheek(std::array<int, nCheek> x = rightCheekIdxs)
      : Cheek(x), kind(CheekKind::Right) {}

  void dispatch(const FaceRegionVisitor &visitor) override { visitor.handleRegion(*this); }

  const CheekKind getKind() const override { return kind; }

private:
  CheekKind kind;
};

/**
 * @brief Utility class for storing facial region info
 *
 */
class Face {
  using FaceRegionMap = std::map<std::string, std::unique_ptr<FaceRegion>>;
  using CurveMap = std::map<std::string, std::vector<cv::Point>>;

public:
  Face();

  const FaceRegionMap &getRegions() const { return regions; }

  const CurveMap &getCurves() const { return curves; }
  void setCurves(const CurveMap &x) { curves = x; }

private:
  FaceRegionMap regions;
  CurveMap curves;
};

/**
 * @brief Curve Fitting Options
 *
 * For now, it stores the length of the curve of each facial region.
 * If the value <= 0, then this facial region will be ignore when traversing facial regions.
 *
 */
class CurveFittingOptions {
public:
  int nJaw;
  int nEye;
  int nEyeBrow;
  int nNose;
  int nMouth;
  int nCheek;

public:
  CurveFittingOptions()
      : nJaw(-1), nEye(25), nEyeBrow(50), nNose(-1), nMouth(40), nCheek(-1) {}
};

class CurveFittingVisitor : public FaceRegionVisitor {
  using CurveMap = std::map<std::string, std::vector<cv::Point>>;

public:
  CurveFittingOptions opts;

public:
  CurveFittingVisitor(const std::vector<cv::Point> &landmarks,
                      CurveFittingOptions options = CurveFittingOptions())
      : landmarks(landmarks), opts(options) {}

  void handleRegion(Jaw &jaw) const override;
  void handleRegion(Eye &eye) const override;
  void handleRegion(EyeBrow &brow) const override;
  void handleRegion(Nose &nose) const override;
  void handleRegion(Mouth &mouth) const override;
  void handleRegion(Cheek &cheek) const override;

  void fit(Face &face) const {
    for (const auto &[key, region] : face.getRegions())
      region->dispatch(*this);

    // Store results
    face.setCurves(curves);
  }

private:
  std::vector<cv::Point> landmarks;
  mutable CurveMap curves;
  mutable std::vector<tinyspline::real> knots;
};

} // namespace fabsoften

#endif
