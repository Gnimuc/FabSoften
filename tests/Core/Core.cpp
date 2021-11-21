#include "Core.h"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>

TEST_CASE("Core", "[Beautifier]") {
  std::filesystem::path projectSrcDir(UNITTEST_PROJECT_DIR);
  auto assetsDir = projectSrcDir / "assets";
  auto modelsDir = projectSrcDir / "models";
  const auto testImgPath = assetsDir / "pexels-aadil-2598024.jpg";
  const auto testModelPath = modelsDir / "shape_predictor_68_face_landmarks.dat";
  fabsoften::Beautifier bf(testImgPath.string(), testModelPath.string());
  bf.createFaceLandmarkDetector();
  SECTION("Face Landmark Detector") { REQUIRE(bf.hasFaceLandmarkDetector()); }
}
