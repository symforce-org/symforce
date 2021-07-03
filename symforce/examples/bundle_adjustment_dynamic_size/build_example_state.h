#pragma once

#include <symforce/opt/values.h>

#include <lcmtypes/sym/optimizer_params_t.hpp>

namespace sym {
namespace bundle_adjustment_dynamic_size {

static constexpr const double kEpsilon = 1e-10;
static constexpr const int kNumViews = 2;
static constexpr const int kNumLandmarks = 20;

enum Var : char {
  VIEW = 'v',                  // Pose3d
  CALIBRATION = 'c',           // Vector4d
  POSE_PRIOR_T = 'T',          // Pose3d
  POSE_PRIOR_SQRT_INFO = 's',  // Matrix6d
  LANDMARK = 'l',              // Scalar
  LANDMARK_PRIOR = 'P',        // Scalar
  LANDMARK_PRIOR_SIGMA = 'S',  // Scalar
  MATCH_SOURCE_COORDS = 'm',   // Vector2d
  MATCH_TARGET_COORDS = 'M',   // Vector2d
  MATCH_WEIGHT = 'W',          // Scalar
  GNC_MU = 'u',                // Scalar
  GNC_SCALE = 'C',             // Scalar
  EPSILON = 'e',               // Scalar
};

sym::Valuesd BuildValues(std::mt19937& gen, const int num_landmarks);

}  // namespace bundle_adjustment_dynamic_size
}  // namespace sym
