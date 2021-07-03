#include "./build_example_state.h"

#include <sym/linear_camera_cal.h>
#include <sym/posed_camera.h>
#include <sym/util/typedefs.h>
#include <symforce/opt/assert.h>

#include "../example_utils/example_state_helpers.h"

namespace sym {
namespace bundle_adjustment_dynamic_size {

sym::Valuesd BuildValues(std::mt19937& gen, const int num_landmarks) {
  static constexpr const double kReprojectionErrorGncScale = 10;
  static constexpr const double kNoisePx = 5;
  static constexpr const double kNumOutliers = 0;
  static constexpr const double kLandmarkRelativeRangeNoise = 0.2;
  static constexpr const double kPoseDifferenceStd = 2;
  static constexpr const double kPoseNoise = 0.1;

  sym::Valuesd values;

  values.Set({Var::EPSILON}, kEpsilon);
  values.Set({Var::GNC_SCALE}, kReprojectionErrorGncScale);
  values.Set(Var::GNC_MU, 0.0);

  // Build two views, with similar but not identical poses looking into the same general area and
  // identical calibrations
  const sym::Pose3d view0 = Random<sym::Pose3d>(gen);
  Vector6d perturbation;
  perturbation << 0.1, -0.2, 0.1, 2.1, 0.4, -0.2;
  const sym::Pose3d view1 =
      view0.Retract(perturbation * std::normal_distribution<double>(0, kPoseDifferenceStd)(gen));

  const Eigen::Vector2i image_shape(1280, 720);
  const sym::LinearCameraCald camera_cal(Eigen::Vector2d(740, 740), Eigen::Vector2d(639.5, 359.5));
  const sym::PosedCamera<sym::LinearCameraCald> cam0(view0, camera_cal, image_shape);
  const sym::PosedCamera<sym::LinearCameraCald> cam1(view1, camera_cal, image_shape);

  values.Set({Var::VIEW, 0}, cam0.Pose());
  values.Set({Var::CALIBRATION, 0}, cam0.Calibration().Data());

  values.Set({Var::VIEW, 1}, cam1.Pose().Retract(kPoseNoise * Random<Vector6d>(gen)));
  values.Set({Var::CALIBRATION, 1}, cam1.Calibration().Data());

  // Pose priors
  // First, create the 0-weight priors:
  for (int i = 0; i < kNumViews; i++) {
    for (int j = 0; j < kNumViews; j++) {
      values.Set({Var::POSE_PRIOR_T, i, j}, sym::Pose3d());
      values.Set({Var::POSE_PRIOR_SQRT_INFO, i, j}, sym::Matrix66d::Zero());
    }
  }

  // Now, the actual prior between 0 and 1
  static constexpr const double kPosePriorNoise = 0.3;
  values.Set({Var::POSE_PRIOR_T, 0, 1},
             view1.Between(view0).Retract(kPosePriorNoise * Random<Vector6d>(gen)));
  values.Set({Var::POSE_PRIOR_SQRT_INFO, 0, 1}, sym::Matrix66d::Identity() / kPosePriorNoise);

  // Sample random correspondences
  const std::vector<Eigen::Vector2d> source_coords =
      example_utils::PickSourceCoordsRandomlyFromGrid<double>(image_shape, num_landmarks, gen);
  const std::vector<double> source_inverse_ranges =
      example_utils::SampleInverseRanges<double>(num_landmarks, gen);

  std::vector<std::pair<Eigen::Matrix<double, 2, 1>, double>> source_observations;
  std::transform(
      source_coords.begin(), source_coords.end(), source_inverse_ranges.begin(),
      std::back_inserter(source_observations),
      [](const auto& uv, const double inverse_range) { return std::make_pair(uv, inverse_range); });
  const std::vector<example_utils::Correspondence<double>> correspondences =
      example_utils::GenerateCorrespondences(cam0, cam1, source_observations, gen, kEpsilon,
                                             kNoisePx, kNumOutliers);

  const auto clamp = [](double x, double min, double max) {
    return (x > max) ? max : ((x < min) ? min : x);
  };

  // Fill matches and landmarks for each correspondence
  std::normal_distribution<double> range_normal_dist(0, kLandmarkRelativeRangeNoise);
  for (int i = 0; i < num_landmarks; i++) {
    const double source_range = 1 / source_inverse_ranges[i];
    const double range_perturbation = clamp(1 + range_normal_dist(gen), 0.5, 2.0);
    values.Set({Var::LANDMARK, i}, 1 / (source_range * range_perturbation));

    const auto& correspondence = correspondences[i];

    values.Set({Var::MATCH_SOURCE_COORDS, 1, i}, correspondence.source_uv);
    values.Set({Var::MATCH_TARGET_COORDS, 1, i}, correspondence.target_uv);
    values.Set({Var::MATCH_WEIGHT, 1, i}, correspondence.is_valid);
    values.Set({Var::LANDMARK_PRIOR, 1, i}, source_inverse_ranges[i]);
    values.Set({Var::LANDMARK_PRIOR_SIGMA, 1, i}, 100.0);
  }

  return values;
}

}  // namespace bundle_adjustment_dynamic_size
}  // namespace sym
