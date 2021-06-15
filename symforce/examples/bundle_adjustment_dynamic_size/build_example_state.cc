#include "./build_example_state.h"

#include <sym/linear_camera_cal.h>
#include <sym/posed_camera.h>
#include <sym/util/typedefs.h>
#include <symforce/opt/assert.h>
#include <symforce/util/random.h>

namespace sym {
namespace bundle_adjustment_dynamic_size {

namespace {

template <typename Scalar>
struct Correspondence {
  Eigen::Vector2d source_uv;
  Eigen::Vector2d target_uv;
  Scalar is_valid;
};

// Generate correspondences in the target camera given observations in the source camera.
template <typename Scalar>
std::vector<Correspondence<Scalar>> GenerateCorrespondences(
    const sym::PosedCamera<sym::LinearCameraCal<Scalar>>& source_cam,
    const sym::PosedCamera<sym::LinearCameraCal<Scalar>>& target_cam,
    const std::vector<std::pair<Eigen::Matrix<Scalar, 2, 1>, Scalar>> source_observations,
    std::mt19937* const gen, const Scalar epsilon = 1e-12, const Scalar noise_px = 0,
    const size_t num_outliers = 0) {
  SYM_ASSERT(gen != nullptr);
  // Create correspondence for each sample
  std::vector<Correspondence<Scalar>> correspondences;
  for (const auto& observation : source_observations) {
    const auto& source_uv = observation.first;
    const auto& inverse_range = observation.second;

    Correspondence<Scalar> correspondence;
    correspondence.source_uv = source_uv;

    if (correspondences.size() < num_outliers) {
      std::uniform_real_distribution<Scalar> uniform_dist_u(0, source_cam.ImageSize()(1));
      std::uniform_real_distribution<Scalar> uniform_dist_v(0, source_cam.ImageSize()(0));

      correspondence.target_uv =
          Eigen::Matrix<Scalar, 2, 1>(uniform_dist_u(*gen), uniform_dist_v(*gen));
      correspondence.is_valid = target_cam.InView(correspondence.target_uv, target_cam.ImageSize());
    } else {
      // Warp the point to the target
      const Eigen::Matrix<Scalar, 2, 1> target_uv_perfect = source_cam.WarpPixel(
          source_uv, inverse_range, target_cam, epsilon, &correspondence.is_valid);

      correspondence.target_uv = target_uv_perfect + noise_px * RandomNormalVector<Scalar, 2>(*gen);
    }

    correspondences.push_back(correspondence);
  }

  return correspondences;
}

// Sample pixel coords at the center of buckets with the given width, aligned with (0, 0).
template <typename Scalar>
std::vector<Eigen::Matrix<Scalar, 2, 1>> SampleRegularGrid(const Eigen::Vector2i& img_shape,
                                                           const int bucket_width_px) {
  std::vector<Eigen::Matrix<Scalar, 2, 1>> coords;

  int row = bucket_width_px / 2;
  while (row < img_shape(0)) {
    int col = bucket_width_px / 2;
    while (col < img_shape(1)) {
      coords.push_back(Eigen::Vector2i(row, col).cast<Scalar>());
      col += bucket_width_px;
    }
    row += bucket_width_px;
  }

  return coords;
}

// Given an image shape, pick a sparse but distributed set of points by
// generating a grid and then shuffling and randomly selecting from that set.
template <typename Scalar>
std::vector<Eigen::Matrix<Scalar, 2, 1>> PickSourceCoordsRandomlyFromGrid(
    const Eigen::Vector2i& image_shape, const int num_coords, std::mt19937* const gen,
    const Scalar bucket_width_px = 100) {
  SYM_ASSERT(gen != nullptr);
  // Sample some pixel coordinates in a grid
  std::vector<Eigen::Matrix<Scalar, 2, 1>> uv_samples =
      SampleRegularGrid<Scalar>(image_shape, bucket_width_px);

  // Pick randomly the required number of source pixels
  SYM_ASSERT(uv_samples.size() >= num_coords);
  std::shuffle(uv_samples.begin(), uv_samples.end(), *gen);

  std::vector<Eigen::Matrix<Scalar, 2, 1>> subset_uv_samples;
  std::copy_n(uv_samples.begin(), num_coords, std::back_inserter(subset_uv_samples));

  return subset_uv_samples;
}

// Sample a number of inverse ranges with a uniform distribution in range
// between the given values.
template <typename Scalar>
std::vector<Scalar> SampleInverseRanges(const size_t num, std::mt19937* const gen,
                                        const bool at_infinity = false, const Scalar close_m = 2.5,
                                        const Scalar far_m = 30.0) {
  SYM_ASSERT(gen != nullptr);
  std::uniform_real_distribution<Scalar> dist =
      at_infinity ? std::uniform_real_distribution<Scalar>(0, 0)
                  : std::uniform_real_distribution<Scalar>(close_m, far_m);

  std::vector<Scalar> inverse_ranges;
  for (size_t i = 0; i < num; i++) {
    inverse_ranges.push_back(1 / dist(*gen));
  }

  return inverse_ranges;
}

}  // namespace

sym::Valuesd BuildValues(std::mt19937& gen, const int num_landmarks) {
  static constexpr const double kEpsilon = 1e-10;
  static constexpr const double kNoisePx = 5;
  static constexpr const double kNumOutliers = 0;
  static constexpr const double kLandmarkRelativeRangeNoise = 0.2;
  static constexpr const double kPoseDifferenceStd = 2;
  static constexpr const double kPoseNoise = 0.1;
  static constexpr const int kNumViews = 2;

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

  values.Set({Var::VIEW, 1}, cam1.Pose().Retract(kPoseNoise * RandomNormalVector<double, 6>(gen)));
  values.Set({Var::CALIBRATION, 1}, cam1.Calibration().Data());

  // Pose priors
  // First, create the 0-weight priors:
  for (int i = 0; i < kNumViews; i++) {
    for (int j = 0; j < kNumViews; j++) {
      values.Set({Var::POSE_PRIOR_R, i, j}, sym::Rot3d());
      values.Set({Var::POSE_PRIOR_T, i, j}, Vector3d());
      values.Set({Var::POSE_PRIOR_WEIGHT, i, j}, 0.0);
      values.Set({Var::POSE_PRIOR_SIGMAS, i, j}, Vector6d::Zero());
    }
  }

  // Now, the actual prior between 0 and 1
  static constexpr const double kPosePriorNoise = 0.3;
  values.Set({Var::POSE_PRIOR_T, 0, 1},
             sym::LieGroupOps<Vector3d>::Retract(
                 sym::GroupOps<Vector3d>::Between(view1.Position(), view0.Position()),
                 kPosePriorNoise * RandomNormalVector<double, 3>(gen), kEpsilon));
  values.Set({Var::POSE_PRIOR_R, 0, 1},
             view1.Rotation()
                 .Between(view0.Rotation())
                 .Retract(kPosePriorNoise * RandomNormalVector<double, 3>(gen)));
  values.Set({Var::POSE_PRIOR_WEIGHT, 0, 1}, 1.0);
  values.Set({Var::POSE_PRIOR_SIGMAS, 0, 1}, kPosePriorNoise * Vector6d::Ones());

  // Sample random correspondences
  const std::vector<Eigen::Vector2d> source_coords =
      PickSourceCoordsRandomlyFromGrid<double>(image_shape, num_landmarks, &gen);
  const std::vector<double> source_inverse_ranges =
      SampleInverseRanges<double>(num_landmarks, &gen);

  std::vector<std::pair<Eigen::Matrix<double, 2, 1>, double>> source_observations;
  std::transform(
      source_coords.begin(), source_coords.end(), source_inverse_ranges.begin(),
      std::back_inserter(source_observations),
      [](const auto& uv, const double inverse_range) { return std::make_pair(uv, inverse_range); });
  const std::vector<Correspondence<double>> correspondences = GenerateCorrespondences(
      cam0, cam1, source_observations, &gen, kEpsilon, kNoisePx, kNumOutliers);

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

optimizer_params_t OptimizerParams() {
  optimizer_params_t params;
  params.iterations = 50;
  params.verbose = false;
  params.initial_lambda = 1.0;
  params.lambda_up_factor = 10.0;
  params.lambda_down_factor = 1 / 10.0;
  params.lambda_lower_bound = 1.0e-8;
  params.lambda_upper_bound = 1000000.0;
  params.early_exit_min_reduction = 1.0e-6;
  params.use_unit_damping = true;
  params.use_diagonal_damping = false;
  params.keep_max_diagonal_damping = false;
  params.diagonal_damping_min = 1e-6;
  params.enable_bold_updates = false;
  return params;
}

}  // namespace bundle_adjustment_dynamic_size
}  // namespace sym
