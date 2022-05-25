/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <vector>

#include <spdlog/spdlog.h>

#include <symforce/opt/assert.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/tic_toc.h>

#include "./common.h"
#include "./gen/keys.h"
#include "./gen/matching_factor.h"
#include "./gen/odometry_factor.h"

namespace robot_3d_localization {

/**
 * Creates a factor for a measurement of the position of landmark j from pose i
 */
template <typename Scalar>
sym::Factor<Scalar> CreateMatchingFactor(const int i, const int j) {
  return sym::Factor<Scalar>::Hessian(sym::MatchingFactor<Scalar>,
                                      {sym::Key::WithSuper(sym::Keys::WORLD_T_BODY, i),
                                       sym::Key::WithSuper(sym::Keys::WORLD_T_LANDMARK, j),
                                       {sym::Keys::BODY_T_LANDMARK_MEASUREMENTS.Letter(), i, j},
                                       sym::Keys::MATCHING_SIGMA},
                                      {sym::Key::WithSuper(sym::Keys::WORLD_T_BODY, i)});
}

/**
 * Creates a factor for the odometry measurement between poses i and i + 1
 */
template <typename Scalar>
sym::Factor<Scalar> CreateOdometryFactor(const int i) {
  return sym::Factor<Scalar>::Hessian(
      sym::OdometryFactor<Scalar>,
      {sym::Key::WithSuper(sym::Keys::WORLD_T_BODY, i),
       sym::Key::WithSuper(sym::Keys::WORLD_T_BODY, i + 1),
       sym::Key::WithSuper(sym::Keys::ODOMETRY_RELATIVE_POSE_MEASUREMENTS, i),
       sym::Keys::ODOMETRY_DIAGONAL_SIGMAS, sym::Keys::EPSILON},
      {sym::Key::WithSuper(sym::Keys::WORLD_T_BODY, i),
       sym::Key::WithSuper(sym::Keys::WORLD_T_BODY, i + 1)});
}

/**
 * Builds a list of factors for the given numbers of poses and landmarks (assuming every pose
 * observes all landmarks)
 */
template <typename Scalar>
std::vector<sym::Factor<Scalar>> BuildDynamicFactors(const int num_poses, const int num_landmarks) {
  std::vector<sym::Factor<Scalar>> factors;

  for (int i = 0; i < num_poses; i++) {
    for (int j = 0; j < num_landmarks; j++) {
      factors.push_back(CreateMatchingFactor<Scalar>(i, j));
    }
  }

  for (int i = 0; i < num_poses - 1; i++) {
    factors.push_back(CreateOdometryFactor<Scalar>(i));
  }

  return factors;
}

/**
 * Build and run the full optimization problem
 */
void RunDynamic() {
  auto values = BuildValues<double>(kNumPoses, kNumLandmarks);

  // Create and set up Optimizer
  const std::vector<sym::Factor<double>> factors =
      BuildDynamicFactors<double>(kNumPoses, kNumLandmarks);

  sym::Optimizer<double> optimizer(RobotLocalizationOptimizerParams(), factors,
                                   sym::kDefaultEpsilon<double>,
                                   "Robot3DScanMatchingOptimizerDynamic");

  // Optimize
  const sym::OptimizationStats<double> stats = optimizer.Optimize(&values);

  // Print out results
  spdlog::info("Optimized State:");
  for (int i = 0; i < kNumPoses; i++) {
    spdlog::info("Pose {}: {}", i,
                 values.At<sym::Pose3d>(sym::Key::WithSuper(sym::Keys::WORLD_T_BODY, i)));
  }

  const auto& iteration_stats = stats.iterations;
  const auto& first_iter = iteration_stats.front();
  const auto& last_iter = iteration_stats.back();

  // Note that the best iteration (with the best error, and representing the Values that gives
  // that error) may not be the last iteration, if later steps did not successfully decrease the
  // cost
  const auto& best_iter = iteration_stats[stats.best_index];

  spdlog::info("Iterations: {}", last_iter.iteration);
  spdlog::info("Lambda: {}", last_iter.current_lambda);
  spdlog::info("Initial error: {}", first_iter.new_error);
  spdlog::info("Final error: {}", best_iter.new_error);
}

// Explicit template specializations for floats and doubles
template sym::Factor<double> CreateMatchingFactor<double>(const int i, const int j);
template sym::Factor<float> CreateMatchingFactor<float>(const int i, const int j);

template sym::Factor<double> CreateOdometryFactor<double>(const int i);
template sym::Factor<float> CreateOdometryFactor<float>(const int i);

template std::vector<sym::Factor<double>> BuildDynamicFactors<double>(const int num_poses,
                                                                      const int num_landmarks);
template std::vector<sym::Factor<float>> BuildDynamicFactors<float>(const int num_poses,
                                                                    const int num_landmarks);

}  // namespace robot_3d_localization
