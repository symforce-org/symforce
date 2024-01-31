/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <spdlog/spdlog.h>

#include <symforce/opt/assert.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>

#include "./common.h"
#include "./gen/keys.h"
#include "./gen/linearization.h"

namespace robot_3d_localization {

template <typename Scalar>
sym::Factor<Scalar> BuildFixedFactor() {
  std::vector<sym::Key> factor_keys;
  for (int i = 0; i < kNumPoses; i++) {
    factor_keys.push_back(sym::Keys::WORLD_T_BODY.WithSuper(i));
  }

  for (int i = 0; i < kNumLandmarks; i++) {
    factor_keys.push_back(sym::Keys::WORLD_T_LANDMARK.WithSuper(i));
  }

  factor_keys.push_back(sym::Keys::ODOMETRY_DIAGONAL_SIGMAS);

  for (int i = 0; i < kNumPoses - 1; i++) {
    factor_keys.push_back(sym::Keys::ODOMETRY_RELATIVE_POSE_MEASUREMENTS.WithSuper(i));
  }

  factor_keys.push_back(sym::Keys::MATCHING_SIGMA);

  for (int i = 0; i < kNumPoses; i++) {
    for (int j = 0; j < kNumLandmarks; j++) {
      factor_keys.push_back({sym::Keys::BODY_T_LANDMARK_MEASUREMENTS.Letter(), i, j});
    }
  }

  factor_keys.push_back(sym::Keys::EPSILON);

  std::vector<sym::Key> optimized_keys;
  for (int i = 0; i < kNumPoses; i++) {
    optimized_keys.push_back(sym::Keys::WORLD_T_BODY.WithSuper(i));
  }

  return sym::Factor<Scalar>::Hessian(Linearization<Scalar>, factor_keys, optimized_keys);
}

void RunFixed() {
  auto values = BuildValues<double>(kNumPoses, kNumLandmarks);

  // Create and set up Optimizer
  const std::vector<sym::Factor<double>> factors = {BuildFixedFactor<double>()};

  sym::Optimizer<double> optimizer(RobotLocalizationOptimizerParams(), factors,
                                   "Robot3DScanMatchingOptimizerFixed");

  // Optimize
  sym::Optimizerd::Stats stats = optimizer.Optimize(values);

  // Print out results
  spdlog::info("Optimized State:");
  for (int i = 0; i < kNumPoses; i++) {
    spdlog::info("Pose {}: {}", i, values.At<sym::Pose3d>(sym::Keys::WORLD_T_BODY.WithSuper(i)));
  }

  const auto& iteration_stats = stats.iterations;
  const auto& first_iter = iteration_stats.front();
  const auto& last_iter = iteration_stats.back();

  // Note that the best iteration (with the best error, and representing the Values that gives that
  // error) may not be the last iteration, if later steps did not successfully decrease the cost
  const auto& best_iter = iteration_stats[stats.best_index];

  spdlog::info("Iterations: {}", last_iter.iteration);
  spdlog::info("Lambda: {}", last_iter.current_lambda);
  spdlog::info("Initial error: {}", first_iter.new_error);
  spdlog::info("Final error: {}", best_iter.new_error);
  spdlog::info("Status: {}", stats.status);
}

template sym::Factor<double> BuildFixedFactor<double>();
template sym::Factor<float> BuildFixedFactor<float>();

}  // namespace robot_3d_localization
