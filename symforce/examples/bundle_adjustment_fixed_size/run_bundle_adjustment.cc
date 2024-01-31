/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <iostream>

#include <sym/pose3.h>
#include <symforce/examples/example_utils/bundle_adjustment_util.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/optimizer.h>

#include "./build_example_state.h"
#include "symforce/bundle_adjustment_fixed_size/linearization.h"

namespace bundle_adjustment_fixed_size {

sym::Factord BuildFactor() {
  const std::vector<sym::Key> factor_keys = {{Var::CALIBRATION, 0},
                                             {Var::VIEW, 0},
                                             {Var::CALIBRATION, 1},
                                             {Var::VIEW, 1},
                                             {Var::POSE_PRIOR_T, 0, 0},
                                             {Var::POSE_PRIOR_SQRT_INFO, 0, 0},
                                             {Var::POSE_PRIOR_T, 0, 1},
                                             {Var::POSE_PRIOR_SQRT_INFO, 0, 1},
                                             {Var::POSE_PRIOR_T, 1, 0},
                                             {Var::POSE_PRIOR_SQRT_INFO, 1, 0},
                                             {Var::POSE_PRIOR_T, 1, 1},
                                             {Var::POSE_PRIOR_SQRT_INFO, 1, 1},
                                             {Var::MATCH_SOURCE_COORDS, 1, 0},
                                             {Var::MATCH_TARGET_COORDS, 1, 0},
                                             {Var::MATCH_WEIGHT, 1, 0},
                                             {Var::LANDMARK_PRIOR, 1, 0},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 0},
                                             {Var::MATCH_SOURCE_COORDS, 1, 1},
                                             {Var::MATCH_TARGET_COORDS, 1, 1},
                                             {Var::MATCH_WEIGHT, 1, 1},
                                             {Var::LANDMARK_PRIOR, 1, 1},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 1},
                                             {Var::MATCH_SOURCE_COORDS, 1, 2},
                                             {Var::MATCH_TARGET_COORDS, 1, 2},
                                             {Var::MATCH_WEIGHT, 1, 2},
                                             {Var::LANDMARK_PRIOR, 1, 2},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 2},
                                             {Var::MATCH_SOURCE_COORDS, 1, 3},
                                             {Var::MATCH_TARGET_COORDS, 1, 3},
                                             {Var::MATCH_WEIGHT, 1, 3},
                                             {Var::LANDMARK_PRIOR, 1, 3},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 3},
                                             {Var::MATCH_SOURCE_COORDS, 1, 4},
                                             {Var::MATCH_TARGET_COORDS, 1, 4},
                                             {Var::MATCH_WEIGHT, 1, 4},
                                             {Var::LANDMARK_PRIOR, 1, 4},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 4},
                                             {Var::MATCH_SOURCE_COORDS, 1, 5},
                                             {Var::MATCH_TARGET_COORDS, 1, 5},
                                             {Var::MATCH_WEIGHT, 1, 5},
                                             {Var::LANDMARK_PRIOR, 1, 5},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 5},
                                             {Var::MATCH_SOURCE_COORDS, 1, 6},
                                             {Var::MATCH_TARGET_COORDS, 1, 6},
                                             {Var::MATCH_WEIGHT, 1, 6},
                                             {Var::LANDMARK_PRIOR, 1, 6},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 6},
                                             {Var::MATCH_SOURCE_COORDS, 1, 7},
                                             {Var::MATCH_TARGET_COORDS, 1, 7},
                                             {Var::MATCH_WEIGHT, 1, 7},
                                             {Var::LANDMARK_PRIOR, 1, 7},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 7},
                                             {Var::MATCH_SOURCE_COORDS, 1, 8},
                                             {Var::MATCH_TARGET_COORDS, 1, 8},
                                             {Var::MATCH_WEIGHT, 1, 8},
                                             {Var::LANDMARK_PRIOR, 1, 8},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 8},
                                             {Var::MATCH_SOURCE_COORDS, 1, 9},
                                             {Var::MATCH_TARGET_COORDS, 1, 9},
                                             {Var::MATCH_WEIGHT, 1, 9},
                                             {Var::LANDMARK_PRIOR, 1, 9},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 9},
                                             {Var::MATCH_SOURCE_COORDS, 1, 10},
                                             {Var::MATCH_TARGET_COORDS, 1, 10},
                                             {Var::MATCH_WEIGHT, 1, 10},
                                             {Var::LANDMARK_PRIOR, 1, 10},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 10},
                                             {Var::MATCH_SOURCE_COORDS, 1, 11},
                                             {Var::MATCH_TARGET_COORDS, 1, 11},
                                             {Var::MATCH_WEIGHT, 1, 11},
                                             {Var::LANDMARK_PRIOR, 1, 11},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 11},
                                             {Var::MATCH_SOURCE_COORDS, 1, 12},
                                             {Var::MATCH_TARGET_COORDS, 1, 12},
                                             {Var::MATCH_WEIGHT, 1, 12},
                                             {Var::LANDMARK_PRIOR, 1, 12},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 12},
                                             {Var::MATCH_SOURCE_COORDS, 1, 13},
                                             {Var::MATCH_TARGET_COORDS, 1, 13},
                                             {Var::MATCH_WEIGHT, 1, 13},
                                             {Var::LANDMARK_PRIOR, 1, 13},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 13},
                                             {Var::MATCH_SOURCE_COORDS, 1, 14},
                                             {Var::MATCH_TARGET_COORDS, 1, 14},
                                             {Var::MATCH_WEIGHT, 1, 14},
                                             {Var::LANDMARK_PRIOR, 1, 14},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 14},
                                             {Var::MATCH_SOURCE_COORDS, 1, 15},
                                             {Var::MATCH_TARGET_COORDS, 1, 15},
                                             {Var::MATCH_WEIGHT, 1, 15},
                                             {Var::LANDMARK_PRIOR, 1, 15},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 15},
                                             {Var::MATCH_SOURCE_COORDS, 1, 16},
                                             {Var::MATCH_TARGET_COORDS, 1, 16},
                                             {Var::MATCH_WEIGHT, 1, 16},
                                             {Var::LANDMARK_PRIOR, 1, 16},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 16},
                                             {Var::MATCH_SOURCE_COORDS, 1, 17},
                                             {Var::MATCH_TARGET_COORDS, 1, 17},
                                             {Var::MATCH_WEIGHT, 1, 17},
                                             {Var::LANDMARK_PRIOR, 1, 17},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 17},
                                             {Var::MATCH_SOURCE_COORDS, 1, 18},
                                             {Var::MATCH_TARGET_COORDS, 1, 18},
                                             {Var::MATCH_WEIGHT, 1, 18},
                                             {Var::LANDMARK_PRIOR, 1, 18},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 18},
                                             {Var::MATCH_SOURCE_COORDS, 1, 19},
                                             {Var::MATCH_TARGET_COORDS, 1, 19},
                                             {Var::MATCH_WEIGHT, 1, 19},
                                             {Var::LANDMARK_PRIOR, 1, 19},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 19},
                                             {Var::LANDMARK, 0},
                                             {Var::LANDMARK, 1},
                                             {Var::LANDMARK, 2},
                                             {Var::LANDMARK, 3},
                                             {Var::LANDMARK, 4},
                                             {Var::LANDMARK, 5},
                                             {Var::LANDMARK, 6},
                                             {Var::LANDMARK, 7},
                                             {Var::LANDMARK, 8},
                                             {Var::LANDMARK, 9},
                                             {Var::LANDMARK, 10},
                                             {Var::LANDMARK, 11},
                                             {Var::LANDMARK, 12},
                                             {Var::LANDMARK, 13},
                                             {Var::LANDMARK, 14},
                                             {Var::LANDMARK, 15},
                                             {Var::LANDMARK, 16},
                                             {Var::LANDMARK, 17},
                                             {Var::LANDMARK, 18},
                                             {Var::LANDMARK, 19},
                                             {Var::GNC_SCALE},
                                             {Var::GNC_MU},
                                             {Var::EPSILON}};

  const std::vector<sym::Key> optimized_keys = {
      {Var::VIEW, 1},      {Var::LANDMARK, 0},  {Var::LANDMARK, 1},  {Var::LANDMARK, 2},
      {Var::LANDMARK, 3},  {Var::LANDMARK, 4},  {Var::LANDMARK, 5},  {Var::LANDMARK, 6},
      {Var::LANDMARK, 7},  {Var::LANDMARK, 8},  {Var::LANDMARK, 9},  {Var::LANDMARK, 10},
      {Var::LANDMARK, 11}, {Var::LANDMARK, 12}, {Var::LANDMARK, 13}, {Var::LANDMARK, 14},
      {Var::LANDMARK, 15}, {Var::LANDMARK, 16}, {Var::LANDMARK, 17}, {Var::LANDMARK, 18},
      {Var::LANDMARK, 19}};

  return sym::Factord::Hessian(bundle_adjustment_fixed_size::Linearization<double>, factor_keys,
                               optimized_keys);
}

void RunBundleAdjustment() {
  // Create initial state
  std::mt19937 gen(42);
  const auto params = BundleAdjustmentProblemParams();
  sym::Valuesd values = BuildValues(gen, params);

  spdlog::info("Initial State:");
  for (int i = 0; i < params.num_views; i++) {
    spdlog::info("Pose {}: {}", i, values.At<sym::Pose3d>({Var::VIEW, i}));
  }
  spdlog::info("Landmarks:");
  for (int i = 0; i < params.num_landmarks; i++) {
    spdlog::info("{} ", values.At<double>({Var::LANDMARK, i}));
  }

  // Create and set up Optimizer
  const sym::optimizer_params_t optimizer_params = sym::example_utils::OptimizerParams();

  sym::Optimizerd optimizer(optimizer_params, {BuildFactor()}, "BundleAdjustmentOptimizer", {},
                            params.epsilon);

  // Optimize
  const sym::Optimizerd::Stats stats = optimizer.Optimize(values);

  // Print out results
  spdlog::info("Optimized State:");
  for (int i = 0; i < params.num_views; i++) {
    spdlog::info("Pose {}: {}", i, values.At<sym::Pose3d>({Var::VIEW, i}));
  }
  spdlog::info("Landmarks:");
  for (int i = 0; i < params.num_landmarks; i++) {
    spdlog::info("{} ", values.At<double>({Var::LANDMARK, i}));
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

  // Check successful convergence
  SYM_ASSERT(best_iter.new_error < 10);
  SYM_ASSERT(stats.status == sym::optimization_status_t::SUCCESS);
}

}  // namespace bundle_adjustment_fixed_size
