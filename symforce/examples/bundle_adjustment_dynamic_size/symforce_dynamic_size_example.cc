#include <sym/factors/between_factor_pose3.h>
#include <sym/factors/inverse_range_landmark_prior_factor.h>
#include <sym/factors/inverse_range_landmark_reprojection_error_factor.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>

#include "../example_utils/example_state_helpers.h"
#include "./build_example_state.h"

namespace sym {
namespace bundle_adjustment_dynamic_size {

std::vector<Factord> BuildFactors(const int num_views) {
  std::vector<Factord> factors;

  // Relative pose priors
  for (int i = 0; i < num_views; i++) {
    for (int j = 0; j < num_views; j++) {
      if (i == j) {
        continue;
      }

      factors.push_back(Factord::Jacobian(sym::BetweenFactorPose3<double>,
                                          {{Var::VIEW, i},
                                           {Var::VIEW, j},
                                           {Var::POSE_PRIOR_T, i, j},
                                           {Var::POSE_PRIOR_SQRT_INFO, i, j},
                                           Var::EPSILON},
                                          {
                                              {Var::VIEW, i},
                                              {Var::VIEW, j},
                                          }));
    }
  }

  // Inverse range priors
  for (int i = 1; i < num_views; i++) {
    for (int landmark_idx = 0; landmark_idx < kNumLandmarks; landmark_idx++) {
      factors.push_back(Factord::Hessian(sym::InverseRangeLandmarkPriorFactor<double>,
                                         {{Var::LANDMARK, landmark_idx},
                                          {Var::LANDMARK_PRIOR, i, landmark_idx},
                                          {Var::MATCH_WEIGHT, i, landmark_idx},
                                          {Var::LANDMARK_PRIOR_SIGMA, i, landmark_idx},
                                          Var::EPSILON},
                                         {{Var::LANDMARK, landmark_idx}}));
    }
  }

  // Reprojection errors
  for (int i = 1; i < num_views; i++) {
    for (int landmark_idx = 0; landmark_idx < kNumLandmarks; landmark_idx++) {
      factors.push_back(Factord::Hessian(sym::InverseRangeLandmarkReprojectionErrorFactor<double>,
                                         {{Var::VIEW, 0},
                                          {Var::CALIBRATION, 0},
                                          {Var::VIEW, i},
                                          {Var::CALIBRATION, i},
                                          {Var::LANDMARK, landmark_idx},
                                          {Var::MATCH_SOURCE_COORDS, i, landmark_idx},
                                          {Var::MATCH_TARGET_COORDS, i, landmark_idx},
                                          {Var::MATCH_WEIGHT, i, landmark_idx},
                                          Var::GNC_MU,
                                          Var::GNC_SCALE,
                                          Var::EPSILON},
                                         {
                                             {Var::VIEW, 0},
                                             {Var::VIEW, i},
                                             {Var::LANDMARK, landmark_idx},
                                         }));
    }
  }

  return factors;
}

std::vector<Key> ComputeKeysToOptimizeWithoutView0(const std::vector<Factord>& factors) {
  std::vector<Key> keys_to_optimize;
  for (const auto& key : ComputeKeysToOptimize(factors, Key::LexicalLessThan)) {
    // Don't optimize view 0
    if (key == Key(Var::VIEW, 0)) {
      continue;
    }

    keys_to_optimize.push_back(key);
  }

  return keys_to_optimize;
}

void RunDynamicBundleAdjustment() {
  // Create initial state
  std::mt19937 gen(42);

  Valuesd values = BuildValues(gen, kNumLandmarks);

  std::cout << "Initial State:" << std::endl;
  for (int i = 0; i < kNumViews; i++) {
    std::cout << "Pose " << i << ": " << values.At<sym::Pose3d>({Var::VIEW, 0}) << std::endl;
  }
  std::cout << "Landmarks: ";
  for (int i = 0; i < kNumLandmarks; i++) {
    std::cout << values.At<double>({Var::LANDMARK, i}) << " ";
  }
  std::cout << std::endl;

  // Create and set up Optimizer
  const std::vector<Factord> factors = BuildFactors(kNumViews);
  const std::vector<Key> optimized_keys = ComputeKeysToOptimizeWithoutView0(factors);

  const optimizer_params_t optimizer_params = example_utils::OptimizerParams();

  Optimizerd optimizer(optimizer_params, factors, kEpsilon, optimized_keys);

  // Optimize
  const bool early_exit = optimizer.Optimize(&values);

  // Print out results
  std::cout << "Optimized State:" << std::endl;
  for (int i = 0; i < kNumViews; i++) {
    std::cout << "Pose " << i << ": " << values.At<sym::Pose3d>({Var::VIEW, i}) << std::endl;
  }
  std::cout << "Landmarks: ";
  for (int i = 0; i < kNumLandmarks; i++) {
    std::cout << values.At<double>({Var::LANDMARK, i}) << " ";
  }
  std::cout << std::endl;

  const auto& iteration_stats = optimizer.Stats().iterations;
  const auto& first_iter = iteration_stats[0];
  const auto& last_iter = iteration_stats[iteration_stats.size() - 1];
  std::cout << "Iterations: " << last_iter.iteration << std::endl;
  std::cout << "Lambda: " << last_iter.current_lambda << std::endl;
  std::cout << "Initial error: " << first_iter.new_error << std::endl;
  std::cout << "Final error: " << last_iter.new_error << std::endl;

  // Check successful convergence
  SYM_ASSERT(last_iter.iteration == 35);
  SYM_ASSERT(last_iter.new_error < 10);
}

}  // namespace bundle_adjustment_dynamic_size
}  // namespace sym
