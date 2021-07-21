#include <sym/factors/between_factor_pose3.h>
#include <sym/factors/inverse_range_landmark_prior_factor.h>
#include <sym/factors/inverse_range_landmark_reprojection_error_factor.h>
#include <symforce/examples/example_utils/bundle_adjustment_util.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>

#include "./build_example_state.h"

namespace bundle_adjustment {

/**
 * Creates a factor for a prior on the relative pose between view i and view j
 */
sym::Factord CreateRelativePosePriorFactor(const int i, const int j) {
  return sym::Factord::Hessian(sym::BetweenFactorPose3<double>,
                               {{Var::VIEW, i},
                                {Var::VIEW, j},
                                {Var::POSE_PRIOR_T, i, j},
                                {Var::POSE_PRIOR_SQRT_INFO, i, j},
                                Var::EPSILON},
                               {
                                   {Var::VIEW, i},
                                   {Var::VIEW, j},
                               });
}

/**
 * Creates a factor for a prior on the inverse range of landmark landmark_idx based on its initial
 * triangulation between view 0 and view i
 */
sym::Factord CreateInverseRangeLandmarkPriorFactor(const int i, const int landmark_idx) {
  return sym::Factord::Hessian(sym::InverseRangeLandmarkPriorFactor<double>,
                               {{Var::LANDMARK, landmark_idx},
                                {Var::LANDMARK_PRIOR, i, landmark_idx},
                                {Var::MATCH_WEIGHT, i, landmark_idx},
                                {Var::LANDMARK_PRIOR_SIGMA, i, landmark_idx},
                                Var::EPSILON},
                               {{Var::LANDMARK, landmark_idx}});
}

/**
 * Creates a factor for a reprojection error residual of landmark landmark_idx projected into view i
 */
sym::Factord CreateInverseRangeLandmarkReprojectionErrorFactor(const int i,
                                                               const int landmark_idx) {
  return sym::Factord::Hessian(sym::InverseRangeLandmarkReprojectionErrorFactor<double>,
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
                               });
}

std::vector<sym::Factord> BuildFactors(const BundleAdjustmentProblemParams& params) {
  std::vector<sym::Factord> factors;

  // Relative pose priors
  for (int i = 0; i < params.num_views; i++) {
    for (int j = 0; j < params.num_views; j++) {
      if (i == j) {
        continue;
      }

      factors.push_back(CreateRelativePosePriorFactor(i, j));
    }
  }

  // Inverse range priors
  for (int i = 1; i < params.num_views; i++) {
    for (int landmark_idx = 0; landmark_idx < params.num_landmarks; landmark_idx++) {
      factors.push_back(CreateInverseRangeLandmarkPriorFactor(i, landmark_idx));
    }
  }

  // Reprojection errors
  for (int i = 1; i < params.num_views; i++) {
    for (int landmark_idx = 0; landmark_idx < params.num_landmarks; landmark_idx++) {
      factors.push_back(CreateInverseRangeLandmarkReprojectionErrorFactor(i, landmark_idx));
    }
  }

  return factors;
}

/**
 * Compute the keys in the Values which are optimized (as opposed to fixed)
 *
 * We fix the pose of view 0 so that the whole problem is constrained; alternatively, we could add a
 * prior on the pose of view 0 and leave it optimized
 */
std::vector<sym::Key> ComputeKeysToOptimizeWithoutView0(const std::vector<sym::Factord>& factors) {
  std::vector<sym::Key> keys_to_optimize;

  // ComputeKeysToOptimize will return all of the keys touched by all of the factors we've
  // created (specifically the optimized keys for those factors, i.e. the keys for which the factors
  // have derivatives, as opposed to other factor parameters like weights or epsilon)
  for (const auto& key : ComputeKeysToOptimize(factors, sym::Key::LexicalLessThan)) {
    // Don't optimize view 0
    if (key == sym::Key(Var::VIEW, 0)) {
      continue;
    }

    keys_to_optimize.push_back(key);
  }

  return keys_to_optimize;
}

void RunBundleAdjustment() {
  // Create initial state
  std::mt19937 gen(42);
  const auto params = BundleAdjustmentProblemParams();

  sym::Valuesd values = BuildValues(gen, params);

  std::cout << "Initial State:" << std::endl;
  for (int i = 0; i < params.num_views; i++) {
    std::cout << "Pose " << i << ": " << values.At<sym::Pose3d>({Var::VIEW, 0}) << std::endl;
  }
  std::cout << "Landmarks: ";
  for (int i = 0; i < params.num_landmarks; i++) {
    std::cout << values.At<double>({Var::LANDMARK, i}) << " ";
  }
  std::cout << std::endl;

  // Create and set up Optimizer
  const std::vector<sym::Factord> factors = BuildFactors(params);
  const std::vector<sym::Key> optimized_keys = ComputeKeysToOptimizeWithoutView0(factors);

  const sym::optimizer_params_t optimizer_params = sym::example_utils::OptimizerParams();

  sym::Optimizerd optimizer(optimizer_params, factors, params.epsilon, optimized_keys,
                            "BundleAdjustmentOptimizer", params.debug_stats,
                            params.check_derivatives);

  // Optimize
  const bool early_exit = optimizer.Optimize(&values);

  // Print out results
  std::cout << "Optimized State:" << std::endl;
  for (int i = 0; i < params.num_views; i++) {
    std::cout << "Pose " << i << ": " << values.At<sym::Pose3d>({Var::VIEW, i}) << std::endl;
  }
  std::cout << "Landmarks: ";
  for (int i = 0; i < params.num_landmarks; i++) {
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

}  // namespace bundle_adjustment
