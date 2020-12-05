#include <iostream>

#include <sym/pose3.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/levenberg_marquardt_solver.h>

#include <lcmtypes/bundle_adjustment_example/state_t.hpp>

#include "./build_example_state.h"
#include "symforce/bundle_adjustment_example/linearization.h"
#include "symforce/bundle_adjustment_example/update.h"

namespace sym {

static constexpr int kPoseDim = 6;
static constexpr int kVariablesDim = kNumLandmarks + (kNumViews - 1) * kPoseDim;
static constexpr int kNumReprojectionErrors = kNumLandmarks * (kNumViews - 1);
static constexpr int kResidualDim = 2 * kNumReprojectionErrors                // Reprojection error
                                    + kNumViews * (kNumViews - 1) * kPoseDim  // Pose prior
                                    + kNumLandmarks;                          // Inverse range prior

void RunFixedBundleAdjustment() {
#if 0
  // Create initial state
  std::mt19937 gen(42);
  Valuesd initial_state;
  FillValuesFromState(BuildState(gen, kNumLandmarks), &initial_state, kNumViews);

  std::cout << "Initial State:" << std::endl;
  for (int i = 0; i < kNumViews; i++) {
    std::cout << "Pose " << i << ": " << sym::Pose3d(initial_state.At<sym::Pose3d>({Var::VIEW, i}))
              << std::endl;
  }
  std::cout << "Landmarks: ";
  for (int i = 0; i < kNumLandmarks; i++) {
    std::cout << initial_state.At<double>({Var::LANDMARK, i}) << " ";
  }
  std::cout << std::endl;

  // Create and set up Optimizer
  const optimizer_params_t optimizer_params = OptimizerParams();

  using Optimizer = sym::LevenbergMarquardtSolverd;

  Optimizer optimizer(optimizer_params, "sym::Optimizer", kEpsilon);

  std::vector<optimizer_iteration_t> iteration_stats;

  // Reset the optimizer state
  optimizer.Reset(initial_state);

  // Wrap linearization function
  const auto linearize_func = [](const bundle_adjustment_example::state_t& state,
                                 sym::Linearizationd* const linearization) {
    bundle_adjustment_example::Linearization<double>(state, &linearization->residual,
                                                     &linearization->jacobian, &linearization->rhs,
                                                     &linearization->hessian_lower);
  };

  // Optimize
  for (int i = 0; i < optimizer_params.iterations; i++) {
    const bool early_exit = optimizer.Iterate(linearize_func, &iteration_stats);
    if (early_exit) {
      break;
    }
  }

  // Print out results
  const Valuesd final_state = optimizer.GetBestValues();

  std::cout << "Optimized State:" << std::endl;
  for (int i = 0; i < kNumViews; i++) {
    std::cout << "Pose " << i << ": " << sym::Pose3d(final_state.At<sym::Pose3d>({Var::VIEW, i}))
              << std::endl;
  }
  std::cout << "Landmarks: ";
  for (int i = 0; i < kNumLandmarks; i++) {
    std::cout << final_state.At<double>({Var::LANDMARK, i}) << " ";
  }
  std::cout << std::endl;

  const auto& first_iter = iteration_stats[0];
  const auto& last_iter = iteration_stats[iteration_stats.size() - 1];
  std::cout << "Iterations: " << last_iter.iteration << std::endl;
  std::cout << "Lambda: " << last_iter.current_lambda << std::endl;
  std::cout << "Initial error: " << first_iter.new_error << std::endl;
  std::cout << "Final error: " << last_iter.new_error << std::endl;

  // Check successful convergence
  SYM_ASSERT(last_iter.iteration == 5);
  SYM_ASSERT(last_iter.new_error < 3);
#endif
}

}  // namespace sym
