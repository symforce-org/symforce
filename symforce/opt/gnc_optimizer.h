/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <lcmtypes/sym/optimizer_gnc_params_t.hpp>

#include "./optimizer.h"

namespace sym {

/**
 * Subclass of Optimizer for using Graduated Non-Convexity (GNC)
 *
 * Assumes the convexity of the cost function is controlled by a hyperparameter mu. When mu == 0 the
 * cost function should be convex and as mu goes to 1 the cost function should smoothly transition
 * to a robust cost.
 */
template <typename BaseOptimizerType>
class GncOptimizer : public BaseOptimizerType {
 public:
  using BaseOptimizer = BaseOptimizerType;
  using Scalar = typename BaseOptimizer::Scalar;

  /**
   * Constructor that copies in factors and keys
   */
  template <typename... OptimizerArgs>
  GncOptimizer(const optimizer_params_t& optimizer_params, const optimizer_gnc_params_t& gnc_params,
               const Key& gnc_mu_key, OptimizerArgs&&... args)
      : BaseOptimizer(optimizer_params, std::forward<OptimizerArgs>(args)...),
        gnc_params_(gnc_params),
        gnc_mu_key_(gnc_mu_key) {}

  virtual ~GncOptimizer() = default;

  /**
   * Optimize the given values in-place.  This will optimize until convergence with the convex cost,
   * then repeatedly make the cost function less convex and optimize to convergence again, on the
   * schedule specified by the gnc_params.
   *
   * If num_iterations < 0 (the default), uses the number of iterations specified by the params at
   * construction.  Note that this is the total number of iterations, the counter does not reset
   * each time the convexity changes.
   */
  using BaseOptimizerType::Optimize;
  virtual void Optimize(Values<Scalar>* const values, int num_iterations,
                        bool populate_best_linearization,
                        OptimizationStats<Scalar>* const stats) override {
    SYM_TIME_SCOPE("GNC<{}>::Optimize", BaseOptimizer::GetName());
    SYM_ASSERT(values != nullptr);
    SYM_ASSERT(stats != nullptr);

    if (num_iterations < 0) {
      num_iterations = this->nonlinear_solver_.Params().iterations;
    }

    bool updating_gnc = (gnc_params_.mu_initial < gnc_params_.mu_max);

    // Initialize the value of mu
    values->template Set<Scalar>(gnc_mu_key_, gnc_params_.mu_initial);

    // Cache the index entry for mu
    const auto mu_index = values->Items().at(gnc_mu_key_);

    optimizer_params_t optimizer_params = this->nonlinear_solver_.Params();
    const double early_exit_min_reduction = optimizer_params.early_exit_min_reduction;
    if (updating_gnc) {
      // Set early-exit for GNC.
      optimizer_params.early_exit_min_reduction = gnc_params_.gnc_update_min_reduction;
    }
    this->UpdateParams(optimizer_params);

    // Iterate.
    BaseOptimizer::Optimize(values, num_iterations, populate_best_linearization, stats);
    while (stats->iterations.size() < num_iterations) {
      // NOTE(aaron): We shouldn't be here unless the optimization early exited (i.e. we had
      // iterations left)
      SYM_ASSERT(stats->early_exited);

      if (!updating_gnc) {
        return;
      }

      // Update the GNC parameter.
      values->Set(gnc_mu_key_, values->template At<Scalar>(mu_index) + gnc_params_.mu_step);

      // Check if we hit the non-convexity threshold.
      if (values->template At<Scalar>(mu_index) >= gnc_params_.mu_max) {
        values->template Set<Scalar>(mu_index, gnc_params_.mu_max);
        // Reset early exit threshold.
        optimizer_params.early_exit_min_reduction = early_exit_min_reduction;
        this->UpdateParams(optimizer_params);
        updating_gnc = false;
      }

      if (optimizer_params.verbose) {
        spdlog::info("Set GNC param to: {}", values->template At<Scalar>(mu_index));
      }

      // NOTE(aaron): This might populate the best linearization multiple times
      OptimizeContinue(values, num_iterations - stats->iterations.size(),
                       populate_best_linearization, stats);
    }
  }

 private:
  void OptimizeContinue(Values<Scalar>* const values, const int num_iterations,
                        const bool populate_best_linearization,
                        OptimizationStats<Scalar>* const stats) {
    SYM_ASSERT(num_iterations >= 0);
    SYM_ASSERT(this->IsInitialized());

    // Reset values, but do not clear other state
    this->nonlinear_solver_.ResetState(*values);

    this->IterateToConvergence(values, num_iterations, populate_best_linearization, stats);
  }

  optimizer_gnc_params_t gnc_params_;
  Key gnc_mu_key_;
};

}  // namespace sym
