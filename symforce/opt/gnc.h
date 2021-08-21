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
  GncOptimizer(const optimizer_params_t& optimizer_params, const optimizer_gnc_params_t& gnc_params,
               const Key& gnc_mu_key, const std::vector<Factor<Scalar>>& factors,
               const Scalar epsilon = 1e-9, const std::string& name = "sym::Optimize",
               const std::vector<Key>& keys = {}, bool debug_stats = false,
               bool check_derivatives = false)
      : BaseOptimizer(optimizer_params, factors, epsilon, name, keys, debug_stats,
                      check_derivatives),
        gnc_params_(gnc_params),
        gnc_mu_key_(gnc_mu_key) {}

  /**
   * Constructor with move constructors for factors and keys.
   */
  GncOptimizer(const optimizer_params_t& optimizer_params, const optimizer_gnc_params_t& gnc_params,
               const Key& gnc_mu_key, std::vector<Factor<Scalar>>&& factors,
               const Scalar epsilon = 1e-9, const std::string& name = "sym::Optimize",
               std::vector<Key>&& keys = {}, bool debug_stats = false,
               bool check_derivatives = false)
      : BaseOptimizer(optimizer_params, std::move(factors), epsilon, name, std::move(keys),
                      debug_stats, check_derivatives),
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
  virtual bool Optimize(Values<Scalar>* const values, int num_iterations = -1,
                        Linearization<Scalar>* const best_linearization = nullptr) override {
    if (num_iterations < 0) {
      num_iterations = this->nonlinear_solver_.Params().iterations;
    }

    bool updating_gnc = true;

    // Initialize the value of mu
    values->template Set<Scalar>(gnc_mu_key_, gnc_params_.mu_initial);

    // Set early-exit for GNC.
    optimizer_params_t optimizer_params = this->nonlinear_solver_.Params();
    optimizer_params.early_exit_min_reduction = gnc_params_.gnc_update_min_reduction;
    this->UpdateParams(optimizer_params);

    // Iterate.
    bool early_exit = BaseOptimizer::Optimize(values, num_iterations, best_linearization);
    while (this->Stats().iterations.size() < num_iterations) {
      if (early_exit) {
        if (updating_gnc) {
          // Update the GNC parameter.
          values->Set(gnc_mu_key_, values->template At<Scalar>(gnc_mu_key_) + gnc_params_.mu_step);

          // Check if we hit the non-convexity threshold.
          if (values->template At<Scalar>(gnc_mu_key_) >= gnc_params_.mu_max) {
            values->template Set<Scalar>(gnc_mu_key_, gnc_params_.mu_max);
            // Reset early exit threshold.
            optimizer_params.early_exit_min_reduction = optimizer_params.early_exit_min_reduction;
            this->UpdateParams(optimizer_params);
            updating_gnc = false;
          }

          if (optimizer_params.verbose) {
            REPORT_STATUS_NOW("Set GNC param to: {}", values->template At<Scalar>(gnc_mu_key_));
          }
        } else {
          return true;
        }
      }

      early_exit = OptimizeContinue(values, num_iterations - this->Stats().iterations.size(),
                                    best_linearization);
    }

    return false;
  }

 private:
  bool OptimizeContinue(Values<Scalar>* const values, const int num_iterations,
                        Linearization<Scalar>* const best_linearization) {
    SYM_ASSERT(num_iterations >= 0);
    SYM_ASSERT(this->IsInitialized());

    // Reset values, but do not clear other state
    this->nonlinear_solver_.ResetState(*values);

    return this->IterateToConvergence(values, num_iterations, best_linearization);
  }

  optimizer_gnc_params_t gnc_params_;
  Key gnc_mu_key_;
};

}  // namespace sym
