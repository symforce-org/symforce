/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include <symforce/opt/gnc_optimizer.h>

#include "symforce_function_codegen_test_data/symengine/gnc_test_data/cpp/symforce/gnc_factors/barron_factor.h"

sym::optimizer_params_t DefaultLmParams() {
  sym::optimizer_params_t params{};
  params.iterations = 50;
  params.verbose = true;
  params.initial_lambda = 1.0;
  params.lambda_up_factor = 4.0;
  params.lambda_down_factor = 1 / 4.0;
  params.lambda_lower_bound = 0.0;
  params.lambda_upper_bound = 1000000.0;
  params.early_exit_min_reduction = 1e-6;
  params.use_unit_damping = true;
  params.use_diagonal_damping = false;
  params.keep_max_diagonal_damping = false;
  params.diagonal_damping_min = 1e-6;
  params.enable_bold_updates = false;
  return params;
}

sym::optimizer_gnc_params_t DefaultGncParams() {
  sym::optimizer_gnc_params_t params{};
  params.mu_initial = 0;
  params.mu_max = 0.99;
  params.mu_step = 0.33;
  params.gnc_update_min_reduction = 1e-3;
  return params;
}

TEST_CASE("Test GNC", "[gnc]") {
  static constexpr const double kEpsilon = 1e-12;
  const int n_residuals = 20;
  const int n_outliers = 3;

  // Create values
  sym::Valuesd initial_values;
  initial_values.Set<sym::Vector5d>('x', sym::Vector5d::Ones());
  initial_values.Set('e', kEpsilon);

  // Pick random normal samples, with some outliers
  std::mt19937 gen(42);
  for (int i = 0; i < n_residuals; i++) {
    if (i < n_outliers) {
      initial_values.Set<sym::Vector5d>(
          {'y', i}, sym::Vector5d::Constant(10) + 0.1 * sym::Random<sym::Vector5d>(gen));
    } else {
      initial_values.Set<sym::Vector5d>({'y', i}, 0.1 * sym::Random<sym::Vector5d>(gen));
    }
  }

  std::vector<sym::Factord> factors;
  for (int i = 0; i < n_residuals; i++) {
    factors.push_back(
        sym::Factord::Hessian(gnc_factors::BarronFactor<double>, {'x', {'y', i}, 'u', 'e'}, {'x'}));
  }

  sym::GncOptimizer<sym::Optimizerd> gnc_optimizer(
      DefaultLmParams(), DefaultGncParams(), 'u', factors, kEpsilon, "sym::Optimize",
      /* keys */ std::vector<sym::Key>{}, /* debug_stats */ false,
      /* check_derivatives */ true, /* include_jacobians */ true);

  spdlog::debug("Initial x: {}", initial_values.At<sym::Vector5d>('x').transpose());

  sym::Valuesd gnc_optimized_values = initial_values;
  const auto gnc_stats = gnc_optimizer.Optimize(gnc_optimized_values);
  spdlog::debug("Final x: {}", gnc_optimized_values.At<sym::Vector5d>('x').transpose());

  sym::Valuesd regular_optimized_values = initial_values;
  regular_optimized_values.Set('u', 0.0);
  sym::Optimize(DefaultLmParams(), factors, regular_optimized_values, kEpsilon);

  spdlog::debug("Final x without GNC: {}",
                regular_optimized_values.At<sym::Vector5d>('x').transpose());

  CHECK(gnc_stats.iterations.size() == 9);
  const sym::Vector5d gnc_optimized_x = gnc_optimized_values.At<sym::Vector5d>('x');
  const sym::Vector5d regular_optimized_x = regular_optimized_values.At<sym::Vector5d>('x');
  CHECK(gnc_optimized_x.norm() < 0.1);
  CHECK(gnc_optimized_x.norm() * 5 < regular_optimized_x.norm());
}
