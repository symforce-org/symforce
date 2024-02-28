/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/gnc_optimizer.h>
#include <symforce/opt/optimizer.h>

#include "symforce_function_codegen_test_data/symengine/gnc_test_data/cpp/symforce/gnc_factors/barron_factor.h"

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

  auto params = sym::DefaultOptimizerParams();
  params.verbose = true;

  sym::GncOptimizer<sym::Optimizerd> gnc_optimizer(
      params, DefaultGncParams(), 'u', factors, kEpsilon, "sym::Optimize",
      /* keys */ std::vector<sym::Key>{}, /* debug_stats */ false,
      /* check_derivatives */ true, /* include_jacobians */ true);

  INFO("Initial x: " << initial_values.At<sym::Vector5d>('x').transpose());

  sym::Valuesd gnc_optimized_values = initial_values;
  const auto gnc_stats = gnc_optimizer.Optimize(gnc_optimized_values);
  INFO("Final x: " << gnc_optimized_values.At<sym::Vector5d>('x').transpose());

  sym::Valuesd regular_optimized_values = initial_values;
  regular_optimized_values.Set('u', 0.0);
  sym::Optimize(params, factors, regular_optimized_values, kEpsilon);

  INFO("Final x without GNC:" << regular_optimized_values.At<sym::Vector5d>('x').transpose());

  CHECK(gnc_stats.iterations.size() == 9);
  const sym::Vector5d gnc_optimized_x = gnc_optimized_values.At<sym::Vector5d>('x');
  const sym::Vector5d regular_optimized_x = regular_optimized_values.At<sym::Vector5d>('x');
  CHECK(gnc_optimized_x.norm() < 0.1);
  CHECK(gnc_optimized_x.norm() * 5 < regular_optimized_x.norm());
  CHECK(gnc_stats.status == sym::optimization_status_t::SUCCESS);
}
