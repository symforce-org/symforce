/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sym/factors/between_factor_matrix31.h>
#include <sym/factors/prior_factor_matrix31.h>
#include <symforce/opt/marginalization.h>
#include <symforce/opt/optimizer.h>

TEST_CASE("Test simple marginalization", "[marginalization]") {
  // Add a bunch of keys, then compute the ordering for marginalization.
  sym::Key measurement_key = sym::Key('y', 0);
  sym::Key sqrt_info_key = sym::Key('i', 0);
  sym::Key epsilon_key = sym::Key('e', 0);

  // Add a bunch of factors. All of them are just linear priors.
  std::vector<sym::Factord> factors;
  std::unordered_set<sym::Key> keys_to_optimize;
  for (int i = 0; i < 10; i++) {
    const sym::Key key = sym::Key('x', i);
    keys_to_optimize.insert(key);
    const std::vector<sym::Key> factor_keys = {key, measurement_key, sqrt_info_key, epsilon_key};
    const std::vector<sym::Key> optimized_keys = {key};
    factors.push_back(
        sym::Factord::Hessian(sym::PriorFactorMatrix31<double>, factor_keys, optimized_keys));
  }

  // Pick some arbitrary keys to marginalize.
  std::unordered_set<sym::Key> keys_to_marginalize;
  keys_to_marginalize.insert(sym::Key('x', 3));
  keys_to_marginalize.insert(sym::Key('x', 7));
  keys_to_marginalize.insert(sym::Key('x', 9));

  const std::vector<sym::Key> marginalization_linearization_order =
      sym::ComputeMarginalizationKeyOrder(keys_to_optimize, keys_to_marginalize);

  // We want the marginalization keys to come first in the ordering.
  const bool is_order_correct = std::is_sorted(
      marginalization_linearization_order.begin(), marginalization_linearization_order.end(),
      [&](const sym::Key& a, const sym::Key& b) {
        const bool is_a_marginalization_key =
            keys_to_marginalize.find(a) != keys_to_marginalize.end();
        const bool is_b_marginalization_key =
            keys_to_marginalize.find(b) != keys_to_marginalize.end();
        if (is_a_marginalization_key && !is_b_marginalization_key) {
          return true;
        }
        return false;
      });
  CHECK(is_order_correct);

  sym::Valuesd values;
  values.Set(measurement_key, Eigen::Vector3d::Ones());
  values.Set(sqrt_info_key, Eigen::Matrix3d::Identity());
  values.Set(epsilon_key, sym::kDefaultEpsilond);
  for (int i = 0; i < 10; i++) {
    values.Set(sym::Key('x', i), Eigen::Vector3d::Zero());
  }

  const auto marginalization_factor_or_info =
      sym::Marginalize(factors, values, keys_to_optimize, keys_to_marginalize);

  CHECK(std::holds_alternative<std::pair<sym::MarginalizationFactord, std::vector<sym::Key>>>(
      marginalization_factor_or_info));

  const auto& [marginalization_factor, marginalization_keys] =
      std::get<std::pair<sym::MarginalizationFactord, std::vector<sym::Key>>>(
          marginalization_factor_or_info);

  // We had 10 variables, and we marginalized 3 of them.
  CHECK(marginalization_factor.linearization_values.NumEntries() == 7);

  // Check that all the right keys are in the marginalization factor and values.
  for (int i = 0; i < 10; i++) {
    const sym::Key key = sym::Key('x', i);
    const bool key_in_marginalization_factor =
        std::find(marginalization_keys.begin(), marginalization_keys.end(), key) !=
        marginalization_keys.end();
    const bool key_in_values = marginalization_factor.linearization_values.Has(key);
    if (keys_to_marginalize.find(key) == keys_to_marginalize.end()) {
      // If the key is not marginalized, it should be in the marginalization factor.
      CHECK(key_in_marginalization_factor);
      CHECK(key_in_values);
    } else {
      // If the key is marginalized, it should not be in the marginalization factor.
      CHECK_FALSE(key_in_marginalization_factor);
      CHECK_FALSE(key_in_values);
    }
  }

  CHECK(marginalization_factor.H.rows() == 7 * 3);  // 7 remaining variables, 3dof each
  CHECK(marginalization_factor.H.rows() == marginalization_factor.H.cols());  // square
  // H and b form a linear system
  CHECK(marginalization_factor.rhs.rows() == marginalization_factor.H.rows());
}

TEST_CASE("Test just computing the Schur complement", "[marginalization]") {
  // We need the Hessian to be positive definite and symmetric.
  const Eigen::MatrixXd random_matrix = Eigen::MatrixXd::Random(9, 9);
  const Eigen::MatrixXd hessian =
      random_matrix.transpose() * random_matrix + Eigen::MatrixXd::Identity(9, 9) * 0.01;
  const Eigen::VectorXd rhs = Eigen::VectorXd::Random(9);

  // Symforce only fills in lower-only because the matrix is symmetric (to save compute). Corrupt
  // the upper triangle (excluding the diagonal) to make sure we handle that case.
  Eigen::MatrixXd hessian_lower_only = hessian;
  for (int i = 0; i < hessian.rows(); ++i) {
    for (int j = i + 1; j < hessian.cols(); ++j) {
      hessian_lower_only(i, j) = std::numeric_limits<double>::quiet_NaN();
    }
  }

  // Ignore the constant term for this test case.
  const auto marginalization_factor_or_info =
      sym::ComputeSchurComplement<double>(hessian_lower_only, rhs, 0, 3);
  CHECK(std::holds_alternative<sym::MarginalizationFactord>(marginalization_factor_or_info));
  const auto& marginalization_factor =
      std::get<sym::MarginalizationFactord>(marginalization_factor_or_info);

  // The expected operation is equivalent to computing the full inverse of the Hessian, dropping
  // the rows and columns, and then computing the inverse again.
  const Eigen::MatrixXd covariance = hessian.inverse();
  const Eigen::MatrixXd remaining_covariance = covariance.bottomRightCorner(6, 6);
  const Eigen::MatrixXd expected_remaining_hessian = remaining_covariance.inverse();

  CHECK(marginalization_factor.H.isApprox(expected_remaining_hessian, 1e-6));

  // NOTE(dominic): I don't have a particularly intuitive way to get the rhs term, but we can at
  // least compare it to the naive full matrix inversion approach.
  const Eigen::VectorXd expected_remaining_rhs =
      rhs.tail(6) -
      hessian.bottomLeftCorner(6, 3) * (hessian.topLeftCorner(3, 3).inverse() * rhs.head(3));
  CHECK(marginalization_factor.rhs.isApprox(expected_remaining_rhs, 1e-6));
}

TEST_CASE("Test optimizing with marginalization", "[marginalization]") {
  sym::Key epsilon_key = sym::Key('e', 0);

  // Set up a graph with a prior on x0 and a between factor on x0 and x1. We can then check that
  // after marginalizing x0, optimizing gives the same result. Because we'll use linear factors,
  // the approximation is perfect.
  std::vector<sym::Factord> factors;
  sym::Valuesd values;
  values.Set(sym::Key('e', 0), sym::kDefaultEpsilond);

  factors.push_back(sym::Factord::Hessian(
      sym::PriorFactorMatrix31<double>,
      std::vector<sym::Key>{sym::Key('x', 0), sym::Key('y', 0), sym::Key('i', 0), epsilon_key},
      std::vector<sym::Key>{sym::Key('x', 0)}));

  values.Set(sym::Key('x', 0), sym::Vector3d::Ones() * 20);  // arbitrary initial value
  values.Set(sym::Key('y', 0), sym::Vector3d::Zero());       // prior at 0
  values.Set(sym::Key('i', 0), sym::Matrix33d::Identity());  // identity covariance

  factors.push_back(
      sym::Factord::Hessian(sym::BetweenFactorMatrix31<double>,
                            std::vector<sym::Key>{sym::Key('x', 0), sym::Key('x', 1),
                                                  sym::Key('y', 1), sym::Key('i', 1), epsilon_key},
                            std::vector<sym::Key>{sym::Key('x', 0), sym::Key('x', 1)}));

  values.Set(sym::Key('x', 1), sym::Vector3d::Ones() * 50);  // arbitrary initial value
  values.Set(sym::Key('y', 1), sym::Vector3d::Zero());       // no delta between x0-x1
  values.Set(sym::Key('i', 1), sym::Matrix33d::Identity());  // identity covariance

  // Just take a single Gauss-Newton step because our graph is actually linear.
  sym::optimizer_params_t params = sym::DefaultOptimizerParams();
  params.iterations = 1;
  params.initial_lambda = 0.0;
  params.check_derivatives = true;
  params.include_jacobians = true;
  params.debug_stats = true;

  sym::Optimizer<double> optimizer_full(params, factors, "full",
                                        {sym::Key('x', 0), sym::Key('x', 1)});
  sym::Valuesd full_optimized_values = values;
  const auto full_optimization_stats = optimizer_full.Optimize(full_optimized_values);

  // x0 and x1 should now be zero.
  CHECK(full_optimized_values.At<sym::Vector3d>(sym::Key('x', 0)).isZero(1e-6));
  CHECK(full_optimized_values.At<sym::Vector3d>(sym::Key('x', 1)).isZero(1e-6));

  // We should have two iteration stats populated.
  CHECK(full_optimization_stats.iterations.size() == 2);
  // First iteration should have non-zero error because we set arbitrary wrong values.
  CHECK(full_optimization_stats.iterations[0].new_error != 0);
  // After our single step, the error should be zero.
  CHECK(std::abs(full_optimization_stats.iterations[1].new_error) < 1e-6);

  // Marginalized x0, leaving just x1.
  auto marginalization_factor_or_info =
      sym::Marginalize(factors, values, {sym::Key('x', 0), sym::Key('x', 1)}, {sym::Key('x', 0)});

  CHECK(std::holds_alternative<std::pair<sym::MarginalizationFactord, std::vector<sym::Key>>>(
      marginalization_factor_or_info));
  auto& [marginalization_factor, marginalization_keys] =
      std::get<std::pair<sym::MarginalizationFactord, std::vector<sym::Key>>>(
          marginalization_factor_or_info);

  // All factors end up being marginalized, so we are left with just the marginal.
  std::vector<sym::Factord> factors_after_marginalization;
  factors_after_marginalization.push_back(
      sym::CreateMarginalizationFactor(std::move(marginalization_factor), marginalization_keys));
  sym::Optimizer<double> optimizer_post_marginalization(params, factors_after_marginalization,
                                                        "post_marginalization",
                                                        std::vector<sym::Key>{sym::Key('x', 1)});
  sym::Valuesd post_marginalization_values = values;
  const auto post_marginalization_optimization_stats =
      optimizer_post_marginalization.Optimize(post_marginalization_values);

  // x1 should be 0 because that's the optimum for the full graph. x0 will be unchanged since it
  // was not optimized.
  CHECK(full_optimized_values.At<sym::Vector3d>(sym::Key('x', 1)).isZero(1e-6));

  // We should have two iteration stats populated.
  CHECK(post_marginalization_optimization_stats.iterations.size() == 2);
  // First iteration should have non-zero error because we set arbitrary wrong values.
  CHECK(post_marginalization_optimization_stats.iterations[0].new_error != 0);
  // After our single step, the error should be zero.
  CHECK(std::abs(post_marginalization_optimization_stats.iterations[1].new_error) < 1e-6);
}
