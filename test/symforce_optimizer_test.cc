/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sym/factors/between_factor_pose3.h>
#include <sym/factors/between_factor_rot3.h>
#include <sym/factors/prior_factor_pose3.h>
#include <sym/factors/prior_factor_rot3.h>
#include <symforce/opt/optimization_stats.h>
#include <symforce/opt/optimizer.h>

sym::optimizer_params_t DefaultLmParams() {
  sym::optimizer_params_t params{};
  params.iterations = 50;
  params.verbose = true;
  params.initial_lambda = 1.0;
  params.lambda_up_factor = 4.0;
  params.lambda_down_factor = 1 / 4.0;
  params.lambda_lower_bound = 0.0;
  params.lambda_upper_bound = 1000000.0;
  params.early_exit_min_reduction = 0.0001;
  params.use_unit_damping = true;
  params.use_diagonal_damping = false;
  params.keep_max_diagonal_damping = false;
  params.diagonal_damping_min = 1e-6;
  params.enable_bold_updates = false;
  return params;
}

/**
 * Test convergence on a toy nonlinear problem:
 *
 *   z = 3.0 - 0.5 * sin((x - 2) / 5) + 1.0 * sin((y + 2) / 10.)
 */
TEST_CASE("Test nonlinear convergence", "[optimizer]") {
  // Create factors
  std::vector<sym::Factord> factors;
  factors.push_back(sym::Factord::Jacobian(
      [](double x, double y, sym::Vector1d* residual, Eigen::Matrix<double, 1, 2>* jacobian) {
        (*residual)[0] = 3.0 - 0.5 * std::sin((x - 2.) / 5.) + 1.0 * std::sin((y + 2.) / 10.);

        if (jacobian) {
          (*jacobian)(0, 0) = -0.1 * std::cos(0.2 * (-2.0 + x));
          (*jacobian)(0, 1) = 0.1 * std::cos(0.1 * (2.0 + y));
        }
      },
      {'x', 'y'}));

  // Create values
  sym::Valuesd values;
  values.Set<double>('x', 0.0);
  values.Set<double>('y', 0.0);
  INFO("Initial values: " << values);

  // Set parameters
  sym::optimizer_params_t params = DefaultLmParams();
  params.initial_lambda = 10.0;
  params.lambda_up_factor = 3.0;
  params.lambda_down_factor = 1.0 / 3.0;
  params.lambda_lower_bound = 0.01;
  params.lambda_upper_bound = 1000.0;
  params.early_exit_min_reduction = 1e-9;
  params.iterations = 25;
  params.use_diagonal_damping = true;
  params.use_unit_damping = true;

  // Optimize
  const auto stats = Optimize(params, factors, values);

  // Check results
  INFO("Optimized values: " << values);

  // Local minimum from Wolfram Alpha
  // pylint: disable=line-too-long
  // https://www.wolframalpha.com/input/?i=z+%3D+3+-+0.5+*+sin((x+-+2)+%2F+5)+%2B+1.0+*+sin((y+%2B+2)+%2F+10.)
  const Eigen::Vector2d expected_gt = {9.854, -17.708};
  const Eigen::Vector2d actual = {values.At<double>('x'), values.At<double>('y')};
  CHECK((expected_gt - actual).norm() < 1e-3);

  // Check success
  CHECK(stats.status == sym::optimization_status_t::SUCCESS);
}

/**
 * Test manifold optimization of Pose3 in a simple chain where we have priors at the start and end
 * and between factors in the middle. When the priors are strong it should act as on-manifold
 * interpolation, and when the between factors are strong it should act as a mean.
 */
TEST_CASE("Test pose smoothing", "[optimizer]") {
  // Constants
  const double epsilon = 1e-10;
  const int num_keys = 10;

  // Costs
  const double prior_start_sigma = 0.1;  // [rad]
  const double prior_last_sigma = 0.1;   // [rad]
  const double between_sigma = 0.5;      // [rad]

  // Set points (doing 180 rotation and 5m translation to keep it hard)
  const sym::Pose3d prior_start(sym::Rot3d::FromYawPitchRoll(0.0, 0.0, 0.0),
                                Eigen::Vector3d::Zero());
  const sym::Pose3d prior_last(sym::Rot3d::FromYawPitchRoll(M_PI, 0.0, 0.0),
                               Eigen::Vector3d(5, 0, 0));

  // Simple wrapper to add a prior
  // TODO(hayk): Make a template specialization mechanism to generalize this to any geo type.
  const auto create_prior_factor = [&epsilon](const sym::Key& key, const sym::Pose3d& prior,
                                              const double sigma) {
    return sym::Factord::Jacobian(
        [&prior, sigma, &epsilon](const sym::Pose3d& pose, sym::Vector6d* const res,
                                  sym::Matrix66d* const jac) {
          const sym::Matrix66d sqrt_info = sym::Vector6d::Constant(1 / sigma).asDiagonal();
          sym::PriorFactorPose3<double>(pose, prior, sqrt_info, epsilon, res, jac);
        },
        {key});
  };

  // Add priors
  std::vector<sym::Factord> factors;
  factors.push_back(create_prior_factor({'P', 0}, prior_start, prior_start_sigma));
  factors.push_back(create_prior_factor({'P', num_keys - 1}, prior_last, prior_last_sigma));

  // Add between factors in a chain
  for (int i = 0; i < num_keys - 1; ++i) {
    factors.push_back(sym::Factord::Jacobian(
        [&between_sigma, &epsilon](const sym::Pose3d& a, const sym::Pose3d& b,
                                   sym::Vector6d* const res,
                                   Eigen::Matrix<double, 6, 12>* const jac) {
          const sym::Matrix66d sqrt_info = sym::Vector6d::Constant(1 / between_sigma).asDiagonal();
          const sym::Pose3d a_T_b = sym::Pose3d::Identity();
          sym::BetweenFactorPose3<double>(a, b, a_T_b, sqrt_info, epsilon, res, jac);
        },
        /* keys */ {{'P', i}, {'P', i + 1}}));
  }

  // Create initial values as random perturbations from the first prior
  sym::Valuesd values;
  std::mt19937 gen(42);
  for (int i = 0; i < num_keys; ++i) {
    const sym::Pose3d value = prior_start.Retract(0.4 * sym::Random<sym::Vector6d>(gen));
    values.Set<sym::Pose3d>({'P', i}, value);
  }

  INFO("Initial values: " << values);
  CAPTURE(prior_start, prior_last);

  // Optimize
  sym::optimizer_params_t params = DefaultLmParams();
  params.iterations = 50;
  params.early_exit_min_reduction = 0.0001;

  sym::Optimizer<double> optimizer(params, factors, epsilon, "sym::Optimize", {},
                                   /* debug_stats */ false, /* check_derivatives */ true,
                                   /* include_jacobians */ true);
  const auto stats = optimizer.Optimize(values);

  INFO("Optimized values: " << values);

  const auto& last_iter = stats.iterations.back();

  // Check successful convergence
  CHECK(last_iter.iteration == 12);
  CHECK(last_iter.current_lambda == Catch::Approx(0.0039).epsilon(1e-1));
  CHECK(last_iter.new_error == Catch::Approx(7.801).epsilon(1e-3));
  CHECK(stats.status == sym::optimization_status_t::SUCCESS);

  // Check that H = J^T J
  const sym::SparseLinearizationd linearization = sym::Linearize<double>(factors, values);
  const Eigen::SparseMatrix<double> jtj =
      linearization.jacobian.transpose() * linearization.jacobian;
  CHECK(linearization.hessian_lower.triangularView<Eigen::Lower>().isApprox(
      jtj.triangularView<Eigen::Lower>(), 1e-6));
}

/**
 * Test manifold optimization of Rot3 in a simple chain where we have priors at the start and end
 * and between factors in the middle. When the priors are strong it should act as on-manifold
 * interpolation, and when the between factors are strong it should act as a mean.
 */
TEST_CASE("Test Rotation smoothing", "[optimizer]") {
  // Constants
  const double epsilon = 1e-15;
  const int num_keys = 10;

  // Costs
  const double prior_start_sigma = 0.1;  // [rad]
  const double prior_last_sigma = 0.1;   // [rad]
  const double between_sigma = 0.5;      // [rad]

  // Set points (doing 180 rotation to keep it hard)
  const sym::Rot3d prior_start = sym::Rot3d::FromYawPitchRoll(0.0, 0.0, 0.0);
  const sym::Rot3d prior_last = sym::Rot3d::FromYawPitchRoll(M_PI, 0.0, 0.0);

  // Simple wrapper to add a prior
  // TODO(hayk): Make a template specialization mechanism to generalize this to any geo type.
  const auto create_prior_factor = [&epsilon](const sym::Key& key, const sym::Rot3d& prior,
                                              const double sigma) {
    return sym::Factord::Jacobian(
        [&prior, sigma, &epsilon](const sym::Rot3d& rot, Eigen::Vector3d* const res,
                                  Eigen::Matrix3d* const jac) {
          const Eigen::Matrix3d sqrt_info = Eigen::Vector3d::Constant(1 / sigma).asDiagonal();
          sym::PriorFactorRot3<double>(rot, prior, sqrt_info, epsilon, res, jac);
        },
        {key});
  };

  // Add priors
  std::vector<sym::Factord> factors;
  factors.push_back(create_prior_factor({'R', 0}, prior_start, prior_start_sigma));
  factors.push_back(create_prior_factor({'R', num_keys - 1}, prior_last, prior_last_sigma));

  // Add between factors in a chain
  for (int i = 0; i < num_keys - 1; ++i) {
    factors.push_back(sym::Factord::Jacobian(
        [&between_sigma, &epsilon](const sym::Rot3d& a, const sym::Rot3d& b,
                                   Eigen::Vector3d* const res,
                                   Eigen::Matrix<double, 3, 6>* const jac) {
          const Eigen::Matrix3d sqrt_info =
              Eigen::Vector3d::Constant(1 / between_sigma).asDiagonal();
          const sym::Rot3d a_T_b = sym::Rot3d::Identity();
          sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, res, jac);
        },
        /* keys */ {{'R', i}, {'R', i + 1}}));
  }

  // Create initial values as random perturbations from the first prior
  sym::Valuesd values;
  std::mt19937 gen(42);
  for (int i = 0; i < num_keys; ++i) {
    const sym::Rot3d value = prior_start.Retract(0.4 * sym::Random<Eigen::Vector3d>(gen));
    values.Set<sym::Rot3d>({'R', i}, value);
  }

  INFO("Initial values: " << values);
  CAPTURE(prior_start, prior_last);

  // Optimize
  sym::optimizer_params_t params = DefaultLmParams();
  params.iterations = 50;
  params.early_exit_min_reduction = 0.0001;

  sym::Optimizer<double> optimizer(params, factors, epsilon);
  const auto stats = optimizer.Optimize(values);

  INFO("Optimized values: " << values);

  const auto& last_iter = stats.iterations.back();

  // Check successful convergence
  CHECK(last_iter.iteration == 6);
  CHECK(last_iter.current_lambda == Catch::Approx(2.4e-4).epsilon(1e-1));
  CHECK(last_iter.new_error == Catch::Approx(2.174).epsilon(1e-3));
  CHECK(stats.status == sym::optimization_status_t::SUCCESS);

  // Check that H = J^T J
  const sym::SparseLinearizationd linearization = sym::Linearize<double>(factors, values);
  const Eigen::SparseMatrix<double> jtj =
      linearization.jacobian.transpose() * linearization.jacobian;
  CHECK(linearization.hessian_lower.triangularView<Eigen::Lower>().isApprox(
      jtj.triangularView<Eigen::Lower>(), 1e-6));
}

/**
 * Test manifold optimization of Rot3
 *
 * Creates a pose graph with every pose connected to every other pose (in both directions) by
 * between factors with a measurement of identity and isotropic covariance
 *
 * First pose is frozen, others are optimized
 *
 * Tests that frozen variables and factors with out-of-order keys work
 */
TEST_CASE("Test nontrivial (frozen, out-of-order) keys", "[optimizer]") {
  // Constants
  const double epsilon = 1e-15;
  const int num_keys = 3;

  // Costs
  const double between_sigma = 0.5;  // [rad]

  std::vector<sym::Factord> factors;

  // Add between factors in both directions
  for (int i = 0; i < num_keys; i++) {
    for (int j = 0; j < num_keys; j++) {
      if (i == j) {
        continue;
      }

      factors.push_back(sym::Factord::Jacobian(
          [&between_sigma, &epsilon](const sym::Rot3d& a, const sym::Rot3d& b,
                                     Eigen::Vector3d* const res,
                                     Eigen::Matrix<double, 3, 6>* const jac) {
            const Eigen::Matrix3d sqrt_info =
                Eigen::Vector3d::Constant(1 / between_sigma).asDiagonal();
            const sym::Rot3d a_T_b = sym::Rot3d::Identity();
            sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, res, jac);
          },
          /* keys */ {{'R', i}, {'R', j}}));
    }
  }

  // Create initial values as random perturbations from the first prior
  sym::Valuesd values;

  std::mt19937 gen(42);

  for (int i = 0; i < num_keys; ++i) {
    const sym::Rot3d value = sym::Rot3d::FromTangent(0.4 * sym::Random<Eigen::Vector3d>(gen));
    values.Set<sym::Rot3d>({'R', i}, value);
  }

  INFO("Initial values: " << values);

  // Optimize
  sym::optimizer_params_t params = DefaultLmParams();
  params.iterations = 50;
  params.early_exit_min_reduction = 0.0001;

  std::vector<sym::Key> optimized_keys;
  for (int i = 1; i < num_keys; i++) {
    optimized_keys.emplace_back('R', i);
  }

  sym::Optimizer<double> optimizer(params, factors, epsilon, "sym::Optimizer", optimized_keys);
  const auto stats = optimizer.Optimize(values);

  INFO("Optimized values: " << values);

  const auto& last_iter = stats.iterations.back();

  // Check successful convergence
  CHECK(last_iter.iteration == 5);
  CHECK(last_iter.current_lambda < 1e-3);
  CHECK(last_iter.new_error < 1e-15);
  CHECK(stats.status == sym::optimization_status_t::SUCCESS);

  // Check that H = J^T J
  const sym::SparseLinearizationd linearization =
      sym::Linearize<double>(factors, values, optimized_keys);
  const Eigen::SparseMatrix<double> jtj =
      linearization.jacobian.transpose() * linearization.jacobian;
  CHECK(linearization.hessian_lower.triangularView<Eigen::Lower>().isApprox(
      jtj.triangularView<Eigen::Lower>(), 1e-6));
}

/**
 * Test that sym::Optimizer can be constructed with different linear solver orderings
 *
 * Currently this is just a check that things compile
 */
TEST_CASE("Check that we can change linear solvers", "[optimizer]") {
  sym::Optimizerd optimizer1(
      DefaultLmParams(), {sym::Factord()}, 1e-10, "sym::Optimizer", {'a'}, false, false, false,
      sym::SparseCholeskySolver<Eigen::SparseMatrix<double>>(
          Eigen::MetisOrdering<Eigen::SparseMatrix<double>::StorageIndex>()));

  sym::Optimizerd optimizer2(
      DefaultLmParams(), {sym::Factord()}, 1e-10, "sym::Optimizer", {'a'}, false, false, false,
      sym::SparseCholeskySolver<Eigen::SparseMatrix<double>>(
          Eigen::NaturalOrdering<Eigen::SparseMatrix<double>::StorageIndex>()));
}

TEST_CASE("Test optimization statuses", "[optimizer]") {
  {
    auto params = sym::DefaultOptimizerParams();

    sym::Valuesd values{};
    values.Set('x', 10.0);

    const auto stats =
        sym::Optimize(params,
                      {sym::Factord::Jacobian(
                          [](const double x, sym::Vector1d* const res, sym::Matrix11d* const jac) {
                            *res << std::pow(x, 9);
                            *jac << 9 * std::pow(x, 8);
                          },
                          {'x'})},
                      values);

    CHECK(stats.status == sym::optimization_status_t::SUCCESS);
    CHECK(stats.failure_reason == sym::levenberg_marquardt_solver_failure_reason_t::INVALID);
  }

  {
    auto params = sym::DefaultOptimizerParams();
    params.iterations = 5;

    sym::Valuesd values{};
    values.Set('x', 10.0);

    const auto stats =
        sym::Optimize(params,
                      {sym::Factord::Jacobian(
                          [](const double x, sym::Vector1d* const res, sym::Matrix11d* const jac) {
                            *res << std::pow(x, 9);
                            *jac << 9 * std::pow(x, 8);
                          },
                          {'x'})},
                      values);

    CHECK(stats.status == sym::optimization_status_t::HIT_ITERATION_LIMIT);
    CHECK(stats.failure_reason == sym::levenberg_marquardt_solver_failure_reason_t::INVALID);
  }

  {
    const auto params = sym::DefaultOptimizerParams();

    sym::Valuesd values{};
    values.Set('x', 10.0);

    const auto stats =
        sym::Optimize(params,
                      {sym::Factord::Jacobian(
                          [](const double x, sym::Vector1d* const res, sym::Matrix11d* const jac) {
                            *res << std::sqrt(std::abs(x));
                            *jac << 1 / (2 * std::sqrt(std::abs(x)));
                          },
                          {'x'})},
                      values);

    CHECK(stats.status == sym::optimization_status_t::FAILED);
    CHECK(stats.failure_reason ==
          sym::levenberg_marquardt_solver_failure_reason_t::LAMBDA_OUT_OF_BOUNDS);
  }
}
