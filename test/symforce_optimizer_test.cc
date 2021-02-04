#include <iostream>

#include <sym/factors/between_factor_rot3.h>
#include <sym/factors/prior_factor_rot3.h>

#include "../symforce/opt/optimizer.h"
#include "../symforce/util/random.h"

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

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
void TestNonlinear() {
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
  std::cout << "Initial values: " << values << std::endl;

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
  Optimize(params, factors, &values);

  // Check results
  std::cout << "Optimized values: " << values << std::endl;

  // Local minimum from Wolfram Alpha
  // pylint: disable=line-too-long
  // https://www.wolframalpha.com/input/?i=z+%3D+3+-+0.5+*+sin((x+-+2)+%2F+5)+%2B+1.0+*+sin((y+%2B+2)+%2F+10.)
  const Eigen::Vector2d expected_gt = {9.854, -17.708};
  const Eigen::Vector2d actual = {values.At<double>('x'), values.At<double>('y')};
  assertTrue((expected_gt - actual).norm() < 1e-3);
}

/**
 * Test manifold optimization of Rot3 in a simple chain where we have priors at the start and end
 * and between factors in the middle. When the priors are strong it should act as on-manifold
 * interpolation, and when the between factors are strong it should act as a mean.
 */
void TestRotSmoothing() {
  // Constants
  const double epsilon = 1e-15;
  const int num_keys = 10;

  // Costs
  const double prior_start_sigma = 0.1;  // [rad]
  const double prior_last_sigma = 0.1;   // [rad]
  const double between_sigma = 0.5;      // [rad]

  // Set points (doing 180 rotation to keep it hard)
  const geo::Rot3d prior_start = geo::Rot3d::FromYawPitchRoll(0.0, 0.0, 0.0);
  const geo::Rot3d prior_last = geo::Rot3d::FromYawPitchRoll(M_PI, 0.0, 0.0);

  // Simple wrapper to add a prior
  // TODO(hayk): Make a template specialization mechanism to generalize this to any geo type.
  const auto create_prior_factor = [&epsilon](const sym::Key& key, const geo::Rot3d& prior,
                                              const double sigma) {
    return sym::Factord::Jacobian(
        [&prior, sigma, &epsilon](const geo::Rot3d& rot, Eigen::Vector3d* const res,
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
        [&between_sigma, &epsilon](const geo::Rot3d& a, const geo::Rot3d& b,
                                   Eigen::Vector3d* const res,
                                   Eigen::Matrix<double, 3, 6>* const jac) {
          const Eigen::Matrix3d sqrt_info =
              Eigen::Vector3d::Constant(1 / between_sigma).asDiagonal();
          const geo::Rot3d a_T_b = geo::Rot3d::Identity();
          sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, res, jac);
        },
        /* keys */ {{'R', i}, {'R', i + 1}}));
  }

  // Create initial values as random pertubations from the first prior
  sym::Valuesd values;
  std::srand(0);
  for (int i = 0; i < num_keys; ++i) {
    const geo::Rot3d value = prior_start.Retract(0.4 * Eigen::Vector3d::Random());
    values.Set<geo::Rot3d>({'R', i}, value);
  }

  std::cout << "Initial values: " << values << std::endl;
  std::cout << "Prior on R0: " << prior_start << std::endl;
  std::cout << "Prior on R[-1]: " << prior_last << std::endl;

  // Optimize
  sym::optimizer_params_t params = DefaultLmParams();
  params.iterations = 50;
  params.early_exit_min_reduction = 0.0001;

  sym::Optimizer<double> optimizer(params, factors, epsilon);
  optimizer.Optimize(&values);

  std::cout << "Optimized values: " << values << std::endl;

  const auto& iteration_stats = optimizer.IterationStats();
  const auto& last_iter = iteration_stats[iteration_stats.size() - 1];
  std::cout << "Iterations: " << last_iter.iteration << std::endl;
  std::cout << "Lambda: " << last_iter.current_lambda << std::endl;
  std::cout << "Final error: " << last_iter.new_error << std::endl;

  // Check successful convergence
  assertTrue(last_iter.iteration == 7);
  assertTrue(fabs(last_iter.current_lambda - 6.1e-5) < 1e-6);
  assertTrue(fabs(last_iter.new_error - 2.174) < 1e-3);

  // Check that H = J^T J
  const sym::Linearizationd linearization = sym::Linearize<double>(factors, values);
  const Eigen::SparseMatrix<double> jtj =
      linearization.jacobian.transpose() * linearization.jacobian;
  assertTrue(linearization.hessian_lower.triangularView<Eigen::Lower>().isApprox(
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
void TestNontrivialKeys() {
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
          [&between_sigma, &epsilon](const geo::Rot3d& a, const geo::Rot3d& b,
                                     Eigen::Vector3d* const res,
                                     Eigen::Matrix<double, 3, 6>* const jac) {
            const Eigen::Matrix3d sqrt_info =
                Eigen::Vector3d::Constant(1 / between_sigma).asDiagonal();
            const geo::Rot3d a_T_b = geo::Rot3d::Identity();
            sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, res, jac);
          },
          /* keys */ {{'R', i}, {'R', j}}));
    }
  }

  // Create initial values as random pertubations from the first prior
  sym::Valuesd values;

  std::mt19937 gen(42);

  for (int i = 0; i < num_keys; ++i) {
    const geo::Rot3d value = geo::Rot3d::FromTangent(0.4 * sym::RandomNormalVector<double, 3>(gen));
    values.Set<geo::Rot3d>({'R', i}, value);
  }

  std::cout << "Initial values: " << values << std::endl;

  // Optimize
  sym::optimizer_params_t params = DefaultLmParams();
  params.iterations = 50;
  params.early_exit_min_reduction = 0.0001;

  std::vector<sym::Key> optimized_keys;
  for (int i = 1; i < num_keys; i++) {
    optimized_keys.emplace_back('R', i);
  }

  sym::Optimizer<double> optimizer(params, factors, epsilon, optimized_keys);
  optimizer.Optimize(&values);

  std::cout << "Optimized values: " << values << std::endl;

  const auto& iteration_stats = optimizer.IterationStats();
  const auto& last_iter = iteration_stats[iteration_stats.size() - 1];
  std::cout << "Iterations: " << last_iter.iteration << std::endl;
  std::cout << "Lambda: " << last_iter.current_lambda << std::endl;
  std::cout << "Final error: " << last_iter.new_error << std::endl;

  // Check successful convergence
  assertTrue(last_iter.iteration == 5);
  assertTrue(last_iter.current_lambda < 1e-3);
  assertTrue(last_iter.new_error < 1e-15);

  // Check that H = J^T J
  const sym::Linearizationd linearization = sym::Linearize<double>(factors, values, optimized_keys);
  const Eigen::SparseMatrix<double> jtj =
      linearization.jacobian.transpose() * linearization.jacobian;
  assertTrue(linearization.hessian_lower.triangularView<Eigen::Lower>().isApprox(
      jtj.triangularView<Eigen::Lower>(), 1e-6));
}

int main(int argc, char** argv) {
  TestNonlinear();

  TestRotSmoothing();

  TestNontrivialKeys();
}
