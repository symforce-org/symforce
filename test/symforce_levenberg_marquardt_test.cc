#include <iostream>

#include "../symforce/opt/levenberg_marquardt_solver.h"
#include "../symforce/util/random.h"

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

/**
 * Test that Gauss Newton converges to the exact result for a linear residual where the solution is
 * zero
 *
 * We generate a random matrix J, and then set the residual to r = J * x
 */
template <typename Scalar>
void TestOneIterationGaussNewton() {
  constexpr const int M = 9;
  constexpr const int N = 5;

  std::mt19937 gen(12345);
  const sym::MatrixX<Scalar> J_MN = sym::RandomNormalMatrix<Scalar, M, N>(gen);

  std::cout << "J_MN:\n" << J_MN << std::endl;
  std::cout << "J_MN^T * J_MN:\n" << (J_MN.transpose() * J_MN).eval() << std::endl;

  constexpr const Scalar kEpsilon = 1e-10;

  sym::optimizer_params_t params{};
  params.initial_lambda = 1.0;
  params.lambda_up_factor = 3.0;
  params.lambda_down_factor = 1.0 / 3.0;
  params.lambda_lower_bound = 0.0;
  params.lambda_upper_bound = 0.0;
  params.iterations = 1;
  params.use_diagonal_damping = false;
  params.use_unit_damping = false;
  sym::LevenbergMarquardtSolver<Scalar> solver(params, "", kEpsilon);

  using StateVector = Eigen::Matrix<Scalar, N, 1>;

  auto residual_func = [&](const sym::Values<Scalar>& values,
                           sym::Linearization<Scalar>* const linearization) {
    const auto state_vec = values.template At<StateVector>('v');
    linearization->residual = J_MN * state_vec;
    linearization->hessian_lower = (J_MN.transpose() * J_MN).sparseView();
    linearization->jacobian = J_MN.sparseView();
    linearization->rhs = J_MN.transpose() * linearization->residual;
  };

  sym::Values<Scalar> values_init{};
  values_init.Set('v', (StateVector::Ones() * 100).eval());
  sym::index_t index = values_init.CreateIndex({'v'});
  sym::Linearization<Scalar> linearization{};

  residual_func(values_init, &linearization);
  const sym::VectorX<Scalar> residual_init = linearization.residual;
  const Scalar error_init = 0.5 * residual_init.squaredNorm();
  std::cerr << "values_init: " << values_init << std::endl;
  std::cerr << "residual_init: " << residual_init.transpose() << std::endl;
  std::cerr << "error_init: " << error_init << std::endl;

  solver.SetIndex(index);
  solver.ResetState(values_init);

  // Collect debug stats so that we have the final residual
  const bool debug_stats = true;

  // Do a single gauss-newton iteration
  std::vector<sym::optimizer_iteration_t> iterations;
  solver.Iterate(residual_func, &iterations, debug_stats);

  const sym::VectorX<Scalar> residual_final = iterations.back().residual.template cast<Scalar>();
  const Scalar error_final = 0.5 * residual_final.squaredNorm();
  std::cerr << "values_final: " << solver.GetBestValues() << std::endl;
  std::cerr << "residual_final: " << residual_final.transpose() << std::endl;
  std::cerr << "error_final: " << error_final << std::endl;

  // Check initial error was high and final is zero
  assertTrue(error_init > 10000.);
  assertTrue(error_final < 1e-8);

  // Check solution is zero
  assertTrue(solver.GetBestValues().template At<StateVector>('v').norm() < 1e-4);
}

int main() {
  TestOneIterationGaussNewton<double>();
  TestOneIterationGaussNewton<float>();
}
