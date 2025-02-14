/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <sym/util/epsilon.h>

#include "./levenberg_marquardt_solver.h"
#include "./optimization_stats.h"

namespace sym {

/**
 * Class for optimizing a nonlinear least-squares problem given a linearization function and an
 * optimization state which may be user-defined. This class largely mirrors the Optimizer
 * class, but differs in the following key ways:
 *
 *    - User-defined state (of type ValuesType) can be used in place of Values objects.
 *    - A user-defined linearization function can be used in place of the Linearizer class (thus,
 *      constructing a list of symforce Factor objects is not required).
 *
 * Compared to the Optimizer class, these differences allow for more efficient optimization of
 * problems whose structure is know at compile-time by avoiding overhead related to working with
 * dynamically-sized problems. Thus, this class should be preferred over the Optimizer class when
 * the problem structure is known at compile-time.
 *
 * Example of user-defined types:
 *
 *     // User-defined values type to store optimized states + constants
 *     struct CustomValuesType {...};
 *
 *     // User defined optimization state
 *     class CustomOptimizationState : public
 *          sym::internal::LevenbergMarquardtStateBase<CustomOptimizationState, CustomValuesType,
 *                                                     Eigen::SparseMatrix<double>> {
 *      public:
 *       // Retracts `Init()` using `update` and stores the result in `New()`
 *       void UpdateNewFromInitImpl(const sym::VectorX<Scalar>& update, const Scalar epsilon) {...}
 *
 *       // Optionally returns serialized LCM type to store in iteration states if debugging is
 *       // enabled
 *       sym::values_t GetLcmTypeImpl(const ValuesType& values) const {...}
 *     };
 *
 *     // User-defined linearization function
 *     void LinearizeCustomValues(const CustomValuesType& values,
 *                                sym::SparseLinearizationd& linearization) {...}
 *
 * Example usage:
 *
 *     using LinearSolverType = sym::SparseCholeskySolver<Eigen::SparseMatrix<double>>;
 *     using CustomNonlinearSolver =
 *         sym::LevenbergMarquardtSolver<double, LinearSolverType, CustomOptimizationState>;
 *     using CustomOptimizer = sym::FixedSizeOptimizer<double, CustomNonlinearSolver>;
 *
 *     CustomOptimizer optimizer(DefaultLmParams());
 *     CustomValuesType custom_values{...};
 *     optimizer.Optimize(custom_values, LinearizeCustomValues);
 *
 * See symforce/test/symforce_fixed_size_optimizer_test.cc for more examples.
 */
template <typename ScalarType, typename _NonlinearSolverType = LevenbergMarquardtSolver<ScalarType>>
class FixedSizeOptimizer {
 public:
  using Scalar = ScalarType;
  using NonlinearSolverType = _NonlinearSolverType;
  using FailureReason = typename NonlinearSolverType::FailureReason;
  using MatrixType = typename NonlinearSolverType::MatrixType;
  using Stats = OptimizationStats<MatrixType>;
  using LinearizeFunc = typename NonlinearSolverType::LinearizeFunc;
  using ValuesType = typename NonlinearSolverType::ValuesType;

  FixedSizeOptimizer(const optimizer_params_t& params,
                     const std::string& name = "sym::FixedSizeOptimizer",
                     const Scalar epsilon = sym::kDefaultEpsilon<Scalar>);

  template <typename... NonlinearSolverArgs>
  FixedSizeOptimizer(const optimizer_params_t& params, const std::string& name,
                     const Scalar epsilon, NonlinearSolverArgs&&... nonlinear_solver_args);

  /**
   * Optimize the given values in-place using the given linearization function
   *
   * @param num_iterations: If < 0 (the default), uses the number of iterations specified by the
   *    params at construction
   * @param populate_best_linearization: If true, the linearization at the best
   *    values will be filled out in the stats
   *
   * @returns The optimization stats
   */
  Stats Optimize(ValuesType& values, const LinearizeFunc& linearize_func,
                 const int num_iterations = -1, const bool populate_best_linearization = false);

  /**
   * Optimize the given values in-place using the given linearization function
   *
   * This overload takes the stats as an argument, and stores into there. This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.
   *
   * @param num_iterations: If < 0 (the default), uses the number of iterations specified by the
   *    params at construction
   * @param populate_best_linearization: If true, the linearization at the best values will be
   *    filled out in the stats
   * @param stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *    allocated fields here, will not reallocate if memory is already allocated in the required
   *    shape (e.g. for repeated calls to Optimize())
   */
  void Optimize(ValuesType& values, const LinearizeFunc& linearize_func, int num_iterations,
                bool populate_best_linearization, Stats& stats);

  /**
   * Optimize the given values in-place using the given linearization function
   *
   * This overload takes the stats as an argument, and stores into there.  This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.
   *
   * @param num_iterations: If < 0 (the default), uses the number of iterations specified by the
   *    params at construction
   * @param stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *    allocated fields here, will not reallocate if memory is already allocated in the
   *    required shape (e.g. for repeated calls to Optimize())
   */
  void Optimize(ValuesType& values, const LinearizeFunc& linearize_func, int num_iterations,
                Stats& stats);

  /**
   * Optimize the given values in-place using the given linearization function
   *
   * This overload takes the stats as an argument, and stores into there.  This allows users to
   * avoid reallocating memory for any of the entries in the stats, for use cases where that's
   * important.
   *
   * @param stats: An OptimizationStats to fill out with the result - if filling out dynamically
   *    allocated fields here, will not reallocate if memory is already allocated in the
   *    required shape (e.g. for repeated calls to Optimize())
   */
  void Optimize(ValuesType& values, const LinearizeFunc& linearize_func, Stats& stats);

  /**
   * Get the full problem covariance at the given linearization.
   *
   * The ordering of entries is the same as the order in the hessian computed by the linearization
   * function.
   *
   * @param covariance A matrix that will be filled out with the full problem covariance.
   */
  void ComputeFullCovariance(const Linearization<MatrixType>& linearization,
                             MatrixX<Scalar>& covariance);

  /**
   * Get the NonlinearSolver object
   */
  const NonlinearSolverType& NonlinearSolver() const;
  NonlinearSolverType& NonlinearSolver();

  /**
   * Update the optimizer params
   */
  void UpdateParams(const optimizer_params_t& params);

  /**
   * Get the params used by the nonlinear solver
   */
  const optimizer_params_t& Params() const;

 private:
  /// The name of this optimizer to be used for printing debug information.
  std::string name_;

  /// Underlying nonlinear solver class.
  NonlinearSolverType nonlinear_solver_;

  Scalar epsilon_;
  bool debug_stats_;
  bool include_jacobians_;

  bool verbose_;
};

// Shorthand instantiations
using FixedSizeOptimizerd = FixedSizeOptimizer<double>;
using FixedSizeOptimizerf = FixedSizeOptimizer<float>;

}  // namespace sym

#include "./fixed_size_optimizer.tcc"

// Explicit instantiation declaration
extern template class sym::FixedSizeOptimizer<double>;
extern template class sym::FixedSizeOptimizer<float>;
