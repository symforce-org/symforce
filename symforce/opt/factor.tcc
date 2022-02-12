/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./assert.h"
#include "./factor.h"
#include "./internal/factor_utils.h"

namespace sym {

template <typename Scalar>
Factor<Scalar>::Factor(HessianFunc&& hessian_func, const std::vector<Key>& keys)
    : Factor(std::move(hessian_func), keys, keys) {}

template <typename Scalar>
Factor<Scalar>::Factor(HessianFunc&& hessian_func, const std::vector<Key>& keys_to_func,
                       const std::vector<Key>& keys_to_optimize)
    : hessian_func_(std::move(hessian_func)),
      sparse_hessian_func_(),
      is_sparse_(false),
      keys_to_optimize_(keys_to_optimize),
      keys_(keys_to_func) {}

template <typename Scalar>
Factor<Scalar>::Factor(SparseHessianFunc&& sparse_hessian_func, const std::vector<Key>& keys)
    : Factor(std::move(sparse_hessian_func), keys, keys) {}

template <typename Scalar>
Factor<Scalar>::Factor(SparseHessianFunc&& sparse_hessian_func,
                       const std::vector<Key>& keys_to_func,
                       const std::vector<Key>& keys_to_optimize)
    : hessian_func_(),
      sparse_hessian_func_(std::move(sparse_hessian_func)),
      is_sparse_(true),
      keys_to_optimize_(keys_to_optimize),
      keys_(keys_to_func) {}

// ------------------------------------------------------------------------------------------------
// Factor::Jacobian constructor
//
// This constructor can accept multiple forms of functions. It introspects the function to
// understand its input/output arguments, performs a series of static assertions, and dispatches
// to the correct helper to be processed.
// ------------------------------------------------------------------------------------------------

template <typename Scalar>
template <typename Functor>
Factor<Scalar> Factor<Scalar>::Jacobian(Functor func, const std::vector<Key>& keys) {
  return Jacobian(func, keys, keys);
}

template <typename Scalar>
template <typename Functor>
Factor<Scalar> Factor<Scalar>::Jacobian(Functor func, const std::vector<Key>& keys_to_func,
                                        const std::vector<Key>& keys_to_optimize) {
  using Traits = function_traits<Functor>;

  SYM_ASSERT(keys_to_func.size() >= keys_to_optimize.size());
  SYM_ASSERT(Traits::num_arguments == keys_to_func.size() + 2);

  // Get matrix types from function signature
  using ResidualVec = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 2>::type>::type;
  using JacobianMat = typename std::remove_pointer<
      typename Traits::template arg<Traits::num_arguments - 1>::type>::type;

  // Check that they're Eigen matrices (nice for error messages)
  static_assert(kIsEigenType<ResidualVec>,
                "ResidualVec (2nd from last argument) should be an Eigen::Matrix");
  static_assert(kIsEigenType<JacobianMat> || kIsSparseEigenType<JacobianMat>,
                "JacobianMat (last argument) should be an Eigen::Matrix or Eigen::SparseMatrix");

  // Check sparse vs dense
  constexpr bool is_sparse = !kIsEigenType<JacobianMat>;

  // Get dimensions
  constexpr int M = ResidualVec::RowsAtCompileTime;
  constexpr int N = JacobianMat::ColsAtCompileTime;
  static_assert(JacobianMat::RowsAtCompileTime == M || is_sparse, "Inconsistent sizes.");
  static_assert(ResidualVec::ColsAtCompileTime == 1, "Inconsistent sizes.");

  static_assert(
      !(is_sparse && M != Eigen::Dynamic),
      "Can't have a fixed-size residual with sparse jacobian for Factor::Jacobian, because we "
      "can't deduce the Rhs size.  Please use a dynamic-size residual.");

  constexpr bool is_dynamic = (M == Eigen::Dynamic) && (N == Eigen::Dynamic);
  constexpr bool is_fixed = (M != Eigen::Dynamic) && (N != Eigen::Dynamic);
  static_assert((is_dynamic || is_fixed), "Matrices cannot be mixed fixed and dynamic.");

  // Dispatch to either the dynamic size or fixed size implementations
  return internal::JacobianDispatcher<is_dynamic, is_sparse, Scalar>{}(func, keys_to_func,
                                                                       keys_to_optimize);
}

// ------------------------------------------------------------------------------------------------
// Factor::Hessian constructor
//
// This constructor can accept multiple forms of functions. It introspects the function to
// understand its input/output arguments, performs a series of static assertions, and dispatches
// to the correct helper to be processed.
// ------------------------------------------------------------------------------------------------

template <typename Scalar>
template <typename Functor>
Factor<Scalar> Factor<Scalar>::Hessian(Functor func, const std::vector<Key>& keys) {
  return Hessian(func, keys, keys);
}

template <typename Scalar>
template <typename Functor>
Factor<Scalar> Factor<Scalar>::Hessian(Functor func, const std::vector<Key>& keys_to_func,
                                       const std::vector<Key>& keys_to_optimize) {
  using Traits = function_traits<Functor>;

  SYM_ASSERT(keys_to_func.size() >= keys_to_optimize.size());
  SYM_ASSERT(Traits::num_arguments == keys_to_func.size() + 4);

  // Get matrix types from function signature
  using ResidualVec = typename internal::HessianFuncTypeHelper<Functor>::ResidualVec;
  using JacobianMat = typename internal::HessianFuncTypeHelper<Functor>::JacobianMat;
  using HessianMat = typename internal::HessianFuncTypeHelper<Functor>::HessianMat;
  using RhsVec = typename internal::HessianFuncTypeHelper<Functor>::RhsVec;

  // Check that they're Eigen matrices (nice for error messages)
  static_assert(kIsEigenType<ResidualVec>,
                "ResidualVec (4th from last argument) should be an Eigen::Matrix");
  static_assert(
      kIsEigenType<JacobianMat> || kIsSparseEigenType<JacobianMat>,
      "JacobianMat (3rd from last argument) should be an Eigen::Matrix or Eigen::SparseMatrix");
  static_assert(
      kIsEigenType<HessianMat> || kIsSparseEigenType<HessianMat>,
      "HessianMat (2nd from last argument) should be an Eigen::Matrix or Eigen::SparseMatrix");
  static_assert(kIsEigenType<RhsVec>, "RhsVec (last argument) should be an Eigen::Matrix");

  // Check sparse vs dense
  constexpr bool jacobian_is_sparse = !kIsEigenType<JacobianMat>;
  constexpr bool hessian_is_sparse = !kIsEigenType<HessianMat>;
  static_assert(jacobian_is_sparse == hessian_is_sparse,
                "Matrices cannot be mixed dense and sparse.");

  // Get dimensions
  constexpr int M = ResidualVec::RowsAtCompileTime;
  constexpr int N = RhsVec::RowsAtCompileTime;
  static_assert(ResidualVec::ColsAtCompileTime == 1, "Inconsistent sizes.");
  static_assert(JacobianMat::RowsAtCompileTime == M || jacobian_is_sparse, "Inconsistent sizes.");
  static_assert(JacobianMat::ColsAtCompileTime == N || jacobian_is_sparse, "Inconsistent sizes.");
  static_assert(HessianMat::RowsAtCompileTime == N || hessian_is_sparse, "Inconsistent sizes.");
  static_assert(HessianMat::ColsAtCompileTime == N || hessian_is_sparse, "Inconsistent sizes.");
  static_assert(RhsVec::ColsAtCompileTime == 1, "Inconsistent sizes.");

  constexpr bool is_dynamic = (M == Eigen::Dynamic) && (N == Eigen::Dynamic);
  constexpr bool is_fixed = (M != Eigen::Dynamic) && (N != Eigen::Dynamic);
  static_assert((is_dynamic || is_fixed), "Matrices cannot be mixed fixed and dynamic.");

  // Dispatch to either the dynamic size or fixed size implementations
  return internal::HessianDispatcher<is_dynamic, jacobian_is_sparse, Scalar>{}(func, keys_to_func,
                                                                               keys_to_optimize);
}

// ----------------------------------------------------------------------------
// LCM type aliases
// ----------------------------------------------------------------------------

template <>
struct LinearizedDenseFactorTypeHelper<double> {
  using Type = linearized_dense_factor_t;
};

template <>
struct LinearizedDenseFactorTypeHelper<float> {
  using Type = linearized_dense_factorf_t;
};

template <>
struct LinearizedSparseFactorTypeHelper<double> {
  using Type = linearized_sparse_factor_t;
};

template <>
struct LinearizedSparseFactorTypeHelper<float> {
  using Type = linearized_sparse_factorf_t;
};

}  // namespace sym
