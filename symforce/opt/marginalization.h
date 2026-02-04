/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <unordered_set>
#include <variant>

#include <lcmtypes/sym/marginalization_factor_t.hpp>
#include <lcmtypes/sym/marginalization_factorf_t.hpp>

#include "./factor.h"
#include "./values.h"

namespace sym {

namespace internal {

template <typename Scalar>
struct MarginalizationFactorLcmType {};

template <>
struct MarginalizationFactorLcmType<double> {
  using Type = marginalization_factor_t;
};

template <>
struct MarginalizationFactorLcmType<float> {
  using Type = marginalization_factorf_t;
};

template <typename Scalar>
using MarginalizationFactorLcmTypeT = typename MarginalizationFactorLcmType<Scalar>::Type;

}  // namespace internal

/**
 * Marginalization factors are linear approximations of information we remove from the optimization
 * problem to bound its size. The struct contains all the data needed to create a marginalization
 * factor.
 *
 * We are storing the factor in a general quadratic form which will be a term used in a non-linear
 * least squares optimization. Taking inspiration from the LevenbergMarquardtSolver doc string:
 * e(x) ~= 0.5 * dx.T * J.T * J * dx + b.T * J * dx + 0.5 * b.T * b
 * where J is the Jacobian of the residual function f(x) and b = f(x0) is the residual at the
 * linearization point.
 *
 * We store H = J.T * J (ie. the Gauss-Newton approximation of the Hessian),
 * rhs = J.T * b, c = b.T * b (dropping the 0.5 factor by convention). This simplifies to:
 * e(x) ~= 0.5 * dx.T * H * dx + rhs.T * dx + 0.5 * c
 * We end up solving the larger system in the form of H' * x = b' to find the optimal value of x.
 */
template <typename ScalarType>
struct MarginalizationFactor {
  using Scalar = ScalarType;
  using LcmType = internal::MarginalizationFactorLcmTypeT<Scalar>;

  MatrixX<Scalar> H{};                    // Hessian = J.T * J
  VectorX<Scalar> rhs{};                  // RHS = J.T * b
  Scalar c{};                             // f = b.T*b
  Values<Scalar> linearization_values{};  // values used for linearization
  std::vector<Key> keys;                  // keys remaining in the problem that this factor touches

  static MarginalizationFactor FromLcmType(const LcmType& msg);

  LcmType GetLcmType() const;
};

using MarginalizationFactorf = MarginalizationFactor<float>;
using MarginalizationFactord = MarginalizationFactor<double>;

// Explicit instantiations
extern template struct MarginalizationFactor<float>;
extern template struct MarginalizationFactor<double>;

// The marginalization operation computes the Schur complement of the linearized system. To do this
// we must group all the keys that we want to marginalize together. Arbitrarily, we choose
// marginalized keys first.
[[nodiscard]] std::vector<Key> ComputeMarginalizationKeyOrder(
    const std::unordered_set<Key>& keys_to_optimize,
    const std::unordered_set<Key>& keys_to_marginalize);

// See Marginalize function for the details on the math. Assumes H is symmetric and lower-only.
template <typename Scalar>
[[nodiscard]] std::variant<MarginalizationFactor<Scalar>, Eigen::ComputationInfo>
ComputeSchurComplement(const MatrixX<Scalar>& H, const VectorX<Scalar>& rhs, Scalar c,
                       int delimiter);

/**
 * Given the set of factors and keys to marginalize, computes the data needed for a marginalization
 * factor. This assumes all factors passed in should be included in the computation.
 *
 * Given the factors, we compute a linearization at the provided values. The system becomes:
 * E = 0.5 | x_u x_l | * | H_uu H_ul | * | x_u | + | rhs_u rhs_l | * | x_u | + 0.5 C
 *                       | H_lu H_ll |   | x_l |                     | x_l |
 * where x_u are the states to be marginalized and x_l are the Markov blanket (the states that
 * remain and have factors that connect to the marginalized states). We use the the Schur complement
 * to eliminate x_u, giving the final system:
 * E = 0.5 x_l.T * H * x_l + rhs.T * x_l + 0.5 * c', where
 * H = H_ll - H_lu * H_uu^{-1} * H_ul = H_ll - H_ul.T * H_uu^{-1} * H_ul
 * rhs = rhs_l - H_lu * H_uu^{-1} * rhs_u = rhs_l - H_ul.T * H_uu^{-1} * rhs_u
 * c' = c - rhs_u.T * A^{-1} rhs_u
 * There are a few references for the H and rhs expressions above (ex: OKVIS paper). For the
 * constant term, find the optimum for x_u (by taking the partial derivative wrt x_u), then
 * x_u* = - A^{-1} * ( B * x_l + rhs_u ) (after substituting and simplifying).
 */
template <typename Scalar>
[[nodiscard]] std::variant<MarginalizationFactor<Scalar>, Eigen::ComputationInfo> Marginalize(
    const std::vector<Factor<Scalar>>& factors, const Values<Scalar>& values,
    const std::unordered_set<Key>& keys_to_optimize,
    const std::unordered_set<Key>& keys_to_marginalize);

/**
 * Create a symforce::Factord representing the marginalization prior to be used in an optimization
 * or future marginalization operation. Building on the derivation above, we can substitute in a
 * new dx = dx' + delta_x and simplify to get:
 * e(x) ~= 0.5 * (dx' + delta_x).T * H * (dx' + delta_x) + rhs.T * (dx' + delta_x) + 0.5 * f
 *       = 0.5 * dx'.T * H * dx' + (rhs + H * delta_x).T * dx'
 *         + 0.5 * (delta_x.T * H * delta_x + 2 * rhs.T * delta_x + c)
 * Thus, the Hessian remains unchanged, the updated rhs is (rhs + H * delta_x), and the updated
 * constant term is (delta_x.T * H * delta_x + 2 * rhs.T * delta_x + c).
 */
template <typename Scalar>
[[nodiscard]] Factor<Scalar> CreateMarginalizationFactor(
    const MarginalizationFactor<Scalar>& marginalization_factor);

}  // namespace sym
