#pragma once

#include <unordered_set>

#include <Eigen/Sparse>

#include <lcmtypes/sym/linearization_factor_helper_t.hpp>

#include "./factor.h"
#include "./values.h"

namespace sym {

/**
 * The result of evaluating multiple Factors at the linearization point given by a Values.
 *
 * Stores the original Factors as well as the LinearizedFactors, and provides tools for
 * aggregating keys and building a large jacobian / hessian for optimization.
 *
 * For efficiency, prefer calling Relinearize instead of re-constructing this object!
 */
template <typename ScalarType>
class Linearization {
 public:
  using Scalar = ScalarType;
  using LinearizedFactor = typename Factor<Scalar>::LinearizedFactor;

  Linearization() {}

  /**
   * Construct a linearization from factors and a values object. This will allocate the
   * appropriate memory for repeated re-linearization, and perform the first linearization.
   *
   * Args:
   *     factors: Only stores a pointer, MUST be in scope for the lifetime of this object!
   *     values: This is used to evaluate factors, but not copied or stored.
   *     key_order: If provided, acts as an ordered set of keys that form the state vector
   *                to optimize. Must equal the set of all factor keys. If not provided,
   *                it is computed from the factors using a default ordering.
   */
  Linearization(const std::vector<Factor<Scalar>>& factors, const Values<Scalar>& values,
                const std::vector<Key>& key_order = {});

  /**
   * Update linearization at a new evaluation point. Returns the total residual dimension M.
   * This is more efficient than reconstructing this object repeatedly.
   */
  void Relinearize(const Values<Scalar>& values);

  /**
   * Basic accessors.
   */
  const std::vector<LinearizedFactor>& LinearizedFactors() const;
  const std::vector<Key>& Keys() const;
  const VectorX<Scalar>& Residual() const;
  const VectorX<Scalar>& Rhs() const;
  const Eigen::SparseMatrix<Scalar>& JacobianSparse() const;
  const Eigen::SparseMatrix<Scalar>& HessianLowerSparse() const;

 private:
  /**
   * Whether this contains values, versus having only been default constructed.
   */
  bool IsInitialized() const;

  /**
   * Allocate all factor storage and compute sparsity pattern. This does a lot of index
   * computation on the first linearization, such that repeated linearization can be fast.
   */
  void InitializeStorageAndIndices();

  /**
   * Hashmap of keys to information about the key's offset in the full problem.
   */
  static std::unordered_map<key_t, index_entry_t> ComputeStateIndex(
      const std::vector<LinearizedFactor>& factors, const std::vector<Key>& keys);

  /**
   * Update the sparse combined problem linearization from a single factor.
   */
  void UpdateFromLinearizedFactorIntoSparse(const LinearizedFactor& linearized_factor,
                                            const linearization_factor_helper_t& factor_helper);
  /**
   * Update the dense combined problem linearization from a single factor.
   */
  void UpdateFromLinearizedFactorIntoDense(const LinearizedFactor& linearized_factor,
                                           const linearization_factor_helper_t& factor_helper);

  /**
   * Update the combined sparse problem linearization from linearized factors by directly
   * updating the indices of the sparse storage.
   */
  void BuildCombinedProblemSparse(const std::vector<LinearizedFactor>& linearized_factors);

  /**
   * Update the combined sparse problem linearization from linearized factors by first
   * updating a combined dense linearization, then taking a sparse view.
   * NOTE(hayk): This should be less efficient and can lead to the sparsity pattern changing
   * if the combined jacobian/hessian happen to have zero entries in symbolically nonzero fields.
   */
  void BuildCombinedProblemDenseThenSparseView(
      const std::vector<LinearizedFactor>& linearized_factors);

  // Pointer to the nonlinear factors
  const std::vector<Factor<Scalar>>* factors_;

  // Linearized factors - stores individual factor residuals, jacobians, etc
  std::vector<LinearizedFactor> linearized_factors_;

  // Keys that form the state vector
  std::vector<Key> keys_;

  // Helpers for updating the combined problem from linearized factors
  std::vector<linearization_factor_helper_t> factor_update_helpers_;

  // Combined linearization
  VectorX<Scalar> residual_;
  VectorX<Scalar> rhs_;
  Eigen::SparseMatrix<Scalar> jacobian_sparse_;
  Eigen::SparseMatrix<Scalar> hessian_lower_sparse_;

  // Dense problem jacobian/hessian
  // Currently only used to compute initial sparsity pattern.
  // TODO(hayk): Kill these, for very large problems don't want to allocate.
  MatrixX<Scalar> jacobian_;
  MatrixX<Scalar> hessian_lower_;
};

// Shorthand instantiations
using Linearizationd = Linearization<double>;
using Linearizationf = Linearization<float>;

// Free function as an alternate way to call.
template <typename Scalar>
Linearization<Scalar> Linearize(const std::vector<Factor<Scalar>>& factors,
                                const Values<Scalar>& values) {
  return Linearization<Scalar>(factors, values);
}

/**
 * Compute the combined set of keys to optimize from the given factors. Order using the given
 * comparison function.
 */
template <typename Scalar, typename Compare>
std::vector<Key> ComputeKeysToOptimize(const std::vector<Factor<Scalar>>& factors,
                                       Compare key_compare) {
  // Some thoughts on efficiency at
  // https://stackoverflow.com/questions/1041620/whats-the-most-efficient-way-to-erase-duplicates-and-sort-a-vector

  // Aggregate uniques
  std::unordered_set<Key> key_set;
  for (const Factor<Scalar>& factor : factors) {
    key_set.insert(factor.Keys().begin(), factor.Keys().end());
  }

  // Copy to vector
  std::vector<Key> keys;
  keys.insert(keys.end(), key_set.begin(), key_set.end());

  // Order
  std::sort(keys.begin(), keys.end(), key_compare);

  return keys;
}

}  // namespace sym
