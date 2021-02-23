#pragma once

#include <unordered_set>

#include <Eigen/Sparse>

#include <lcmtypes/sym/linearization_factor_helper_t.hpp>

#include "./factor.h"
#include "./linearization.h"
#include "./values.h"

namespace sym {

/**
 * Class for evaluating multiple Factors at the linearization point given by a Values.
 *
 * Stores the original Factors as well as the LinearizedFactors, and provides tools for
 * aggregating keys and building a large jacobian / hessian for optimization.
 *
 * For efficiency, prefer calling Relinearize instead of re-constructing this object!
 */
template <typename ScalarType>
class Linearizer {
 public:
  using Scalar = ScalarType;
  using LinearizedFactor = typename Factor<Scalar>::LinearizedFactor;

  /**
   * Construct a Linearizer from factors and optional keys
   *
   * Args:
   *     factors: Only stores a pointer, MUST be in scope for the lifetime of this object!
   *     key_order: If provided, acts as an ordered set of keys that form the state vector
   *                to optimize. Can equal the set of all factor keys or a subset of all
   *                factor keys. If not provided, it is computed from all keys for all
   *                factors using a default ordering.
   */
  Linearizer(const std::vector<Factor<Scalar>>& factors, const std::vector<Key>& key_order = {});

  /**
   * Update linearization at a new evaluation point. Returns the total residual dimension M.
   * This is more efficient than reconstructing this object repeatedly.  On the first call, it will
   * allocate memory and perform analysis needed for efficient repeated relinearization.
   *
   * TODO(aaron): This should be const except that it can initialize the object
   */
  void Relinearize(const Values<Scalar>& values, Linearization<Scalar>* const linearization);

  /**
   * Extract covariances for all optimized variables individually from the full problem
   * covariance.  For each variable, the returned matrix is the corresponding block from the
   * diagonal of the full covariance matrix.  Requires that the Linearizer has already been
   * initialized
   */
  void SplitCovariancesByKey(
      const sym::MatrixX<Scalar>& covariance,
      std::unordered_map<Key, MatrixX<Scalar>>* const covariances_by_key) const;

  /**
   * Whether this contains values, versus having not been evaluated yet
   */
  bool IsInitialized() const;

  /**
   * Basic accessors.
   */
  const std::vector<LinearizedFactor>& LinearizedFactors() const;
  const std::vector<Key>& Keys() const;

 private:
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
                                            const linearization_factor_helper_t& factor_helper,
                                            Linearization<Scalar>* const linearization) const;
  /**
   * Update the combined residual and rhs, along with triplet lists for the sparse matrices, from a
   * single factor
   */
  void UpdateFromLinearizedFactorIntoTripletLists(
      const LinearizedFactor& linearized_factor, const linearization_factor_helper_t& factor_helper,
      sym::VectorX<Scalar>* const residual, sym::VectorX<Scalar>* const rhs,
      std::vector<Eigen::Triplet<Scalar>>* const jacobian_triplets,
      std::vector<Eigen::Triplet<Scalar>>* const hessian_lower_triplets) const;

  /**
   * Check if a Linearization has the correct sizes, and if not, initialize it
   */
  void EnsureLinearizationHasCorrectSize(Linearization<Scalar>* const linearization) const;

  /**
   * Update the combined sparse problem linearization from linearized factors by directly
   * updating the indices of the sparse storage.
   */
  void BuildCombinedProblemSparse(const std::vector<LinearizedFactor>& linearized_factors,
                                  Linearization<Scalar>* const linearization) const;

  /**
   * Create the combined problem linearization from linearized_factors which should have all ones in
   * potentially nonzero locations.  This is done by creating the sparse matrices from triplet
   * lists.
   */
  void BuildCombinedProblemFromOnesFactors(
      const std::vector<LinearizedFactor>& all_ones_linearized_factors,
      Linearization<Scalar>* const linearization);

  bool initialized_{false};

  // Pointer to the nonlinear factors
  const std::vector<Factor<Scalar>>* factors_;

  // Linearized factors - stores individual factor residuals, jacobians, etc
  std::vector<LinearizedFactor> linearized_factors_;

  // Keys that form the state vector
  std::vector<Key> keys_;

  // Index of the keys in the state vector
  std::unordered_map<key_t, index_entry_t> state_index_;

  // Helpers for updating the combined problem from linearized factors
  std::vector<linearization_factor_helper_t> factor_update_helpers_;

  Linearization<Scalar> linearization_ones_;
};

// Free function as an alternate way to call.
template <typename Scalar>
Linearization<Scalar> Linearize(const std::vector<Factor<Scalar>>& factors,
                                const Values<Scalar>& values,
                                const std::vector<Key>& keys_to_optimize = {}) {
  Linearization<Scalar> linearization;
  Linearizer<Scalar>(factors, keys_to_optimize).Relinearize(values, &linearization);
  return linearization;
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
