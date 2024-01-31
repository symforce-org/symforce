/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/SparseCore>

#include <lcmtypes/sym/linearization_dense_factor_helper_t.hpp>
#include <lcmtypes/sym/linearization_sparse_factor_helper_t.hpp>

#include "./factor.h"
#include "./internal/linearized_dense_factor_pool.h"
#include "./linearization.h"
#include "./values.h"

namespace sym {

/**
 * Class for evaluating multiple Factors at the linearization point given by a Values.
 *
 * Stores the original Factors as well as the LinearizedFactors, and provides tools for
 * aggregating keys and building a large jacobian / hessian for optimization.
 *
 * For efficiency, prefer calling Relinearize() instead of re-constructing this object!
 */
template <typename ScalarType>
class Linearizer {
 public:
  using Scalar = ScalarType;
  using LinearizedDenseFactor = typename Factor<Scalar>::LinearizedDenseFactor;
  using LinearizedSparseFactor = typename Factor<Scalar>::LinearizedSparseFactor;
  using LinearizationType = SparseLinearization<Scalar>;

  /**
   * Construct a Linearizer from factors and optional keys
   *
   * @param factors: Only stores a pointer, MUST be in scope for the lifetime of this object!
   * @param key_order: If provided, acts as an ordered set of keys that form the state vector
   *    to optimize. Can equal the set of all factor keys or a subset of all factor keys. If not
   *    provided, it is computed from all keys for all factors using a default ordering.
   * @param debug_checks: Whether to perform additional sanity checks for NaNs.  This uses
   *    additional compute but not additional memory except for logging.
   */
  Linearizer(const std::string& name, const std::vector<Factor<Scalar>>& factors,
             const std::vector<Key>& key_order = {}, bool include_jacobians = false,
             bool debug_checks = false);

  /**
   * Update linearization at a new evaluation point
   *
   * This is more efficient than reconstructing this object repeatedly. On the first call, it will
   * allocate memory and perform analysis needed for efficient repeated relinearization.
   *
   * TODO(aaron): This should be const except that it can initialize the object
   */
  void Relinearize(const Values<Scalar>& values, SparseLinearization<Scalar>& linearization);

  /**
   * Whether this contains values, versus having not been evaluated yet
   */
  bool IsInitialized() const;

  // ----------------------------------------------------------------------------
  // Basic accessors
  // ----------------------------------------------------------------------------

  const std::vector<LinearizedSparseFactor>& LinearizedSparseFactors() const;

  const std::vector<Key>& Keys() const;

  // NOTE(brad): Offset of index entries is sum of all tangent_dims of all previous index entries
  // (order of index entries determined by order of corresponding keys in Keys()). Contains entry
  // for each key in Keys().
  const std::unordered_map<key_t, index_entry_t>& StateIndex() const;

 private:
  /**
   * Allocate all factor storage and compute sparsity pattern. This does a lot of index
   * computation on the first linearization, such that repeated linearization can be fast.
   */
  void BuildInitialLinearization(const Values<Scalar>& values);

  /**
   * Update the sparse combined problem linearization from a single factor.
   */
  void UpdateFromLinearizedDenseFactorIntoSparse(
      const LinearizedDenseFactor& linearized_factor,
      const linearization_dense_factor_helper_t& factor_helper,
      SparseLinearization<Scalar>& linearization) const;
  void UpdateFromLinearizedSparseFactorIntoSparse(
      const LinearizedSparseFactor& linearized_factor,
      const linearization_sparse_factor_helper_t& factor_helper,
      SparseLinearization<Scalar>& linearization) const;

  /**
   * Update the combined residual and rhs, along with triplet lists for the sparse matrices, from a
   * single factor
   */
  void UpdateFromDenseFactorIntoTripletLists(
      const LinearizedDenseFactor& linearized_factor,
      const linearization_dense_factor_helper_t& factor_helper,
      std::vector<Eigen::Triplet<Scalar>>& jacobian_triplets,
      std::vector<Eigen::Triplet<Scalar>>& hessian_lower_triplets) const;
  void UpdateFromSparseFactorIntoTripletLists(
      const LinearizedSparseFactor& linearized_factor,
      const linearization_sparse_factor_helper_t& factor_helper,
      std::vector<Eigen::Triplet<Scalar>>& jacobian_triplets,
      std::vector<Eigen::Triplet<Scalar>>& hessian_lower_triplets) const;

  /**
   * Check if a Linearization has the correct sizes, and if not, initialize it
   */
  void EnsureLinearizationHasCorrectSize(SparseLinearization<Scalar>& linearization) const;

  bool initialized_{false};

  // The name of this linearizer to be used for printing debug information.
  std::string name_;

  // Pointer to the nonlinear factors
  const std::vector<Factor<Scalar>>* factors_;

  // The index for each factor in the values.  Cached the first time we linearize, to avoid repeated
  // unordered_map lookups
  std::vector<std::vector<index_entry_t>> factor_indices_;

  bool include_jacobians_;

  bool debug_checks_;

  // Linearized factors - stores individual factor residuals, jacobians, etc
  internal::LinearizedDenseFactorPool<Scalar> linearized_dense_factors_;  // one per Jacobian shape
  std::vector<LinearizedSparseFactor> linearized_sparse_factors_;         // one per sparse factor

  // Keys that form the state vector
  std::vector<Key> keys_;

  // Index of the keys in the state vector
  std::unordered_map<key_t, index_entry_t> state_index_;

  // Helpers for updating the combined problem from linearized factors
  std::vector<linearization_dense_factor_helper_t> dense_factor_update_helpers_;
  std::vector<linearization_sparse_factor_helper_t> sparse_factor_update_helpers_;

  // Numerical linearization from the very first linearization that is used to initialize new
  // LevenbergMarquardtState::StateBlocks (at most 3 times) and isn't touched on each subsequent
  // relinearization.
  SparseLinearization<Scalar> init_linearization_;
};

/**
 * Free function as an alternate way to call the Linearizer
 */
template <typename Scalar>
SparseLinearization<Scalar> Linearize(const std::vector<Factor<Scalar>>& factors,
                                      const Values<Scalar>& values,
                                      const std::vector<Key>& keys_to_optimize = {},
                                      const std::string& linearizer_name = "Linearizer") {
  SparseLinearization<Scalar> linearization;
  Linearizer<Scalar>(linearizer_name, factors, keys_to_optimize).Relinearize(values, linearization);
  return linearization;
}

}  // namespace sym

// Explicit instantiation declaration
extern template class sym::Linearizer<double>;
extern template class sym::Linearizer<float>;
