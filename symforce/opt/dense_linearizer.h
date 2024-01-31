/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <lcmtypes/sym/linearization_offsets_t.hpp>

#include "./factor.h"
#include "./internal/linearized_dense_factor_pool.h"
#include "./linearization.h"

namespace sym {

template <typename Scalar>
class DenseLinearizer {
 public:
  using LinearizedDenseFactor = typename Factor<Scalar>::LinearizedDenseFactor;
  using LinearizationType = DenseLinearization<Scalar>;

  /**
   * Construct a Linearizer from factors and optional keys
   *
   * @param name: name of linearizer used for debug information
   * @param factors: Only stores a pointer, MUST be in scope for the lifetime of this object!
   * @param key_order: If provided, acts as an ordered set of keys that form the state vector to
   *     optimize. Can equal the set of all factor keys or a subset of all factor keys. If not
   *     provided, it is computed from all keys for all factors using a default ordering.
   * @param include_jacobians: Relinearize only allocates and fills out the jacobian if true.
   * @param debug_checks: Whether to perform additional sanity checks for NaNs.  This uses
   *    additional compute but not additional memory except for logging.
   */
  DenseLinearizer(const std::string& name, const std::vector<Factor<Scalar>>& factors,
                  const std::vector<Key>& key_order = {}, bool include_jacobians = false,
                  bool debug_checks = false);

  /**
   * Returns whether Relinearize() has already been called once.
   *
   * Matters because many calculations need to be called only on the first linearization that are
   * then cached for subsequent use. Also, if Relinearize() has already been called, then the
   * matrices in the linearization are expected to already be allocated to the right size.
   */
  bool IsInitialized() const;

  /**
   * The keys to optimize, in the order they appear in the state vector (i.e., in rhs).
   */
  const std::vector<Key>& Keys() const;

  /**
   * A map from all optimized keys in the problem to an index entry for the corresponding optimized
   * values (where the offset is into the problem state vector, i.e., the rhs of a linearization).
   *
   * Only read if IsInitialized() returns true (i.e., after the first call to Relinearize).
   */
  const std::unordered_map<key_t, index_entry_t>& StateIndex() const;

  /**
   * Update linearization at a new evaluation point.
   * This is more efficient than reconstructing this object repeatedly. On the first call, it will
   * allocate memory and perform analysis needed for efficient repeated relinearization.
   *
   * On the first call to Relinearize, the matrices in linearization will be allocated and sized
   * correctly. On subsequent calls, the matrices of linearization are expected to already be
   * allocated and sized correctly.
   *
   * TODO(aaron): This should be const except that it can initialize the object
   */
  void Relinearize(const Values<Scalar>& values, DenseLinearization<Scalar>& linearization);

 private:
  // The name of this linearizer to be used for printing debug information.
  std::string name_;
  const std::vector<Factor<Scalar>>* factors_;
  std::vector<Key> keys_;
  std::unordered_map<key_t, index_entry_t> state_index_;
  internal::LinearizedDenseFactorPool<Scalar> linearized_dense_factors_;
  bool is_initialized_;
  bool include_jacobians_;
  bool debug_checks_;

  // The index for each factor in the values. Cached the first time we linearize, to avoid repeated
  // unordered_map lookups
  std::vector<std::vector<index_entry_t>> factor_indices_;

  // One std::vector<linearization_offsets_t> per factor in factors_ (same order).
  // nth linearization_offsets_t of a vector is the offsets into factor and state vectors of the
  // nth key to optimize (NOT linearized key, a factor key may be linearized but not optimized) of
  // the corresponding factor.
  std::vector<std::vector<linearization_offsets_t>> factor_keyoffsets_;

  /**
   * Evaluates the linearizations of the factors at values into linearization, caching all values
   * needed for relinearization along the way.
   *
   * Specifically:
   *  - Calculates state_index_
   *  - Allocates all factor linearizations in linearized_dense_factors_
   *  - Calculates factor_indices_
   *  - Calculates factor_keyoffsets_
   *
   * Sets is_initialized_ to true.
   *
   * NOTE(brad): Things might break if you call it twice.
   */
  void InitialLinearization(const Values<Scalar>& values,
                            DenseLinearization<Scalar>& linearization);
};

}  // namespace sym

extern template class sym::DenseLinearizer<double>;
extern template class sym::DenseLinearizer<float>;
