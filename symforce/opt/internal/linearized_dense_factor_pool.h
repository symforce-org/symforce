/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <vector>

#include "../factor.h"
#include "./linearizer_utils.h"

namespace sym {

namespace internal {

/**
 * Class to abstract out the method of getting a temporary LinearizedDenseFactor in which
 * to evaluate factor's linearizations.
 *
 * Exists because adding these features directly to the Linearizer class makes that harder
 * to read and to allow the functionality to be reused by the dense linearizer.
 *
 * At a more abstract level, exists because we can't pass the same MatrixXd's to the different
 * factors because then Eigen will implicitly reallocate the memory of each MatrixXd every
 * time the vector size changes. So we pass them MatrixXds who already have the right size.
 *
 * This class could be deleted if we could pass a factor a block of memory that can be made
 * larger if needed by not performing any new allocation if not.
 *
 * NOTE(brad): We do not have a corresponding class for the sparse linearized factors because
 * there we currently have a separate linearized factor per factor. This is to avoid the
 * need to re-write the column pointers and row indices during each factor call.
 *
 * Usage (during creation):
 *   LinearizedDenseFactorPool<Scalar> linearized_dense_factors;
 *   ...
 *   linearized_dense_factors.reserve(number_of_dense_factors);
 *   // Don't touch size_tracker other than to pass it to AppendFactorSize
 *   // Can be disposed of after last call to AppendFactorSize
 *   // Same SizeTracker must be used for every call to AppendFactorSize
 *   typename LinearizedDenseFactorPool<Scalar>::SizeTracker size_tracker;
 *
 *   for (Factor& factor : factors) {
 *     int res_dim = ...;
 *     int rhs_dim = ...;
 *
 *     linearized_dense_factors.AppendFactorSize(res_dim, rhs_dim, size_tracker);
 *   }
 */
template <typename Scalar>
class LinearizedDenseFactorPool {
 public:
  using SizeTracker = std::unordered_map<std::pair<int, int>, int, internal::StdPairHash>;
  using LinearizedDenseFactor = typename Factor<Scalar>::LinearizedDenseFactor;

  /**
   * Reserve space for the number of dense factor to be appended with AppendFactorSize.
   */
  void reserve(const size_t dense_factor_count) {
    index_per_factor_.reserve(dense_factor_count);
  }

  /**
   * Call this method for each factor in the same order you wish to index into
   * PreallocatedLinearizedDenseFactor with.
   *
   * Preconditions:
   *   - dims_to_index was default initialized
   *   - The same SizeTracker is used for every call to AppendFactorSize
   *   - dims_to_index has only been mutated by AppendFactorSize with this instance of
   *     LinearizedDenseFactorPool
   *
   * If not already present, adds a linearization with residual and rhs dimensions provided
   * to unique_linearizations.
   * Appends to factor_indices the index of the linearization in unique_linearizations with
   * dimensions res_dim and rhs_dim.
   */
  void AppendFactorSize(const int res_dim, const int rhs_dim, SizeTracker& dims_to_index) {
    // Invariant: dims_to_index contains a key (dim1, dim2) iff AppendFactorSize was called with
    // those dims. dims_to_index[(dim1, dim2)] = n  ==> AppendFactorSize was first called with
    // those dims on its nth call.
    const auto index_at_and_was_inserted =
        dims_to_index.emplace(std::make_pair(res_dim, rhs_dim), unique_linearized_factors_.size());

    index_per_factor_.push_back(index_at_and_was_inserted.first->second);
    if (index_at_and_was_inserted.second) {
      unique_linearized_factors_.emplace_back();
      LinearizedDenseFactor& new_linearization = unique_linearized_factors_.back();

      new_linearization.residual.resize(res_dim, 1);
      new_linearization.jacobian.resize(res_dim, rhs_dim);
      new_linearization.hessian.resize(rhs_dim, rhs_dim);
      new_linearization.rhs.resize(rhs_dim, 1);
    }
  }

  /**
   * Returns a linearized dense factor whose size is correct for the dense_index'th dense
   * factor.
   */
  LinearizedDenseFactor& at(const int dense_index) {
    return unique_linearized_factors_.at(index_per_factor_.at(dense_index));
  }

 private:
  // One LinearizedDenseFactor for each unique pair of res_dim and rhs_dim seen
  std::vector<LinearizedDenseFactor> unique_linearized_factors_;

  // Indices into unique_linearized_factors_; one per dense factor;
  // unique_linearized_factors_[index_per_factor_[n]] has the same res_dim and rhs_dim
  // as the nth call to AppendFactorSize
  std::vector<int> index_per_factor_;
};

}  // namespace internal

}  // namespace sym
