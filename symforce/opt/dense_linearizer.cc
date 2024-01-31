/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./dense_linearizer.h"

#include <tuple>

#include "./internal/linearizer_utils.h"

namespace sym {

template <typename Scalar>
DenseLinearizer<Scalar>::DenseLinearizer(const std::string& name,
                                         const std::vector<Factor<Scalar>>& factors,
                                         const std::vector<Key>& key_order,
                                         const bool include_jacobians, const bool debug_checks)
    : name_(name),
      factors_{&factors},
      state_index_{},
      is_initialized_{false},
      include_jacobians_{include_jacobians},
      debug_checks_{debug_checks} {
  if (key_order.empty()) {
    keys_ = ComputeKeysToOptimize(factors);
  } else {
    keys_ = key_order;
  }
}

template <typename ScalarType>
bool DenseLinearizer<ScalarType>::IsInitialized() const {
  return is_initialized_;
}

template <typename ScalarType>
const std::vector<Key>& DenseLinearizer<ScalarType>::Keys() const {
  return keys_;
}

template <typename ScalarType>
const std::unordered_map<key_t, index_entry_t>& DenseLinearizer<ScalarType>::StateIndex() const {
  SYM_ASSERT(IsInitialized());
  return state_index_;
}

template <typename Scalar>
using LinearizedDenseFactor = typename DenseLinearizer<Scalar>::LinearizedDenseFactor;

template <typename Scalar, typename Derived>
void CopyJacobianFactorToCombined(const Eigen::DenseBase<Derived>& jacobian,
                                  const std::vector<linearization_offsets_t>& key_offsets,
                                  const int residual_offset,
                                  DenseLinearization<Scalar>& linearization) {
  const int residual_dim = jacobian.rows();
  for (const linearization_offsets_t& offsets : key_offsets) {
    linearization.jacobian.block(residual_offset, offsets.combined_offset, residual_dim,
                                 offsets.tangent_dim) =
        jacobian.block(0, offsets.factor_offset, residual_dim, offsets.tangent_dim);
  }
}

template <typename Scalar>
void CopyRhsFactorToCombined(const LinearizedDenseFactor<Scalar>& linearized_dense_factor,
                             const std::vector<linearization_offsets_t>& key_offsets,
                             DenseLinearization<Scalar>& linearization) {
  for (const linearization_offsets_t& offsets : key_offsets) {
    linearization.rhs.segment(offsets.combined_offset, offsets.tangent_dim) +=
        linearized_dense_factor.rhs.segment(offsets.factor_offset, offsets.tangent_dim);
  }
}

template <typename Scalar>
void CopyHessianFactorToCombined(const LinearizedDenseFactor<Scalar>& factor_linearization,
                                 const std::vector<linearization_offsets_t>& key_offsets,
                                 DenseLinearization<Scalar>& linearization) {
  // NOTE(brad): Assumes no repeats in key_offsets (and thus no repeats in a factor's
  // OptimizedKeys()). If there is a repeat, then the strict upper triangle of a diagonal
  // block will be copied. Not the biggest deal, but it's a waste.

  // Loop over key blocks of lower triangle of factor linearization hessian
  for (int col_index = 0; col_index < static_cast<int>(key_offsets.size()); col_index++) {
    const linearization_offsets_t& col_block = key_offsets[col_index];

    // Diagonal block is triangular
    linearization.hessian_lower
        .block(col_block.combined_offset, col_block.combined_offset, col_block.tangent_dim,
               col_block.tangent_dim)
        .template triangularView<Eigen::Lower>() +=
        factor_linearization.hessian.block(col_block.factor_offset, col_block.factor_offset,
                                           col_block.tangent_dim, col_block.tangent_dim);

    // strict lower triangle
    for (int row_index = col_index + 1; row_index < static_cast<int>(key_offsets.size());
         row_index++) {
      const linearization_offsets_t& row_block = key_offsets[row_index];

      // If keys have reverse order in combined hessian, the combined block will be in the
      // upper triangle, in which case we must transpose to get the lower triangle version.
      if (row_block.combined_offset < col_block.combined_offset) {
        linearization.hessian_lower.block(col_block.combined_offset, row_block.combined_offset,
                                          col_block.tangent_dim, row_block.tangent_dim) +=
            factor_linearization.hessian
                .block(row_block.factor_offset, col_block.factor_offset, row_block.tangent_dim,
                       col_block.tangent_dim)
                .transpose();
      } else {
        linearization.hessian_lower.block(row_block.combined_offset, col_block.combined_offset,
                                          row_block.tangent_dim, col_block.tangent_dim) +=
            factor_linearization.hessian.block(row_block.factor_offset, col_block.factor_offset,
                                               row_block.tangent_dim, col_block.tangent_dim);
      }
    }
  }
}

/**
 * Linearizes this->factors_ at values into linearization. Is different than subsequent runs
 * because it additionally:
 * - Allocates space for the dense linearized factors
 * - Figures out which parts of the problem linearization correspond to which parts of each
 *   factor linearization
 * - Allocates memory for each factor linearization of each shape
 * - Performs sanity checks that the outputs of each factor are consistent
 * - Computes indices into values for the arguments of each factor
 */
template <typename Scalar>
void DenseLinearizer<Scalar>::InitialLinearization(const Values<Scalar>& values,
                                                   DenseLinearization<Scalar>& linearization) {
  // Compute state vector index
  int32_t offset = 0;
  for (const Key& key : keys_) {
    auto entry = values.IndexEntryAt(key);
    entry.offset = offset;
    state_index_[key.GetLcmType()] = entry;

    offset += entry.tangent_dim;
  }

  // N is sum of tangent dim of all keys to optimize. Equal to num cols of jacobian
  const int32_t N = offset;

  // Allocate final storage for the combined RHS since it is dense and has a known size before
  // linearizing factors. Allocate temporary storage for the residual because the combined residual
  // dimension is not known yet.
  linearization.rhs.resize(N);
  linearization.rhs.setZero();
  linearization.hessian_lower.resize(N, N);
  linearization.hessian_lower.template triangularView<Eigen::Lower>().setZero();

  // NOTE(brad): Currently we assume all factors are dense factors. The reason for this is that it
  // is simpler to only copy dense factors into a dense linearization, and want to get that version
  // working first before worrying about sparse factors.
  linearized_dense_factors_.reserve(factors_->size());
  typename internal::LinearizedDenseFactorPool<Scalar>::SizeTracker size_tracker;

  LinearizedDenseFactor factor_linearization{};

  std::vector<Scalar> combined_residual;
  std::vector<decltype(factor_linearization.jacobian)> factor_jacobians;

  // Track these to make sure that all combined keys are touched by at least one factor.
  std::unordered_set<Key> keys_touched_by_factors;

  // Inside this loop
  for (const auto& factor : *factors_) {
    for (const auto& key : factor.OptimizedKeys()) {
      keys_touched_by_factors.insert(key);
    }

    // First, we evaluate into a temporary linearized factor
    factor_indices_.push_back(values.CreateIndex(factor.AllKeys()).entries);
    factor.Linearize(values, factor_linearization, &factor_indices_.back());
    if (debug_checks_) {
      internal::CheckLinearizedFactor(name_, factor, values, factor_linearization,
                                      factor_indices_.back());
    }

    factor_keyoffsets_.emplace_back();
    const int32_t tangent_dim = [&] {
      int32_t factor_tangent_dim;
      std::tie(factor_keyoffsets_.back(), factor_tangent_dim) =
          internal::FactorOffsets<linearization_offsets_t>(values, factor.OptimizedKeys(),
                                                           state_index_);
      return factor_tangent_dim;
    }();
    const std::vector<linearization_offsets_t>& key_offsets = factor_keyoffsets_.back();

    if (key_offsets.empty()) {
      std::vector<key_t> input_keys;
      for (const Key& key : factor.OptimizedKeys()) {
        input_keys.push_back(key.GetLcmType());
      }

      spdlog::warn(
          "LM<{}>: Optimizing a factor that touches no optimized keys! Optimized input keys for "
          "the factor are: {}",
          name_, input_keys);
    }

    internal::AssertConsistentShapes(tangent_dim, factor_linearization, include_jacobians_);

    // Make sure a temporary of the right dimension is kept for relinearizations
    linearized_dense_factors_.AppendFactorSize(factor_linearization.residual.rows(),
                                               factor_linearization.rhs.rows(), size_tracker);

    // Add contributions of residual and jacobian to temporary vectors because we don't yet know
    // their total size.
    combined_residual.insert(
        combined_residual.cend(), factor_linearization.residual.data(),
        factor_linearization.residual.data() + factor_linearization.residual.size());
    if (include_jacobians_) {
      factor_jacobians.emplace_back(std::move(factor_linearization.jacobian));
    }

    // Add contribution of rhs and hessian (can be done directly because we know the whole size)
    CopyRhsFactorToCombined(factor_linearization, key_offsets, linearization /* mut */);
    CopyHessianFactorToCombined(factor_linearization, key_offsets, linearization /* mut */);
  }
  // Now that we know the full problem size, we can construct the residual and jacobian
  linearization.residual =
      Eigen::Map<VectorX<Scalar>>(combined_residual.data(), combined_residual.size());
  if (include_jacobians_) {
    linearization.jacobian.resize(combined_residual.size(), N);
    linearization.jacobian.setZero();

    int row_offset = 0;
    for (int i = 0; i < static_cast<int>(factor_jacobians.size()); i++) {
      const auto& jacobian = factor_jacobians[i];
      CopyJacobianFactorToCombined(jacobian, factor_keyoffsets_[i], row_offset, linearization);
      row_offset += jacobian.rows();
    }
  }

  if (keys_.size() != keys_touched_by_factors.size()) {
    for (const auto& key : keys_) {
      if (keys_touched_by_factors.count(key) == 0) {
        throw std::runtime_error(
            fmt::format("Key {} is in the state vector but is not optimized by any factor.", key));
      }
    }
  }

  linearization.SetInitialized();
}

template <typename ScalarType>
void DenseLinearizer<ScalarType>::Relinearize(const Values<ScalarType>& values,
                                              DenseLinearization<ScalarType>& linearization) {
  if (is_initialized_) {
    // Set rhs & hessian_lower to 0 as they will be built additively
    linearization.rhs.setZero();
    linearization.hessian_lower.template triangularView<Eigen::Lower>().setZero();
    // The parts of linearization.jacobian that aren't being set are assumed to have already
    // been set to 0 by InitialLinearization and to have not been mutated since.

    int residual_offset = 0;
    for (int i = 0; i < static_cast<int>(factors_->size()); i++) {
      const auto& factor = (*factors_)[i];
      auto& linearized_dense_factor = linearized_dense_factors_.at(i);
      factor.Linearize(values, linearized_dense_factor, &factor_indices_[i]);
      if (debug_checks_) {
        internal::CheckLinearizedFactor(name_, factor, values, linearized_dense_factor,
                                        factor_indices_.back());
      }

      const LinearizedDenseFactor& factor_linearization = linearized_dense_factors_.at(i);
      const std::vector<linearization_offsets_t>& key_offsets = factor_keyoffsets_[i];
      const int residual_dim = factor_linearization.residual.size();

      // Copy factor_linearization values into linearization
      linearization.residual.segment(residual_offset, residual_dim) = factor_linearization.residual;
      if (include_jacobians_) {
        CopyJacobianFactorToCombined(factor_linearization.jacobian, key_offsets, residual_offset,
                                     linearization /* mut */);
      }
      CopyRhsFactorToCombined(factor_linearization, key_offsets, linearization /* mut */);
      CopyHessianFactorToCombined(factor_linearization, key_offsets, linearization /* mut */);

      residual_offset += residual_dim;
    }

  } else {
    InitialLinearization(values, linearization /* mut */);
  }
}

}  // namespace sym

template class sym::DenseLinearizer<double>;
template class sym::DenseLinearizer<float>;
