/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./linearizer.h"

#include "./assert.h"
#include "./internal/linearizer_utils.h"
#include "./optional.h"
#include "./tic_toc.h"
#include "symforce/opt/factor.h"

namespace sym {

// ----------------------------------------------------------------------------
// Public Methods
// ----------------------------------------------------------------------------

template <typename ScalarType>
Linearizer<ScalarType>::Linearizer(const std::string& name,
                                   const std::vector<Factor<Scalar>>& factors,
                                   const std::vector<Key>& key_order, const bool include_jacobians)
    : name_(name),
      factors_(&factors),
      include_jacobians_(include_jacobians),
      linearized_dense_factors_(),
      linearized_sparse_factors_() {
  if (key_order.empty()) {
    keys_ = ComputeKeysToOptimize(factors);
  } else {
    keys_ = key_order;
  }

  size_t num_sparse_factors = 0;
  size_t num_dense_factors = 0;
  for (const auto& factor : *factors_) {
    if (factor.IsSparse()) {
      num_sparse_factors++;
    } else {
      num_dense_factors++;
    }
  }

  linearized_sparse_factors_.resize(num_sparse_factors);
  sparse_factor_update_helpers_.reserve(num_sparse_factors);

  dense_factor_update_helpers_.reserve(num_dense_factors);
}

template <typename ScalarType>
void Linearizer<ScalarType>::Relinearize(const Values<Scalar>& values,
                                         SparseLinearization<Scalar>& linearization) {
  if (IsInitialized()) {
    SYM_TIME_SCOPE("Linearizer<{}>::Relinearize::NonFirst()", name_);

    EnsureLinearizationHasCorrectSize(linearization);

    // Zero out blocks that are built additively
    linearization.rhs.setZero();
    Eigen::Map<VectorX<Scalar>>(linearization.hessian_lower.valuePtr(),
                                linearization.hessian_lower.nonZeros())
        .setZero();

    // Evaluate the factors
    size_t sparse_idx{0};
    size_t dense_idx{0};
    for (int i = 0; i < static_cast<int>(factors_->size()); i++) {
      const auto& factor = (*factors_)[i];

      if (factor.IsSparse()) {
        auto& linearized_sparse_factor = linearized_sparse_factors_.at(sparse_idx);
        // TODO: Only compute factor Jacobians when include_jacobians_ is true.
        factor.Linearize(values, linearized_sparse_factor, &factor_indices_[i]);

        UpdateFromLinearizedSparseFactorIntoSparse(
            linearized_sparse_factor, sparse_factor_update_helpers_.at(sparse_idx), linearization);

        ++sparse_idx;
      } else {
        // Use temporary with the right size to avoid allocating after initialization.
        auto& linearized_dense_factor = linearized_dense_factors_.at(dense_idx);
        // TODO: Only compute factor Jacobians when include_jacobians_ is true.
        factor.Linearize(values, linearized_dense_factor, &factor_indices_[i]);

        UpdateFromLinearizedDenseFactorIntoSparse(
            linearized_dense_factor, dense_factor_update_helpers_.at(dense_idx), linearization);

        ++dense_idx;
      }
    }

    linearization.SetInitialized();
  } else {
    SYM_TIME_SCOPE("Linearizer<{}>::Relinearize::First()", name_);

    BuildInitialLinearization(values);

    linearization = init_linearization_;
  }
}

template <typename ScalarType>
bool Linearizer<ScalarType>::IsInitialized() const {
  return initialized_;
}

template <typename ScalarType>
const std::vector<typename Factor<ScalarType>::LinearizedSparseFactor>&
Linearizer<ScalarType>::LinearizedSparseFactors() const {
  return linearized_sparse_factors_;
}

template <typename ScalarType>
const std::vector<Key>& Linearizer<ScalarType>::Keys() const {
  return keys_;
}

template <typename ScalarType>
const std::unordered_map<key_t, index_entry_t>& Linearizer<ScalarType>::StateIndex() const {
  SYM_ASSERT(IsInitialized());
  return state_index_;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

template <typename ScalarType>
void Linearizer<ScalarType>::BuildInitialLinearization(const Values<Scalar>& values) {
  // Compute state vector index
  int32_t offset = 0;
  for (const Key& key : keys_) {
    auto entry = values.IndexEntryAt(key);
    entry.offset = offset;
    state_index_[key.GetLcmType()] = entry;

    offset += entry.tangent_dim;
  }

  const int32_t N = offset;

  // Allocate final storage for the combined RHS since it is dense and has a known size before
  // linearizing factors. Allocate temporary storage for the residual because the combined residual
  // dimension is not known yet.
  init_linearization_.rhs.resize(N);
  init_linearization_.rhs.setZero();

  std::vector<Scalar> residual;

  std::vector<Eigen::Triplet<Scalar>> jacobian_triplets;
  std::vector<Eigen::Triplet<Scalar>> hessian_lower_triplets;

  int32_t combined_residual_offset = 0;

  // Track these to make sure that all combined keys are touched by at least one factor.
  std::unordered_set<Key> keys_touched_by_factors;

  linearized_dense_factors_.reserve(factors_->size() - linearized_sparse_factors_.size());
  typename internal::LinearizedDenseFactorPool<Scalar>::SizeTracker dense_factor_size_tracker;

  // Evaluate all factors, processing the dense ones in place and storing the sparse ones for
  // later
  LinearizedDenseFactor linearized_dense_factor{};
  size_t sparse_idx{0};
  factor_indices_.reserve(factors_->size());
  for (const auto& factor : *factors_) {
    factor_indices_.push_back(values.CreateIndex(factor.AllKeys()).entries);

    for (const auto& key : factor.OptimizedKeys()) {
      keys_touched_by_factors.insert(key);
    }

    if (factor.IsSparse()) {
      LinearizedSparseFactor& linearized_factor = linearized_sparse_factors_.at(sparse_idx);
      ++sparse_idx;
      factor.Linearize(values, linearized_factor, &factor_indices_.back());

      auto helper_and_dimension =
          internal::ComputeFactorHelper<linearization_sparse_factor_helper_t>(
              linearized_factor, values, factor.OptimizedKeys(), state_index_, name_,
              combined_residual_offset);
      internal::AssertConsistentShapes(helper_and_dimension.second, linearized_factor,
                                       include_jacobians_);
      sparse_factor_update_helpers_.push_back(std::move(helper_and_dimension.first));
      const auto& factor_helper = sparse_factor_update_helpers_.back();

      UpdateFromSparseFactorIntoTripletLists(linearized_factor, factor_helper, jacobian_triplets,
                                             hessian_lower_triplets);

      // Fill in the combined residual slice
      for (int res_i = 0; res_i < factor_helper.residual_dim; ++res_i) {
        residual.push_back(linearized_factor.residual(res_i));
      }

      // Add contribution from right-hand-side
      for (const linearization_offsets_t& key_helper : factor_helper.key_helpers) {
        init_linearization_.rhs.segment(key_helper.combined_offset, key_helper.tangent_dim) +=
            linearized_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);
      }
    } else {
      factor.Linearize(values, linearized_dense_factor, &factor_indices_.back());

      // Make sure a temporary of the right dimension is kept for relinearizations
      linearized_dense_factors_.AppendFactorSize(linearized_dense_factor.residual.rows(),
                                                 linearized_dense_factor.rhs.rows(),
                                                 dense_factor_size_tracker);

      // Create dense factor helper
      auto helper_and_dimension =
          internal::ComputeFactorHelper<linearization_dense_factor_helper_t>(
              linearized_dense_factor, values, factor.OptimizedKeys(), state_index_, name_,
              combined_residual_offset);
      internal::AssertConsistentShapes(helper_and_dimension.second, linearized_dense_factor,
                                       include_jacobians_);
      dense_factor_update_helpers_.push_back(std::move(helper_and_dimension.first));

      const auto& factor_helper = dense_factor_update_helpers_.back();

      // Create dense factor triplets
      UpdateFromDenseFactorIntoTripletLists(linearized_dense_factor, factor_helper,
                                            jacobian_triplets, hessian_lower_triplets);

      // Fill in the combined residual slice
      for (int i = 0; i < factor_helper.residual_dim; i++) {
        residual.push_back(linearized_dense_factor.residual(i));
      }

      // Add contributions from right-hand-side
      for (const linearization_dense_key_helper_t& key_helper : factor_helper.key_helpers) {
        init_linearization_.rhs.segment(key_helper.combined_offset, key_helper.tangent_dim) +=
            linearized_dense_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);
      }
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

  SYM_ASSERT(static_cast<int32_t>(residual.size()) == combined_residual_offset);

  // Allocate storage of the rest of the combined linearization
  const int32_t M = combined_residual_offset;
  init_linearization_.residual.resize(M);
  if (include_jacobians_) {
    init_linearization_.jacobian.resize(M, N);
  }
  init_linearization_.hessian_lower.resize(N, N);

  // Create the matrices
  for (int i = 0; i < M; i++) {
    init_linearization_.residual(i) = residual[i];
  }

  if (include_jacobians_) {
    init_linearization_.jacobian.setFromTriplets(jacobian_triplets.begin(),
                                                 jacobian_triplets.end());
    SYM_ASSERT(init_linearization_.jacobian.isCompressed());
  }

  init_linearization_.hessian_lower.setFromTriplets(hessian_lower_triplets.begin(),
                                                    hessian_lower_triplets.end());
  SYM_ASSERT(init_linearization_.hessian_lower.isCompressed());

  init_linearization_.SetInitialized();

  // Create a hash map from the sparse nonzero indices of the jacobian/hessian to their storage
  // offset within the sparse array.
  optional<internal::CoordsToStorageMap> jacobian_row_col_to_storage_offset{};
  if (include_jacobians_) {
    jacobian_row_col_to_storage_offset.emplace(
        internal::CoordsToStorageOffset(init_linearization_.jacobian));
  }
  const auto hessian_row_col_to_storage_offset =
      internal::CoordsToStorageOffset(init_linearization_.hessian_lower);

  // Use the hash map to mark sparse storage offsets for every row of each key block of each
  // factor
  for (auto& dense_factor_update_helper : dense_factor_update_helpers_) {
    internal::ComputeKeyHelperSparseColOffsets<Scalar>(jacobian_row_col_to_storage_offset,
                                                       hessian_row_col_to_storage_offset,
                                                       dense_factor_update_helper);
  }
  for (int i = 0; i < static_cast<int>(linearized_sparse_factors_.size()); ++i) {
    const LinearizedSparseFactor& linearized_factor = linearized_sparse_factors_.at(i);
    linearization_sparse_factor_helper_t& factor_helper = sparse_factor_update_helpers_[i];
    internal::ComputeKeyHelperSparseMap<Scalar>(linearized_factor,
                                                jacobian_row_col_to_storage_offset,
                                                hessian_row_col_to_storage_offset, factor_helper);
  }

  initialized_ = true;
}

template <typename ScalarType>
void Linearizer<ScalarType>::UpdateFromLinearizedDenseFactorIntoSparse(
    const LinearizedDenseFactor& linearized_factor,
    const linearization_dense_factor_helper_t& factor_helper,
    SparseLinearization<Scalar>& linearization) const {
  // The residual dimension must be the same, even for factors that return VectorX.  If the residual
  // size changes, the optimizer must be re-created.
  SYM_ASSERT(factor_helper.residual_dim == linearized_factor.residual.size());

  // Fill in the combined residual slice
  linearization.residual.segment(factor_helper.combined_residual_offset,
                                 factor_helper.residual_dim) = linearized_factor.residual;

  // For each key
  for (int key_i = 0; key_i < static_cast<int>(factor_helper.key_helpers.size()); ++key_i) {
    const linearization_dense_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    if (include_jacobians_) {
      // Fill in jacobian block, column by column
      for (int col_block = 0; col_block < key_helper.tangent_dim; ++col_block) {
        Eigen::Map<VectorX<Scalar>>(
            linearization.jacobian.valuePtr() + key_helper.jacobian_storage_col_starts[col_block],
            factor_helper.residual_dim) =
            linearized_factor.jacobian.block(0, key_helper.factor_offset + col_block,
                                             factor_helper.residual_dim, 1);
      }
    }

    // Add contribution from right-hand-side
    linearization.rhs.segment(key_helper.combined_offset, key_helper.tangent_dim) +=
        linearized_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);

    // Add contribution from diagonal hessian block, column by column
    auto col_start_iter = key_helper.hessian_storage_col_starts.begin();
    for (int col_block = 0; col_block < key_helper.tangent_dim; ++col_block) {
      const auto col_start = *col_start_iter;
      col_start_iter++;
      Eigen::Map<VectorX<Scalar>>(linearization.hessian_lower.valuePtr() + col_start,
                                  key_helper.tangent_dim - col_block) +=
          linearized_factor.hessian.block(key_helper.factor_offset + col_block,
                                          key_helper.factor_offset + col_block,
                                          key_helper.tangent_dim - col_block, 1);
    }

    // Add contributions from off-diagonal hessian blocks, column by column
    // Here key_i represents the block row and key_j represents the block column into the hessian.
    // Remember we're filling in the lower triangle, and because we're column major every column
    // of the block is contiguous in sparse storage.
    for (int key_j = 0; key_j < key_i; key_j++) {
      const linearization_dense_key_helper_t& key_helper_j = factor_helper.key_helpers[key_j];

      if (key_helper_j.combined_offset < key_helper.combined_offset) {
        for (int32_t col_j = 0; col_j < static_cast<int32_t>(key_helper_j.tangent_dim); ++col_j) {
          const auto col_start = *col_start_iter;
          col_start_iter++;
          Eigen::Map<VectorX<Scalar>>(linearization.hessian_lower.valuePtr() + col_start,
                                      key_helper.tangent_dim) +=
              linearized_factor.hessian.block(key_helper.factor_offset,
                                              key_helper_j.factor_offset + col_j,
                                              key_helper.tangent_dim, 1);
        }
      } else {
        for (int32_t col_i = 0; col_i < static_cast<int32_t>(key_helper.tangent_dim); ++col_i) {
          const auto col_start = *col_start_iter;
          col_start_iter++;
          Eigen::Map<VectorX<Scalar>>(linearization.hessian_lower.valuePtr() + col_start,
                                      key_helper_j.tangent_dim) +=
              linearized_factor.hessian
                  .block(key_helper.factor_offset + col_i, key_helper_j.factor_offset, 1,
                         key_helper_j.tangent_dim)
                  .transpose();
        }
      }
    }
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::UpdateFromLinearizedSparseFactorIntoSparse(
    const LinearizedSparseFactor& linearized_factor,
    const linearization_sparse_factor_helper_t& factor_helper,
    SparseLinearization<Scalar>& linearization) const {
  // The residual dimension must be the same, even for factors that return VectorX.  If the residual
  // size changes, the optimizer must be re-created.
  SYM_ASSERT(factor_helper.residual_dim == linearized_factor.residual.size());

  // Fill in the combined residual slice
  linearization.residual.segment(factor_helper.combined_residual_offset,
                                 factor_helper.residual_dim) = linearized_factor.residual;

  // Add contribution from right-hand-side
  for (int key_i = 0; key_i < static_cast<int>(factor_helper.key_helpers.size()); ++key_i) {
    const linearization_offsets_t& key_helper = factor_helper.key_helpers[key_i];

    linearization.rhs.segment(key_helper.combined_offset, key_helper.tangent_dim) +=
        linearized_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);
  }

  // Fill out jacobian
  if (include_jacobians_) {
    SYM_ASSERT(factor_helper.jacobian_index_map.size() ==
               static_cast<size_t>(linearized_factor.jacobian.nonZeros()));
    for (int i = 0; i < static_cast<int>(factor_helper.jacobian_index_map.size()); i++) {
      linearization.jacobian.valuePtr()[factor_helper.jacobian_index_map[i]] =
          linearized_factor.jacobian.valuePtr()[i];
    }
  }

  // Fill out hessian
  SYM_ASSERT(factor_helper.hessian_index_map.size() ==
             static_cast<size_t>(linearized_factor.hessian.nonZeros()));
  for (int i = 0; i < static_cast<int>(factor_helper.hessian_index_map.size()); i++) {
    linearization.hessian_lower.valuePtr()[factor_helper.hessian_index_map[i]] +=
        linearized_factor.hessian.valuePtr()[i];
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::UpdateFromDenseFactorIntoTripletLists(
    const LinearizedDenseFactor& linearized_factor,
    const linearization_dense_factor_helper_t& factor_helper,
    std::vector<Eigen::Triplet<Scalar>>& jacobian_triplets,
    std::vector<Eigen::Triplet<Scalar>>& hessian_lower_triplets) const {
  const auto update_triplets_from_blocks =
      [](const int rows, const int cols, const int lhs_row_start, const int lhs_col_start,
         const bool lower_triangle_only, std::vector<Eigen::Triplet<Scalar>>& triplets,
         Eigen::Ref<const MatrixX<ScalarType>> block) {
        for (int block_row = 0; block_row < rows; block_row++) {
          for (int block_col = 0; block_col < (lower_triangle_only ? block_row + 1 : cols);
               block_col++) {
            triplets.emplace_back(lhs_row_start + block_row, lhs_col_start + block_col,
                                  block(block_row, block_col));
          }
        }
      };

  // For each key
  for (int key_i = 0; key_i < static_cast<int>(factor_helper.key_helpers.size()); ++key_i) {
    const linearization_dense_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    // Fill in jacobian block
    if (include_jacobians_) {
      update_triplets_from_blocks(
          factor_helper.residual_dim, key_helper.tangent_dim,
          factor_helper.combined_residual_offset, key_helper.combined_offset, false,
          jacobian_triplets,
          linearized_factor.jacobian.block(0, key_helper.factor_offset, factor_helper.residual_dim,
                                           key_helper.tangent_dim));
    }

    // Add contribution from diagonal hessian block
    update_triplets_from_blocks(
        key_helper.tangent_dim, key_helper.tangent_dim, key_helper.combined_offset,
        key_helper.combined_offset, true, hessian_lower_triplets,
        linearized_factor.hessian.block(key_helper.factor_offset, key_helper.factor_offset,
                                        key_helper.tangent_dim, key_helper.tangent_dim));

    // Add contributions from off-diagonal hessian blocks
    for (int key_j = 0; key_j < key_i; ++key_j) {
      const linearization_dense_key_helper_t& key_helper_j = factor_helper.key_helpers[key_j];

      // If key_j is actually after key_i in the full problem, swap indices to put it in the
      // lower triangle.
      if (key_helper.combined_offset > key_helper_j.combined_offset) {
        update_triplets_from_blocks(
            key_helper.tangent_dim, key_helper_j.tangent_dim, key_helper.combined_offset,
            key_helper_j.combined_offset, false, hessian_lower_triplets,
            linearized_factor.hessian.block(key_helper.factor_offset, key_helper_j.factor_offset,
                                            key_helper.tangent_dim, key_helper_j.tangent_dim));
      } else {
        update_triplets_from_blocks(key_helper_j.tangent_dim, key_helper.tangent_dim,
                                    key_helper_j.combined_offset, key_helper.combined_offset, false,
                                    hessian_lower_triplets,
                                    linearized_factor.hessian
                                        .block(key_helper.factor_offset, key_helper_j.factor_offset,
                                               key_helper.tangent_dim, key_helper_j.tangent_dim)
                                        .transpose());
      }
    }
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::UpdateFromSparseFactorIntoTripletLists(
    const LinearizedSparseFactor& linearized_factor,
    const linearization_sparse_factor_helper_t& factor_helper,
    std::vector<Eigen::Triplet<Scalar>>& jacobian_triplets,
    std::vector<Eigen::Triplet<Scalar>>& hessian_lower_triplets) const {
  std::vector<int> key_for_factor_offset;
  // key_for_factor_offset.reserve();
  for (int key_i = 0; key_i < static_cast<int>(factor_helper.key_helpers.size()); key_i++) {
    for (int key_offset = 0; key_offset < factor_helper.key_helpers[key_i].tangent_dim;
         key_offset++) {
      key_for_factor_offset.push_back(key_i);
    }
  }

  if (include_jacobians_) {
    for (int outer_i = 0; outer_i < linearized_factor.jacobian.outerSize(); ++outer_i) {
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(linearized_factor.jacobian,
                                                                  outer_i);
           it; ++it) {
        const auto row = it.row();
        const auto col = it.col();

        const auto& key_helper = factor_helper.key_helpers[key_for_factor_offset[col]];
        const auto problem_col = col - key_helper.factor_offset + key_helper.combined_offset;
        jacobian_triplets.emplace_back(row + factor_helper.combined_residual_offset, problem_col,
                                       it.value());
      }
    }
  }

  for (int outer_i = 0; outer_i < linearized_factor.hessian.outerSize(); ++outer_i) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(linearized_factor.hessian, outer_i);
         it; ++it) {
      const auto row = it.row();
      const auto col = it.col();

      const auto& key_helper_row = factor_helper.key_helpers[key_for_factor_offset[row]];
      const auto& key_helper_col = factor_helper.key_helpers[key_for_factor_offset[col]];
      const auto problem_row = row - key_helper_row.factor_offset + key_helper_row.combined_offset;
      const auto problem_col = col - key_helper_col.factor_offset + key_helper_col.combined_offset;

      // Put the entry in the lower triangle - even if the factor hessian is lower triangular, the
      // entry might naively go into the upper triangle if the key order is reversed in the full
      // problem
      if (problem_row >= problem_col) {
        hessian_lower_triplets.emplace_back(problem_row, problem_col, it.value());
      } else {
        hessian_lower_triplets.emplace_back(problem_col, problem_row, it.value());
      }
    }
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::EnsureLinearizationHasCorrectSize(
    SparseLinearization<Scalar>& linearization) const {
  if (linearization.residual.size() == 0) {
    // Linearization has never been initialized
    // NOTE(aaron): This is independent of linearization.IsInitialized(), i.e. a Linearization can
    // have been initialized in the past and have the correct sizes/sparsity but have been reset
    SYM_ASSERT(init_linearization_.IsInitialized());

    // Allocate storage of combined linearization
    linearization.residual.resize(init_linearization_.residual.size());
    linearization.rhs.resize(init_linearization_.rhs.size());
    if (include_jacobians_) {
      linearization.jacobian = init_linearization_.jacobian;
    }
    linearization.hessian_lower = init_linearization_.hessian_lower;
    SYM_ASSERT(linearization.jacobian.isCompressed());
    SYM_ASSERT(linearization.hessian_lower.isCompressed());
  } else {
    const int M = init_linearization_.residual.size();
    const int N = init_linearization_.rhs.size();

    SYM_ASSERT(linearization.residual.size() == M);
    if (include_jacobians_) {
      SYM_ASSERT(linearization.jacobian.rows() == M && linearization.jacobian.cols() == N);
    }
    SYM_ASSERT(linearization.hessian_lower.rows() == N && linearization.hessian_lower.cols() == N);
    SYM_ASSERT(linearization.rhs.size() == N);
  }
}

}  // namespace sym

// Explicit instantiation
template class sym::Linearizer<double>;
template class sym::Linearizer<float>;
