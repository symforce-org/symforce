/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./linearizer.h"

#include "./assert.h"
#include "./internal/linearizer_utils.h"

namespace sym {

// ----------------------------------------------------------------------------
// Public Methods
// ----------------------------------------------------------------------------

template <typename ScalarType>
Linearizer<ScalarType>::Linearizer(const std::vector<Factor<Scalar>>& factors,
                                   const std::vector<Key>& key_order)
    : factors_(&factors), dense_linearized_factors_(), sparse_linearized_factors_() {
  if (key_order.empty()) {
    keys_ = ComputeKeysToOptimize(factors);
  } else {
    keys_ = key_order;
  }

  for (const auto& factor : *factors_) {
    if (factor.IsSparse()) {
      sparse_linearized_factors_.emplace_back();
    } else {
      dense_linearized_factors_.emplace_back();
    }
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::Relinearize(const Values<Scalar>& values,
                                         Linearization<Scalar>* const linearization) {
  SYM_ASSERT(linearization != nullptr);

  // Evaluate the factors
  auto sparse_iter = std::begin(sparse_linearized_factors_);
  auto dense_iter = std::begin(dense_linearized_factors_);
  for (const auto& factor : *factors_) {
    if (factor.IsSparse()) {
      factor.Linearize(values, &*sparse_iter);
      ++sparse_iter;
    } else {
      factor.Linearize(values, &*dense_iter);
      ++dense_iter;
    }
  }

  // Allocate matrices and create index if it's the first time
  if (!IsInitialized()) {
    InitializeStorageAndIndices();
  }

  // Update combined problem from factors, using precomputed indices
  BuildCombinedProblemSparse(dense_linearized_factors_, sparse_linearized_factors_, linearization);
}

template <typename ScalarType>
bool Linearizer<ScalarType>::CheckKeysAreContiguousAtStart(const std::vector<Key>& keys,
                                                           size_t* const block_dim) const {
  SYM_ASSERT(!keys.empty());

  auto full_problem_keys_iter = keys_.begin();
  auto keys_iter = keys.begin();
  for (; keys_iter != keys.end(); ++full_problem_keys_iter, ++keys_iter) {
    if (full_problem_keys_iter == keys_.end()) {
      throw std::runtime_error("Keys has extra entries that are not in the full problem");
    }

    if (*full_problem_keys_iter != *keys_iter) {
      if (state_index_.find(keys_iter->GetLcmType()) == state_index_.end()) {
        throw std::runtime_error("Tried to check key which is not in the full problem");
      } else {
        // The next key is in the problem, it's just out of order; so we return false
        return false;
      }
    }
  }

  if (block_dim != nullptr) {
    const auto& index_entry = state_index_.at(keys.back().GetLcmType());
    *block_dim = index_entry.offset + index_entry.tangent_dim;
  }

  return true;
}

template <typename ScalarType>
bool Linearizer<ScalarType>::IsInitialized() const {
  return initialized_;
}

template <typename ScalarType>
std::pair<const std::vector<typename Factor<ScalarType>::LinearizedDenseFactor>&,
          const std::vector<typename Factor<ScalarType>::LinearizedSparseFactor>&>
Linearizer<ScalarType>::LinearizedFactors() const {
  return std::make_pair(dense_linearized_factors_, sparse_linearized_factors_);
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
void Linearizer<ScalarType>::InitializeStorageAndIndices() {
  SYM_ASSERT(!IsInitialized());

  // Get the residual dimension
  int32_t M = 0;
  for (const auto& factor : dense_linearized_factors_) {
    M += factor.residual.rows();
  }
  for (const auto& factor : sparse_linearized_factors_) {
    M += factor.residual.rows();
  }

  // Compute state vector index
  state_index_ = ComputeStateIndex(dense_linearized_factors_, sparse_linearized_factors_, keys_);

  // Create factor helpers
  int32_t combined_residual_offset = 0;
  dense_factor_update_helpers_.reserve(dense_linearized_factors_.size());
  for (const auto& factor : dense_linearized_factors_) {
    dense_factor_update_helpers_.push_back(
        internal::ComputeFactorHelper<Scalar, typename Factor<Scalar>::LinearizedDenseFactor,
                                      linearization_dense_factor_helper_t>(
            factor, state_index_, combined_residual_offset));
  }

  // This will put all the sparse factors at the end of the combined residual, which may not be
  // desirable?
  sparse_factor_update_helpers_.reserve(sparse_linearized_factors_.size());
  for (const auto& factor : sparse_linearized_factors_) {
    sparse_factor_update_helpers_.push_back(
        internal::ComputeFactorHelper<Scalar, typename Factor<Scalar>::LinearizedSparseFactor,
                                      linearization_sparse_factor_helper_t>(
            factor, state_index_, combined_residual_offset));
  }

  // Sanity check
  SYM_ASSERT(combined_residual_offset == M);

  // Get the state dimension
  int32_t N = 0;
  for (const auto& pair : state_index_) {
    N += pair.second.tangent_dim;
  }

  // Allocate storage of combined linearization
  linearization_ones_.residual.resize(M);
  linearization_ones_.rhs.resize(N);
  linearization_ones_.jacobian.resize(M, N);
  linearization_ones_.hessian_lower.resize(N, N);

  // Update combined dense jacobian/hessian from indices, then create a sparseView. This will be
  // filled out assuming nonzeros in factors are all ones so there will not happen to be any
  // numerical zeros.
  BuildCombinedProblemSparsityPattern(&linearization_ones_);

  // Create a hash map from the sparse nonzero indices of the jacobian/hessian to their storage
  // offset within the sparse array.
  const auto jacobian_row_col_to_storage_offset =
      internal::CoordsToStorageOffset(linearization_ones_.jacobian);
  const auto hessian_row_col_to_storage_offset =
      internal::CoordsToStorageOffset(linearization_ones_.hessian_lower);

  // Use the hash map to mark sparse storage offsets for every row of each key block of each factor
  for (int i = 0; i < dense_linearized_factors_.size(); ++i) {
    const LinearizedDenseFactor& linearized_factor = dense_linearized_factors_[i];
    linearization_dense_factor_helper_t& factor_helper = dense_factor_update_helpers_[i];
    internal::ComputeKeyHelperSparseColOffsets<Scalar>(
        linearized_factor, jacobian_row_col_to_storage_offset, hessian_row_col_to_storage_offset,
        factor_helper);
  }
  for (int i = 0; i < sparse_linearized_factors_.size(); ++i) {
    const LinearizedSparseFactor& linearized_factor = sparse_linearized_factors_[i];
    linearization_sparse_factor_helper_t& factor_helper = sparse_factor_update_helpers_[i];
    internal::ComputeKeyHelperSparseMap<Scalar>(linearized_factor,
                                                jacobian_row_col_to_storage_offset,
                                                hessian_row_col_to_storage_offset, factor_helper);
  }

  initialized_ = true;
}

template <typename ScalarType>
std::unordered_map<key_t, index_entry_t> Linearizer<ScalarType>::ComputeStateIndex(
    const std::vector<LinearizedDenseFactor>& factors,
    const std::vector<LinearizedSparseFactor>& sparse_factors, const std::vector<Key>& keys) {
  // Convert keys to set
  const std::unordered_set<Key> key_set(keys.begin(), keys.end());

  // Aggregate index entries from linearized factors
  std::unordered_map<key_t, index_entry_t> state_index;

  const auto add_factor_to_index = [&key_set, &state_index](const auto& factor) {
    for (const index_entry_t& entry : factor.index.entries) {
      // Skip keys that are not optimized (i.e. not in keys)
      if (key_set.count(entry.key) == 0) {
        continue;
      }

      // Add the entry if not present, otherwise just check consistency
      auto it = state_index.find(entry.key);
      if (it == state_index.end()) {
        state_index[entry.key] = entry;
      } else {
        SYM_ASSERT(it->second.type == entry.type);
        SYM_ASSERT(it->second.storage_dim == entry.storage_dim);
        SYM_ASSERT(it->second.tangent_dim == entry.tangent_dim);
      }
    }
  };

  for (const auto& factor : factors) {
    add_factor_to_index(factor);
  }

  for (const auto& factor : sparse_factors) {
    add_factor_to_index(factor);
  }

  // Sanity check
  // NOTE(aaron): If this fails, it probably means you've passed keys to optimize that don't have
  // any corresponding factors
  SYM_ASSERT(state_index.size() == keys.size());

  // Go back through and set offsets relative to the key ordering
  // This is setting the state vector, and jacobian/hessian dimension.
  int32_t offset = 0;
  for (const Key& key : keys) {
    index_entry_t& entry = state_index.at(key.GetLcmType());
    entry.offset = offset;
    offset += entry.tangent_dim;
  }

  return state_index;
}

template <typename ScalarType>
void Linearizer<ScalarType>::UpdateFromLinearizedDenseFactorIntoSparse(
    const LinearizedDenseFactor& linearized_factor,
    const linearization_dense_factor_helper_t& factor_helper,
    Linearization<Scalar>* const linearization) const {
  // The residual dimension must be the same, even for factors that return VectorX.  If the residual
  // size changes, the optimizer must be re-created.
  SYM_ASSERT(factor_helper.residual_dim == linearized_factor.residual.size());

  // Fill in the combined residual slice
  linearization->residual.segment(factor_helper.combined_residual_offset,
                                  factor_helper.residual_dim) = linearized_factor.residual;

  // For each key
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    const linearization_dense_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    // Fill in jacobian block, column by column
    for (int col_block = 0; col_block < key_helper.tangent_dim; ++col_block) {
      Eigen::Map<VectorX<Scalar>>(
          linearization->jacobian.valuePtr() + key_helper.jacobian_storage_col_starts[col_block],
          factor_helper.residual_dim) =
          linearized_factor.jacobian.block(0, key_helper.factor_offset + col_block,
                                           factor_helper.residual_dim, 1);
    }

    // Add contribution from right-hand-side
    linearization->rhs.segment(key_helper.combined_offset, key_helper.tangent_dim) +=
        linearized_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);

    // Add contribution from diagonal hessian block, column by column
    for (int col_block = 0; col_block < key_helper.tangent_dim; ++col_block) {
      const std::vector<int32_t>& diag_col_starts = key_helper.hessian_storage_col_starts[key_i];
      Eigen::Map<VectorX<Scalar>>(
          linearization->hessian_lower.valuePtr() + diag_col_starts[col_block],
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
      const std::vector<int32_t>& col_starts = key_helper.hessian_storage_col_starts[key_j];

      if (key_helper_j.combined_offset < key_helper.combined_offset) {
        for (int32_t col_j = 0; col_j < col_starts.size(); ++col_j) {
          Eigen::Map<VectorX<Scalar>>(linearization->hessian_lower.valuePtr() + col_starts[col_j],
                                      key_helper.tangent_dim) +=
              linearized_factor.hessian.block(key_helper.factor_offset,
                                              key_helper_j.factor_offset + col_j,
                                              key_helper.tangent_dim, 1);
        }
      } else {
        for (int32_t col_i = 0; col_i < col_starts.size(); ++col_i) {
          Eigen::Map<VectorX<Scalar>>(linearization->hessian_lower.valuePtr() + col_starts[col_i],
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
    Linearization<Scalar>* const linearization) const {
  // The residual dimension must be the same, even for factors that return VectorX.  If the residual
  // size changes, the optimizer must be re-created.
  SYM_ASSERT(factor_helper.residual_dim == linearized_factor.residual.size());

  // Fill in the combined residual slice
  linearization->residual.segment(factor_helper.combined_residual_offset,
                                  factor_helper.residual_dim) = linearized_factor.residual;

  // Add contribution from right-hand-side
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    const linearization_sparse_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    linearization->rhs.segment(key_helper.combined_offset, key_helper.tangent_dim) +=
        linearized_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);
  }

  // Fill out jacobian
  SYM_ASSERT(factor_helper.jacobian_index_map.size() == linearized_factor.jacobian.nonZeros());
  for (int i = 0; i < factor_helper.jacobian_index_map.size(); i++) {
    linearization->jacobian.valuePtr()[factor_helper.jacobian_index_map[i]] =
        linearized_factor.jacobian.valuePtr()[i];
  }

  // Fill out hessian
  SYM_ASSERT(factor_helper.hessian_index_map.size() == linearized_factor.hessian.nonZeros());
  for (int i = 0; i < factor_helper.hessian_index_map.size(); i++) {
    linearization->hessian_lower.valuePtr()[factor_helper.hessian_index_map[i]] +=
        linearized_factor.hessian.valuePtr()[i];
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::UpdatePatternFromDenseFactorIntoTripletLists(
    const linearization_dense_factor_helper_t& factor_helper,
    std::vector<Eigen::Triplet<Scalar>>* const jacobian_triplets,
    std::vector<Eigen::Triplet<Scalar>>* const hessian_lower_triplets) const {
  const auto update_triplets_from_blocks =
      [](const int rows, const int cols, const int lhs_row_start, const int lhs_col_start,
         const bool lower_triangle_only, std::vector<Eigen::Triplet<Scalar>>* const triplets) {
        for (int block_row = 0; block_row < rows; block_row++) {
          for (int block_col = 0; block_col < (lower_triangle_only ? block_row + 1 : cols);
               block_col++) {
            triplets->emplace_back(lhs_row_start + block_row, lhs_col_start + block_col, 1);
          }
        }
      };

  // For each key
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    const linearization_dense_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    // Fill in jacobian block
    update_triplets_from_blocks(factor_helper.residual_dim, key_helper.tangent_dim,
                                factor_helper.combined_residual_offset, key_helper.combined_offset,
                                false, jacobian_triplets);

    // Add contribution from diagonal hessian block
    update_triplets_from_blocks(key_helper.tangent_dim, key_helper.tangent_dim,
                                key_helper.combined_offset, key_helper.combined_offset, true,
                                hessian_lower_triplets);

    // Add contributions from off-diagonal hessian blocks
    for (int key_j = 0; key_j < key_i; ++key_j) {
      const linearization_dense_key_helper_t& key_helper_j = factor_helper.key_helpers[key_j];
      if (key_helper.combined_offset > key_helper_j.combined_offset) {
        update_triplets_from_blocks(key_helper.tangent_dim, key_helper_j.tangent_dim,
                                    key_helper.combined_offset, key_helper_j.combined_offset, false,
                                    hessian_lower_triplets);
      } else {
        // If key_j is actually after key_i in the full problem, swap indices to put it in the lower
        // triangle
        update_triplets_from_blocks(key_helper_j.tangent_dim, key_helper.tangent_dim,
                                    key_helper_j.combined_offset, key_helper.combined_offset, false,
                                    hessian_lower_triplets);
      }
    }
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::UpdatePatternFromSparseFactorIntoTripletLists(
    const LinearizedSparseFactor& linearized_factor,
    const linearization_sparse_factor_helper_t& factor_helper,
    std::vector<Eigen::Triplet<Scalar>>* const jacobian_triplets,
    std::vector<Eigen::Triplet<Scalar>>* const hessian_lower_triplets) const {
  std::vector<int> key_for_factor_offset;
  // key_for_factor_offset.reserve();
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); key_i++) {
    for (int key_offset = 0; key_offset < factor_helper.key_helpers[key_i].tangent_dim;
         key_offset++) {
      key_for_factor_offset.push_back(key_i);
    }
  }

  for (int outer_i = 0; outer_i < linearized_factor.jacobian.outerSize(); ++outer_i) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(linearized_factor.jacobian,
                                                                outer_i);
         it; ++it) {
      const auto row = it.row();
      const auto col = it.col();

      const auto& key_helper = factor_helper.key_helpers[key_for_factor_offset[col]];
      const auto problem_col = col - key_helper.factor_offset + key_helper.combined_offset;
      jacobian_triplets->emplace_back(row + factor_helper.combined_residual_offset, problem_col, 1);
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
        hessian_lower_triplets->emplace_back(problem_row, problem_col, 1);
      } else {
        hessian_lower_triplets->emplace_back(problem_col, problem_row, 1);
      }
    }
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::EnsureLinearizationHasCorrectSize(
    Linearization<Scalar>* const linearization) const {
  if (linearization->residual.size() == 0) {
    // Linearization has never been initialized
    // NOTE(aaron): This is independent of linearization.IsInitialized(), i.e. a Linearization can
    // have been initialized in the past and have the correct sizes/sparsity but have been reset
    SYM_ASSERT(linearization_ones_.IsInitialized());

    // Allocate storage of combined linearization
    linearization->residual.resize(linearization_ones_.residual.size());
    linearization->rhs.resize(linearization_ones_.rhs.size());
    linearization->jacobian = linearization_ones_.jacobian;
    linearization->hessian_lower = linearization_ones_.hessian_lower;
    SYM_ASSERT(linearization->jacobian.isCompressed());
    SYM_ASSERT(linearization->hessian_lower.isCompressed());
  } else {
    const int M = linearization_ones_.residual.size();
    const int N = linearization_ones_.rhs.size();

    SYM_ASSERT(linearization->residual.size() == M);
    SYM_ASSERT(linearization->jacobian.rows() == M && linearization->jacobian.cols() == N);
    SYM_ASSERT(linearization->hessian_lower.rows() == N &&
               linearization->hessian_lower.cols() == N);
    SYM_ASSERT(linearization->rhs.size() == N);
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::BuildCombinedProblemSparse(
    const std::vector<LinearizedDenseFactor>& dense_linearized_factors,
    const std::vector<LinearizedSparseFactor>& sparse_linearized_factors,
    Linearization<Scalar>* const linearization) const {
  EnsureLinearizationHasCorrectSize(linearization);

  // Zero out blocks that are built additively
  linearization->rhs.setZero();
  Eigen::Map<VectorX<Scalar>>(linearization->hessian_lower.valuePtr(),
                              linearization->hessian_lower.nonZeros())
      .setZero();

  // Update each factor using precomputed index helpers
  for (int i = 0; i < dense_linearized_factors.size(); ++i) {
    UpdateFromLinearizedDenseFactorIntoSparse(dense_linearized_factors[i],
                                              dense_factor_update_helpers_[i], linearization);
  }
  for (int i = 0; i < sparse_linearized_factors.size(); ++i) {
    UpdateFromLinearizedSparseFactorIntoSparse(sparse_linearized_factors[i],
                                               sparse_factor_update_helpers_[i], linearization);
  }

  linearization->SetInitialized();
}

template <typename ScalarType>
void Linearizer<ScalarType>::BuildCombinedProblemSparsityPattern(
    Linearization<Scalar>* const linearization) const {
  std::vector<Eigen::Triplet<Scalar>> jacobian_triplets;
  std::vector<Eigen::Triplet<Scalar>> hessian_lower_triplets;

  // Update each factor using precomputed index helpers
  for (int i = 0; i < dense_factor_update_helpers_.size(); ++i) {
    UpdatePatternFromDenseFactorIntoTripletLists(dense_factor_update_helpers_[i],
                                                 &jacobian_triplets, &hessian_lower_triplets);
  }
  for (int i = 0; i < sparse_linearized_factors_.size(); i++) {
    UpdatePatternFromSparseFactorIntoTripletLists(sparse_linearized_factors_[i],
                                                  sparse_factor_update_helpers_[i],
                                                  &jacobian_triplets, &hessian_lower_triplets);
  }

  // Create the sparse matrices
  {
    linearization->jacobian.setFromTriplets(jacobian_triplets.begin(), jacobian_triplets.end());
    linearization->hessian_lower.setFromTriplets(hessian_lower_triplets.begin(),
                                                 hessian_lower_triplets.end());
    SYM_ASSERT(linearization->jacobian.isCompressed());
    SYM_ASSERT(linearization->hessian_lower.isCompressed());
  }

  linearization->SetInitialized();
}

}  // namespace sym

// Explicit instantiation
template class sym::Linearizer<double>;
template class sym::Linearizer<float>;
