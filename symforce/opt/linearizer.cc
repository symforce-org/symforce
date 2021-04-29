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
    : factors_(&factors), linearized_factors_(factors.size()) {
  if (key_order.empty()) {
    keys_ = ComputeKeysToOptimize(factors, &Key::LexicalLessThan);
  } else {
    keys_ = key_order;
  }
}

template <typename ScalarType>
void Linearizer<ScalarType>::Relinearize(const Values<Scalar>& values,
                                         Linearization<Scalar>* const linearization) {
  SYM_ASSERT(linearization != nullptr);

  // Evaluate the factors
  for (size_t i = 0; i < factors_->size(); ++i) {
    (*factors_)[i].Linearize(values, &linearized_factors_[i]);
  }

  // Allocate matrices and create index if it's the first time
  if (!IsInitialized()) {
    InitializeStorageAndIndices();
  }

// Update combined problem from factors, using precomputed indices
#if 1
  BuildCombinedProblemSparse(linearized_factors_, linearization);
#else
  BuildCombinedProblemDenseThenSparseView(linearized_factors_, linearization);
#endif
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

    if (full_problem_keys_iter->GetLcmType() != keys_iter->GetLcmType()) {
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
const std::vector<typename Factor<ScalarType>::LinearizedFactor>&
Linearizer<ScalarType>::LinearizedFactors() const {
  return linearized_factors_;
}

template <typename ScalarType>
const std::vector<Key>& Linearizer<ScalarType>::Keys() const {
  return keys_;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

template <typename ScalarType>
void Linearizer<ScalarType>::InitializeStorageAndIndices() {
  SYM_ASSERT(!IsInitialized());

  // Get the residual dimension
  int32_t M = 0;
  for (const auto& factor : linearized_factors_) {
    M += factor.residual.rows();
  }

  // Compute state vector index
  state_index_ = ComputeStateIndex(linearized_factors_, keys_);

  // Create factor helpers
  int32_t combined_residual_offset = 0;
  factor_update_helpers_.reserve(linearized_factors_.size());
  for (const auto& factor : linearized_factors_) {
    factor_update_helpers_.push_back(
        internal::ComputeFactorHelper<Scalar>(factor, state_index_, combined_residual_offset));
  }
  // Sanity check
  SYM_ASSERT(combined_residual_offset == M);

  // Get the state dimension
  int32_t N = 0;
  for (const auto& pair : state_index_) {
    N += pair.second.tangent_dim;
  }

  // Create linearized factors with all one values, to be used to create sparsity pattern
  std::vector<LinearizedFactor> one_value_linearized_factors;
  one_value_linearized_factors.reserve(linearized_factors_.size());
  for (int i = 0; i < linearized_factors_.size(); ++i) {
    one_value_linearized_factors.push_back(
        internal::CopyLinearizedFactorAllOnes<Scalar>(linearized_factors_[i]));
  }

  // Allocate storage of combined linearization
  linearization_ones_.residual.resize(M);
  linearization_ones_.rhs.resize(N);
  linearization_ones_.jacobian.resize(M, N);
  linearization_ones_.hessian_lower.resize(N, N);

  // Update combined dense jacobian/hessian from indices, then create a sparseView. This will
  // yield an accurate symbolic sparsity pattern because we're using all ones so there will not
  // happen to be any numerical zeros.
  BuildCombinedProblemFromOnesFactors(one_value_linearized_factors, &linearization_ones_);

  // Create a hash map from the sparse nonzero indices of the jacobian/hessian to their storage
  // offset within the sparse array.
  const auto jacobian_row_col_to_storage_offset =
      internal::CoordsToStorageOffset(linearization_ones_.jacobian);
  const auto hessian_row_col_to_storage_offset =
      internal::CoordsToStorageOffset(linearization_ones_.hessian_lower);

  // Use the hash map to mark sparse storage offsets for every row of each key block of each factor
  for (int i = 0; i < linearized_factors_.size(); ++i) {
    const LinearizedFactor& linearized_factor = linearized_factors_[i];
    linearization_factor_helper_t& factor_helper = factor_update_helpers_[i];
    internal::ComputeKeyHelperSparseColOffsets<Scalar>(
        linearized_factor, jacobian_row_col_to_storage_offset, hessian_row_col_to_storage_offset,
        factor_helper);
  }

  initialized_ = true;
}

template <typename ScalarType>
std::unordered_map<key_t, index_entry_t> Linearizer<ScalarType>::ComputeStateIndex(
    const std::vector<LinearizedFactor>& factors, const std::vector<Key>& keys) {
  // Convert keys to set
  const std::unordered_set<Key> key_set(keys.begin(), keys.end());

  // Aggregate index entries from linearized factors
  std::unordered_map<key_t, index_entry_t> state_index;
  for (const auto& factor : factors) {
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
void Linearizer<ScalarType>::UpdateFromLinearizedFactorIntoSparse(
    const LinearizedFactor& linearized_factor, const linearization_factor_helper_t& factor_helper,
    Linearization<Scalar>* const linearization) const {
  // Fill in the combined residual slice
  linearization->residual.segment(factor_helper.combined_residual_offset,
                                  factor_helper.residual_dim) = linearized_factor.residual;

  // For each key
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    const linearization_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

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
      const linearization_key_helper_t& key_helper_j = factor_helper.key_helpers[key_j];
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
void Linearizer<ScalarType>::UpdateFromLinearizedFactorIntoTripletLists(
    const LinearizedFactor& linearized_factor, const linearization_factor_helper_t& factor_helper,
    sym::VectorX<Scalar>* const residual, sym::VectorX<Scalar>* const rhs,
    std::vector<Eigen::Triplet<Scalar>>* const jacobian_triplets,
    std::vector<Eigen::Triplet<Scalar>>* const hessian_lower_triplets) const {
  // NOTE(aaron):  This function could be simplified if we assume the linearized_factor
  // linearization is all ones and dense.  However, in the future we may want to allow for sparse
  // linearizations for individual factors, and the implementation for that is much closer to what's
  // here now; so, I'm leaving it this way

  // Fill in the combined residual slice
  residual->segment(factor_helper.combined_residual_offset, factor_helper.residual_dim) =
      linearized_factor.residual;

  const auto update_triplets_from_blocks =
      [](const int rhs_row_start, const int rhs_col_start, const int rows, const int cols,
         const int lhs_row_start, const int lhs_col_start, const sym::MatrixX<Scalar>& rhs,
         const bool lower_triangle_only, std::vector<Eigen::Triplet<Scalar>>* const triplets) {
        const auto rhs_block = rhs.block(rhs_row_start, rhs_col_start, rows, cols);
        for (int block_row = 0; block_row < rows; block_row++) {
          for (int block_col = 0; block_col < (lower_triangle_only ? block_row + 1 : cols);
               block_col++) {
            triplets->emplace_back(lhs_row_start + block_row, lhs_col_start + block_col,
                                   rhs_block(block_row, block_col));
          }
        }
      };

  // For each key
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    const linearization_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    // Fill in jacobian block
    update_triplets_from_blocks(0, key_helper.factor_offset, factor_helper.residual_dim,
                                key_helper.tangent_dim, factor_helper.combined_residual_offset,
                                key_helper.combined_offset, linearized_factor.jacobian, false,
                                jacobian_triplets);

    // Add contribution from right-hand-side
    rhs->segment(key_helper.combined_offset, key_helper.tangent_dim) +=
        linearized_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);

    // Add contribution from diagonal hessian block
    update_triplets_from_blocks(key_helper.factor_offset, key_helper.factor_offset,
                                key_helper.tangent_dim, key_helper.tangent_dim,
                                key_helper.combined_offset, key_helper.combined_offset,
                                linearized_factor.hessian, true, hessian_lower_triplets);

    // Add contributions from off-diagonal hessian blocks
    for (int key_j = 0; key_j < key_i; ++key_j) {
      const linearization_key_helper_t& key_helper_j = factor_helper.key_helpers[key_j];
      if (key_helper.combined_offset > key_helper_j.combined_offset) {
        update_triplets_from_blocks(key_helper.factor_offset, key_helper_j.factor_offset,
                                    key_helper.tangent_dim, key_helper_j.tangent_dim,
                                    key_helper.combined_offset, key_helper_j.combined_offset,
                                    linearized_factor.hessian, false, hessian_lower_triplets);
      } else {
        // If key_j is actually after key_i in the full problem, swap indices to put it in the lower
        // triangle
        update_triplets_from_blocks(
            key_helper_j.factor_offset, key_helper.factor_offset, key_helper_j.tangent_dim,
            key_helper.tangent_dim, key_helper_j.combined_offset, key_helper.combined_offset,
            linearized_factor.hessian.transpose(), false, hessian_lower_triplets);
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
    const std::vector<LinearizedFactor>& linearized_factors,
    Linearization<Scalar>* const linearization) const {
  EnsureLinearizationHasCorrectSize(linearization);

  // Zero out blocks that are built additively
  linearization->rhs.setZero();
  Eigen::Map<VectorX<Scalar>>(linearization->hessian_lower.valuePtr(),
                              linearization->hessian_lower.nonZeros())
      .setZero();

  // Update each factor using precomputed index helpers
  for (int i = 0; i < linearized_factors.size(); ++i) {
    UpdateFromLinearizedFactorIntoSparse(linearized_factors[i], factor_update_helpers_[i],
                                         linearization);
  }

  linearization->SetInitialized();
}

template <typename ScalarType>
void Linearizer<ScalarType>::BuildCombinedProblemFromOnesFactors(
    const std::vector<LinearizedFactor>& all_ones_linearized_factors,
    Linearization<Scalar>* const linearization) {
  // Zero out the rhs, since it's built additively
  linearization->rhs.setZero();

  std::vector<Eigen::Triplet<Scalar>> jacobian_triplets;
  std::vector<Eigen::Triplet<Scalar>> hessian_lower_triplets;

  // Update each factor using precomputed index helpers
  for (int i = 0; i < all_ones_linearized_factors.size(); ++i) {
    UpdateFromLinearizedFactorIntoTripletLists(
        all_ones_linearized_factors[i], factor_update_helpers_[i], &linearization->residual,
        &linearization->rhs, &jacobian_triplets, &hessian_lower_triplets);
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
