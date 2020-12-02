#include "./linearization.h"

#include <iostream>

#include "./assert.h"

namespace sym {

template <typename ScalarType>
std::unordered_map<key_t, index_entry_t> Linearization<ScalarType>::ComputeStateIndex(
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

// Ordered map from (row, col) location in a dense matrix to a storage offset in a sparse matrix
// Ordered by column then row, and only supports constructing with strictly increasing keys and size
// known ahead of time
class CoordsToStorageOrderedMap {
 public:
  CoordsToStorageOrderedMap() : data_() {}

  void reserve(const size_t size) {
    data_.reserve(size);
  }

  void insert(const std::pair<std::pair<int32_t, int32_t>, int32_t>& key_and_value) {
    const auto& key = key_and_value.first;
    const auto& value = key_and_value.second;
    SYM_ASSERT(data_.empty() || ColumnOrderingLessThan(data_.back().first, key));
    data_.emplace_back(key, value);
  }

  int32_t at(const std::pair<int32_t, int32_t>& key) const {
    // Binary search data_ for key
    const auto at_and_after_iterator =
        std::equal_range(data_.begin(), data_.end(), std::make_pair(key, /* unused value */ 0),
                         ColumnOrderingPairLessThan);
    if (at_and_after_iterator.first + 1 != at_and_after_iterator.second) {
      throw std::out_of_range("Key not in CoordsToStorageMap");
    }
    return at_and_after_iterator.first->second;
  }

 private:
  static bool ColumnOrderingPairLessThan(
      const std::pair<std::pair<int32_t, int32_t>, int32_t>& a_pair,
      const std::pair<std::pair<int32_t, int32_t>, int32_t>& b_pair) {
    return ColumnOrderingLessThan(a_pair.first, b_pair.first);
  }

  static bool ColumnOrderingLessThan(const std::pair<int32_t, int32_t>& a,
                                     const std::pair<int32_t, int32_t>& b) {
    return std::make_pair(a.second, a.first) < std::make_pair(b.second, b.first);
  }

  std::vector<std::pair<std::pair<int32_t, int32_t>, int32_t>> data_;
};

struct StdPairHash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    std::size_t ret = 0;
    sym::internal::hash_combine(ret, x.first, x.second);
    return ret;
  }
};

// TODO(aaron):  For large enough problems we'll probably want the option to switch this to
// std::unordered_map
using CoordsToStorageMap = CoordsToStorageOrderedMap;

template <typename Scalar>
CoordsToStorageMap CoordsToStorageOffset(const Eigen::SparseMatrix<Scalar>& mat) {
  CoordsToStorageMap coords_to_storage_offset;
  coords_to_storage_offset.reserve(mat.nonZeros());
  int32_t storage_index = 0;
  for (int col = 0; col < mat.outerSize(); ++col) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(mat, col); it; ++it) {
      coords_to_storage_offset.insert(
          std::make_pair(std::make_pair(it.row(), it.col()), storage_index));
      storage_index += 1;
    }
  }
  SYM_ASSERT(storage_index == mat.nonZeros());
  return coords_to_storage_offset;
}

template <typename Scalar>
void ComputeKeyHelperSparseColOffsets(
    const typename Factor<Scalar>::LinearizedFactor& linearized_factor,
    const CoordsToStorageMap& jacobian_row_col_to_storage_offset,
    const CoordsToStorageMap& hessian_row_col_to_storage_offset,
    linearization_factor_helper_t& factor_helper) {
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    linearization_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    key_helper.jacobian_storage_col_starts.resize(key_helper.tangent_dim);
    for (int32_t col = 0; col < key_helper.tangent_dim; ++col) {
      key_helper.jacobian_storage_col_starts[col] = jacobian_row_col_to_storage_offset.at(
          std::make_pair(factor_helper.combined_residual_offset, key_helper.combined_offset + col));
    }

    key_helper.hessian_storage_col_starts.resize(key_i + 1);
    key_helper.num_other_keys = key_helper.hessian_storage_col_starts.size();

    // Diagonal block
    std::vector<int32_t>& diag_col_starts = key_helper.hessian_storage_col_starts[key_i];
    diag_col_starts.resize(key_helper.tangent_dim);
    for (int32_t col = 0; col < key_helper.tangent_dim; ++col) {
      diag_col_starts[col] = hessian_row_col_to_storage_offset.at(
          std::make_pair(key_helper.combined_offset + col, key_helper.combined_offset + col));
    }

    // Off diagonal blocks
    for (int key_j = 0; key_j < key_i; key_j++) {
      const linearization_key_helper_t& j_key_helper = factor_helper.key_helpers[key_j];
      std::vector<int32_t>& col_starts = key_helper.hessian_storage_col_starts[key_j];

      // If key_j comes after key_i in the full problem, we need to transpose things
      if (j_key_helper.combined_offset < key_helper.combined_offset) {
        col_starts.resize(j_key_helper.tangent_dim);
        for (int32_t j_col = 0; j_col < j_key_helper.tangent_dim; ++j_col) {
          col_starts[j_col] = hessian_row_col_to_storage_offset.at(
              std::make_pair(key_helper.combined_offset, j_key_helper.combined_offset + j_col));
        }
      } else {
        col_starts.resize(key_helper.tangent_dim);
        for (int32_t i_col = 0; i_col < key_helper.tangent_dim; ++i_col) {
          col_starts[i_col] = hessian_row_col_to_storage_offset.at(
              std::make_pair(j_key_helper.combined_offset, key_helper.combined_offset + i_col));
        }
      }
    }
  }
}

template <typename ScalarType>
Linearization<ScalarType>::Linearization(const std::vector<Factor<Scalar>>& factors,
                                         const Values<Scalar>& values,
                                         const std::vector<Key>& key_order)
    : factors_(&factors), linearized_factors_(factors.size()) {
  if (key_order.empty()) {
    keys_ = ComputeKeysToOptimize(factors, &Key::LexicalLessThan);
  } else {
    keys_ = key_order;
  }

  Relinearize(values);
}

template <typename ScalarType>
bool Linearization<ScalarType>::IsInitialized() const {
  return residual_.size() > 0;
}

/**
 * Create a linearized factor with the same structure as the given one, but with all ones
 * in the entries.
 */
template <typename Scalar>
typename Factor<Scalar>::LinearizedFactor CopyLinearizedFactorAllOnes(
    const typename Factor<Scalar>::LinearizedFactor& factor) {
  typename Factor<Scalar>::LinearizedFactor one_value_factor;
  one_value_factor.index = factor.index;
  one_value_factor.residual =
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Ones(factor.residual.rows());
  one_value_factor.jacobian = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Ones(
      factor.jacobian.rows(), factor.jacobian.cols());
  one_value_factor.hessian = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Ones(
      factor.hessian.rows(), factor.hessian.cols());
  one_value_factor.rhs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Ones(factor.rhs.rows());
  return one_value_factor;
}

template <typename Scalar>
linearization_factor_helper_t ComputeFactorHelper(
    const typename Factor<Scalar>::LinearizedFactor& factor,
    const std::unordered_map<key_t, index_entry_t>& state_index, int& combined_residual_offset) {
  linearization_factor_helper_t factor_helper;

  factor_helper.residual_dim = factor.residual.rows();
  factor_helper.combined_residual_offset = combined_residual_offset;

  for (int key_i = 0; key_i < factor.index.entries.size(); ++key_i) {
    const index_entry_t& entry = factor.index.entries[key_i];

    const bool key_is_optimized = state_index.count(entry.key) > 0;
    if (!key_is_optimized) {
      continue;
    }

    factor_helper.key_helpers.emplace_back();
    linearization_key_helper_t& key_helper = factor_helper.key_helpers.back();

    // Offset of this key within the factor's state
    key_helper.factor_offset = entry.offset;
    key_helper.tangent_dim = entry.tangent_dim;

    // Offset of this key within the combined state
    key_helper.combined_offset = state_index.at(entry.key).offset;
  }

  // Increment offset into the combined residual
  combined_residual_offset += factor_helper.residual_dim;

  return factor_helper;
}

template <typename ScalarType>
void Linearization<ScalarType>::InitializeStorageAndIndices() {
  SYM_ASSERT(!IsInitialized());

  // Get the residual dimension
  int32_t M = 0;
  for (const auto& factor : linearized_factors_) {
    M += factor.residual.rows();
  }

  // Compute state vector index
  const std::unordered_map<key_t, index_entry_t> state_index =
      ComputeStateIndex(linearized_factors_, keys_);

  // Create factor helpers
  int32_t combined_residual_offset = 0;
  factor_update_helpers_.reserve(linearized_factors_.size());
  for (const auto& factor : linearized_factors_) {
    factor_update_helpers_.push_back(
        ComputeFactorHelper<Scalar>(factor, state_index, combined_residual_offset));
  }
  // Sanity check
  SYM_ASSERT(combined_residual_offset == M);

  // Get the state dimension
  int32_t N = 0;
  for (const auto& pair : state_index) {
    N += pair.second.tangent_dim;
  }

  // Allocate storage of combined linearization
  residual_.resize(M, 1);
  rhs_.resize(N, 1);
  jacobian_.resize(M, N);
  hessian_lower_.resize(N, N);

  // Zero out
  residual_.setZero();
  rhs_.setZero();
  jacobian_.setZero();
  hessian_lower_.setZero();

  // Create linearized factors with all one values, to be used to create sparsity pattern
  std::vector<LinearizedFactor> one_value_linearized_factors;
  one_value_linearized_factors.reserve(linearized_factors_.size());
  for (int i = 0; i < linearized_factors_.size(); ++i) {
    one_value_linearized_factors.push_back(
        CopyLinearizedFactorAllOnes<Scalar>(linearized_factors_[i]));
  }

  // Update combined dense jacobian/hessian from indices, then create a sparseView. This will
  // yield an accurate symbolic sparsity pattern because we're using all ones so there will not
  // happen to be any numerical zeros.
  BuildCombinedProblemDenseThenSparseView(one_value_linearized_factors);

  // Create a hash map from the sparse nonzero indices of the jacobian/hessian to their storage
  // offset within the sparse array.
  const auto jacobian_row_col_to_storage_offset = CoordsToStorageOffset(jacobian_sparse_);
  const auto hessian_row_col_to_storage_offset = CoordsToStorageOffset(hessian_lower_sparse_);

  // Use the hash map to mark sparse storage offsets for every row of each key block of each factor
  for (int i = 0; i < linearized_factors_.size(); ++i) {
    const LinearizedFactor& linearized_factor = linearized_factors_[i];
    linearization_factor_helper_t& factor_helper = factor_update_helpers_[i];
    ComputeKeyHelperSparseColOffsets<Scalar>(linearized_factor, jacobian_row_col_to_storage_offset,
                                             hessian_row_col_to_storage_offset, factor_helper);
  }
}

template <typename ScalarType>
void Linearization<ScalarType>::Relinearize(const Values<Scalar>& values) {
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
  BuildCombinedProblemSparse(linearized_factors_);
#else
  BuildCombinedProblemDenseThenSparseView(linearized_factors_);
#endif
}

template <typename ScalarType>
void Linearization<ScalarType>::BuildCombinedProblemSparse(
    const std::vector<LinearizedFactor>& linearized_factors) {
  // Zero out blocks that are built additively
  rhs_.setZero();
  Eigen::Map<sym::VectorX<Scalar>>(hessian_lower_sparse_.valuePtr(),
                                   hessian_lower_sparse_.nonZeros())
      .setZero();

  // Update each factor using precomputed index helpers
  for (int i = 0; i < linearized_factors.size(); ++i) {
    UpdateFromLinearizedFactorIntoSparse(linearized_factors[i], factor_update_helpers_[i]);
  }
}

template <typename ScalarType>
void Linearization<ScalarType>::BuildCombinedProblemDenseThenSparseView(
    const std::vector<LinearizedFactor>& linearized_factors) {
  // Zero out blocks that are built additively
  rhs_.setZero();
  hessian_lower_.template triangularView<Eigen::Lower>().setZero();

  // Update each factor using precomputed index helpers
  for (int i = 0; i < linearized_factors.size(); ++i) {
    UpdateFromLinearizedFactorIntoDense(linearized_factors[i], factor_update_helpers_[i]);
  }

  // Convert to sparse matrices, to interface with the solver
  // NOTE(hayk): This will NOT yield the expected sparsity pattern if there are numerical
  // (but not symbolic) zeros in these matrices.
  {
    jacobian_sparse_ = jacobian_.sparseView();
    hessian_lower_sparse_ = hessian_lower_.sparseView();
    SYM_ASSERT(jacobian_sparse_.isCompressed());
    SYM_ASSERT(hessian_lower_sparse_.isCompressed());
  }
}

template <typename ScalarType>
void Linearization<ScalarType>::UpdateFromLinearizedFactorIntoDense(
    const LinearizedFactor& linearized_factor, const linearization_factor_helper_t& factor_helper) {
  // Fill in the combined residual slice
  residual_.segment(factor_helper.combined_residual_offset, factor_helper.residual_dim) =
      linearized_factor.residual;

  // For each key
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    const linearization_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    // Fill in jacobian block
    jacobian_.block(factor_helper.combined_residual_offset, key_helper.combined_offset,
                    factor_helper.residual_dim, key_helper.tangent_dim) =
        linearized_factor.jacobian.block(0, key_helper.factor_offset, factor_helper.residual_dim,
                                         key_helper.tangent_dim);

    // Add contribution from right-hand-side
    rhs_.segment(key_helper.combined_offset, key_helper.tangent_dim) +=
        linearized_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);

    // Add contribution from diagonal hessian block
    hessian_lower_
        .block(key_helper.combined_offset, key_helper.combined_offset, key_helper.tangent_dim,
               key_helper.tangent_dim)
        .template triangularView<Eigen::Lower>() +=
        linearized_factor.hessian.block(key_helper.factor_offset, key_helper.factor_offset,
                                        key_helper.tangent_dim, key_helper.tangent_dim);

    // Add contributions from off-diagonal hessian blocks
    for (int key_j = 0; key_j < key_i; ++key_j) {
      const linearization_key_helper_t& key_helper_j = factor_helper.key_helpers[key_j];
      if (key_helper.combined_offset > key_helper_j.combined_offset) {
        hessian_lower_.block(key_helper.combined_offset, key_helper_j.combined_offset,
                             key_helper.tangent_dim, key_helper_j.tangent_dim) +=
            linearized_factor.hessian.block(key_helper.factor_offset, key_helper_j.factor_offset,
                                            key_helper.tangent_dim, key_helper_j.tangent_dim);
      } else {
        // If key_j is actually after key_i in the full problem, swap indices to put it in the lower
        // triangle
        hessian_lower_.block(key_helper_j.combined_offset, key_helper.combined_offset,
                             key_helper_j.tangent_dim, key_helper.tangent_dim) +=
            linearized_factor.hessian
                .block(key_helper.factor_offset, key_helper_j.factor_offset, key_helper.tangent_dim,
                       key_helper_j.tangent_dim)
                .transpose();
      }
    }
  }
}

template <typename ScalarType>
void Linearization<ScalarType>::UpdateFromLinearizedFactorIntoSparse(
    const LinearizedFactor& linearized_factor, const linearization_factor_helper_t& factor_helper) {
  // Fill in the combined residual slice
  residual_.segment(factor_helper.combined_residual_offset, factor_helper.residual_dim) =
      linearized_factor.residual;

  // For each key
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    const linearization_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

    // Fill in jacobian block, column by column
    for (int col_block = 0; col_block < key_helper.tangent_dim; ++col_block) {
      Eigen::Map<sym::VectorX<Scalar>>(
          jacobian_sparse_.valuePtr() + key_helper.jacobian_storage_col_starts[col_block],
          factor_helper.residual_dim) =
          linearized_factor.jacobian.block(0, key_helper.factor_offset + col_block,
                                           factor_helper.residual_dim, 1);
    }

    // Add contribution from right-hand-side
    rhs_.segment(key_helper.combined_offset, key_helper.tangent_dim) +=
        linearized_factor.rhs.segment(key_helper.factor_offset, key_helper.tangent_dim);

    // Add contribution from diagonal hessian block, column by column
    for (int col_block = 0; col_block < key_helper.tangent_dim; ++col_block) {
      const std::vector<int32_t>& diag_col_starts = key_helper.hessian_storage_col_starts[key_i];
      Eigen::Map<sym::VectorX<Scalar>>(
          hessian_lower_sparse_.valuePtr() + diag_col_starts[col_block],
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
          Eigen::Map<sym::VectorX<Scalar>>(hessian_lower_sparse_.valuePtr() + col_starts[col_j],
                                           key_helper.tangent_dim) +=
              linearized_factor.hessian.block(key_helper.factor_offset,
                                              key_helper_j.factor_offset + col_j,
                                              key_helper.tangent_dim, 1);
        }
      } else {
        for (int32_t col_i = 0; col_i < col_starts.size(); ++col_i) {
          Eigen::Map<sym::VectorX<Scalar>>(hessian_lower_sparse_.valuePtr() + col_starts[col_i],
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
const std::vector<typename Factor<ScalarType>::LinearizedFactor>&
Linearization<ScalarType>::LinearizedFactors() const {
  return linearized_factors_;
}

template <typename ScalarType>
const std::vector<Key>& Linearization<ScalarType>::Keys() const {
  return keys_;
}

template <typename ScalarType>
const VectorX<ScalarType>& Linearization<ScalarType>::Residual() const {
  return residual_;
}

template <typename ScalarType>
const VectorX<ScalarType>& Linearization<ScalarType>::Rhs() const {
  return rhs_;
}

template <typename ScalarType>
const Eigen::SparseMatrix<ScalarType>& Linearization<ScalarType>::JacobianSparse() const {
  return jacobian_sparse_;
}

template <typename ScalarType>
const Eigen::SparseMatrix<ScalarType>& Linearization<ScalarType>::HessianLowerSparse() const {
  return hessian_lower_sparse_;
}

}  // namespace sym

// Explicit instantiation
template class sym::Linearization<double>;
template class sym::Linearization<float>;
