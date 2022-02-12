/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/Sparse>

#include <lcmtypes/sym/linearization_dense_factor_helper_t.hpp>
#include <lcmtypes/sym/linearization_sparse_factor_helper_t.hpp>

#include "../factor.h"
#include "./hash_combine.h"

namespace sym {
namespace internal {

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
    const typename Factor<Scalar>::LinearizedDenseFactor& linearized_factor,
    const CoordsToStorageMap& jacobian_row_col_to_storage_offset,
    const CoordsToStorageMap& hessian_row_col_to_storage_offset,
    linearization_dense_factor_helper_t& factor_helper) {
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); ++key_i) {
    linearization_dense_key_helper_t& key_helper = factor_helper.key_helpers[key_i];

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
      const linearization_dense_key_helper_t& j_key_helper = factor_helper.key_helpers[key_j];
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

template <typename Scalar>
void ComputeKeyHelperSparseMap(
    const typename Factor<Scalar>::LinearizedSparseFactor& linearized_factor,
    const CoordsToStorageMap& jacobian_row_col_to_storage_offset,
    const CoordsToStorageMap& hessian_row_col_to_storage_offset,
    linearization_sparse_factor_helper_t& factor_helper) {
  // We're computing this in UpdateFromSparseOnesFactorIntoTripletLists too...
  std::vector<int> key_for_factor_offset;
  // Reserve?
  for (int key_i = 0; key_i < factor_helper.key_helpers.size(); key_i++) {
    for (int key_offset = 0; key_offset < factor_helper.key_helpers[key_i].tangent_dim;
         key_offset++) {
      key_for_factor_offset.push_back(key_i);
    }
  }

  factor_helper.jacobian_index_map.reserve(linearized_factor.jacobian.nonZeros());
  for (int outer_i = 0; outer_i < linearized_factor.jacobian.outerSize(); ++outer_i) {
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(linearized_factor.jacobian,
                                                                outer_i);
         it; ++it) {
      const auto row = it.row();
      const auto col = it.col();

      const auto& key_helper = factor_helper.key_helpers[key_for_factor_offset[col]];
      const auto problem_col = col - key_helper.factor_offset + key_helper.combined_offset;
      factor_helper.jacobian_index_map.push_back(jacobian_row_col_to_storage_offset.at(
          std::make_pair(row + factor_helper.combined_residual_offset, problem_col)));
    }
  }

  factor_helper.hessian_index_map.reserve(linearized_factor.hessian.nonZeros());
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
      const auto full_problem_indices = problem_row >= problem_col
                                            ? std::make_pair(problem_row, problem_col)
                                            : std::make_pair(problem_col, problem_row);
      factor_helper.hessian_index_map.push_back(
          hessian_row_col_to_storage_offset.at(full_problem_indices));
    }
  }
}

template <typename Scalar, typename LinearizedFactorT, typename FactorHelperT>
FactorHelperT ComputeFactorHelper(const LinearizedFactorT& factor,
                                  const std::unordered_map<key_t, index_entry_t>& state_index,
                                  int& combined_residual_offset) {
  FactorHelperT factor_helper;

  factor_helper.residual_dim = factor.residual.rows();
  factor_helper.combined_residual_offset = combined_residual_offset;

  for (int key_i = 0; key_i < factor.index.entries.size(); ++key_i) {
    const index_entry_t& entry = factor.index.entries[key_i];

    const bool key_is_optimized = state_index.count(entry.key) > 0;
    if (!key_is_optimized) {
      continue;
    }

    factor_helper.key_helpers.emplace_back();
    auto& key_helper = factor_helper.key_helpers.back();

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

}  // namespace internal
}  // namespace sym
