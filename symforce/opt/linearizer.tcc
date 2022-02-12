/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./linearizer.h"

namespace sym {

template <typename ScalarType>
template <typename MatrixType>
void Linearizer<ScalarType>::SplitCovariancesByKey(
    const MatrixType& covariance_block, const std::vector<Key>& keys,
    std::unordered_map<Key, MatrixX<Scalar>>* const covariances_by_key) const {
  SYM_ASSERT(IsInitialized());

  // Fill out the covariance blocks
  for (const auto& key : keys) {
    const auto& entry = state_index_.at(key.GetLcmType());

    (*covariances_by_key)[key] =
        covariance_block.block(entry.offset, entry.offset, entry.tangent_dim, entry.tangent_dim);
  }

  // Make sure we have the expected number of keys
  // If this fails, it might mean that you passed a covariances_by_key that contained keys from a
  // different Linearizer or Optimizer, or previously called with a different subset of the problem
  // keys
  SYM_ASSERT(covariances_by_key->size() == keys.size());
}

}  // namespace sym
