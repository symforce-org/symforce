/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <unordered_map>

#include <Eigen/SparseCore>

#include <sym/util/typedefs.h>
#include <symforce/opt/linearizer.h>

#include "../sparse_schur_solver.h"

namespace sym {
namespace internal {

/**
 * Computes a block of the covariance matrix
 *
 * Requires that the hessian is of the form
 *
 *     A = ( B    E )
 *         ( E^T  C )
 *
 * with C block diagonal.  The computed covariance block is the marginal covariance of the variables
 * corresponding to B
 *
 * Args:
 *     hessian_lower: The lower triangular portion of the Hessian.  This will be modified in place
 *     block_dim: The dimension of B
 *     covariance_block: The matrix in which the result is stored
 */
template <typename Scalar>
void ComputeCovarianceBlockWithSchurComplement(Eigen::SparseMatrix<Scalar>& hessian_lower,
                                               const size_t block_dim, const Scalar epsilon,
                                               sym::MatrixX<Scalar>& covariance_block) {
  const int marginalized_dim = hessian_lower.rows() - block_dim;

  // Damp the C portion of the hessian, which is the block we need to invert directly
  hessian_lower.diagonal().tail(marginalized_dim).array() += epsilon;

  // Compute the inverse of the Schur complement
  // TODO(aaron): Cache the solver and the sparsity pattern for this. Similarly this doesn't handle
  // numerical vs symbolic nonzeros, which shouldn't be an issue if we aren't saving the sparsity
  // pattern
  sym::SparseSchurSolver<Eigen::SparseMatrix<Scalar>> schur_solver{};
  schur_solver.ComputeSymbolicSparsity(hessian_lower, marginalized_dim);
  schur_solver.Factorize(hessian_lower);
  covariance_block = sym::MatrixX<Scalar>::Identity(block_dim, block_dim);
  schur_solver.SInvInPlace(covariance_block);
}

/**
 * Computes the top left square block of the covariance matrix. This is the overload for dense
 * matrices. Does more or less the exact same thing.
 *
 * Args:
 *     hessian_lower: The lower triangular portion of the Hessian.  This will be modified in place
 *     block_dim: The dimension of computed block of the covariance matrix
 *     covariance_block: The matrix in which the result is stored
 **/
template <typename Scalar>
void ComputeCovarianceBlockWithSchurComplement(MatrixX<Scalar>& hessian_lower,
                                               const size_t block_dim, const Scalar epsilon,
                                               sym::MatrixX<Scalar>& covariance_block) {
  // NOTE(brad): If the hessian were of the form:
  // [ A   B ]
  // [ B^T C ],
  // We could instead compute the schur complement A - BC^{-1}Bt, then invert and return that,
  // but it seems only marginally faster than the below, and is more complicated.
  hessian_lower.diagonal().array() += epsilon;

  Eigen::LLT<MatrixX<Scalar>> llt(hessian_lower);
  covariance_block = llt.solve(MatrixX<Scalar>::Identity(hessian_lower.rows(), block_dim))
                         .block(0, 0, block_dim, block_dim);
}

/**
 * Extract covariances for optimized variables individually from the full problem covariance.  For
 * each variable in `keys`, the returned matrix is the corresponding block from the diagonal of
 * the full covariance matrix.  Requires that the Linearizer has already been initialized
 */
template <typename Scalar, typename LinearizerType, typename MatrixType>
void SplitCovariancesByKey(const LinearizerType& linearizer, const MatrixType& covariance_block,
                           const std::vector<Key>& keys,
                           std::unordered_map<Key, MatrixX<Scalar>>& covariances_by_key) {
  SYM_ASSERT(linearizer.IsInitialized());

  // Fill out the covariance blocks
  const auto& state_index = linearizer.StateIndex();
  for (const auto& key : keys) {
    const auto& entry = state_index.at(key.GetLcmType());

    covariances_by_key[key] =
        covariance_block.block(entry.offset, entry.offset, entry.tangent_dim, entry.tangent_dim);
  }

  // Make sure we have the expected number of keys
  // If this fails, it might mean that you passed a covariances_by_key that contained keys from a
  // different Linearizer or Optimizer, or previously called with a different subset of the problem
  // keys
  SYM_ASSERT(covariances_by_key.size() == keys.size());
}

}  // namespace internal
}  // namespace sym
