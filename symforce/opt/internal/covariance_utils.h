/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Sparse>

#include <sym/util/typedefs.h>

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
void ComputeCovarianceBlockWithSchurComplement(Eigen::SparseMatrix<Scalar>* const hessian_lower,
                                               const size_t block_dim, const Scalar epsilon,
                                               sym::MatrixX<Scalar>* const covariance_block) {
  const int marginalized_dim = hessian_lower->rows() - block_dim;

  // Damp the C portion of the hessian, which is the block we need to invert directly
  hessian_lower->diagonal().tail(marginalized_dim).array() += epsilon;

  // Compute the inverse of the Schur complement
  // TODO(aaron): Cache the solver and the sparsity pattern for this. Similarly this doesn't handle
  // numerical vs symbolic nonzeros, which shouldn't be an issue if we aren't saving the sparsity
  // pattern
  sym::SparseSchurSolver<Eigen::SparseMatrix<Scalar>> schur_solver{};
  schur_solver.ComputeSymbolicSparsity(*hessian_lower, marginalized_dim);
  schur_solver.Factorize(*hessian_lower);
  *covariance_block = sym::MatrixX<Scalar>::Identity(block_dim, block_dim);
  schur_solver.SInvInPlace(covariance_block);
}

}  // namespace internal
}  // namespace sym
