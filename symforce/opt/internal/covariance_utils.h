/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <unordered_map>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <sym/util/typedefs.h>
#include <symforce/opt/linearizer.h>

#include "../sparse_schur_solver.h"

namespace sym {
namespace internal {

/**
 * Get upper-left block of A^-1 using the Schur complement
 *
 * The matrix A and its submatrices are of the form
 *
 *     A = ( B    E )
 *         ( E^T  C )
 *
 * C must be positive semi-definite.  It can be singular, as long as any singular dimensions are
 * rows/columns that are entirely zero.  Other singular C matrices are not supported.  In the
 * future, it would be possible to handle them with a rank-revealing decomposition of C which would
 * likely be more expensive.
 *
 * @param A_lower Sparse symmetric matrix A.  Only the lower triangle is read.
 * @param B_dim size of B
 * @param epsilon small value to clamp the diagonal of C to
 * @tparam Ordering Ordering class to use for sparse cholesky decomposition of C.  Defaults to AMD,
 * which is a good choice for one-off sparse factorization, being fast to compute and giving
 * reasonable quality orderings.
 */
template <class SparseMat, class Ordering = Eigen::AMDOrdering<int>>
[[nodiscard]] Eigen::ComputationInfo ComputeCovarianceBlockWithSchurComplementFromSparseC(
    const SparseMat& A_lower, const int B_dim,
    MatrixX<typename SparseMat::Scalar>& covariance_block,
    const double epsilon = std::numeric_limits<typename SparseMat::Scalar>::epsilon()) {
  // NOTE(aaron): The implementation here is decent, but could probably be better.  At minimum, it
  // doesn't expose the ability to reuse either allocated memory or matrix structure information.
  using Scalar = typename SparseMat::Scalar;

  const int total_dim = A_lower.rows();
  const int C_dim = total_dim - B_dim;

  // ---------- 2. sparse C and E ----------------------------
  SparseMat C = A_lower.block(B_dim, B_dim, C_dim, C_dim);
  SparseMat E_T = A_lower.block(B_dim, 0, C_dim, B_dim);

  // Clamp diagonal to epsilon.  Ideally we'd also check for columns we clamped that the
  // corresponding row is all zeroes, meaning that it's disconnected from the variables in B, and
  // this won't affect the result.  That would be a bit expensive though?
  //
  // This also doesn't handle the case where C is singular, but none of its diagonal entries are 0.
  for (int i = 0; i < C_dim; ++i) {
    if (C.coeff(i, i) <= epsilon) {
      C.coeffRef(i, i) = epsilon;
    }
  }

  // ---------- 3. sparse LDLᵀ of C --------------------------
  Eigen::SimplicialLDLT<SparseMat, Eigen::Lower, Ordering> ldlt;
  ldlt.compute(C);
  if (ldlt.info() != Eigen::Success) {
    return ldlt.info();
  }

  // ---------- 4. Schur complement S = B − E C⁻¹ Eᵀ ---------
  MatrixX<Scalar> S = A_lower.block(0, 0, B_dim, B_dim);  // start with B
  VectorX<Scalar> x;

  for (int j = 0; j < B_dim; ++j) {
    // rhs := j-th column of Eᵀ   (size C)
    const Eigen::SparseVector<Scalar> rhs = E_T.col(j);

    // x  := C⁻¹ rhs              (size C, dense)
    x = ldlt.solve(rhs);

    // update whole column j with E x
    S.col(j) -= E_T.transpose() * x;
  }

  // ---------- 5. tiny dense LDLᵀ to get covariance ---------
  covariance_block.setIdentity(B_dim, B_dim);
  S.ldlt().solveInPlace(covariance_block);

  return Eigen::Success;
}

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
 * @param hessian_lower: The lower triangular portion of the Hessian.  This will be modified in
 * place
 * @param block_dim: The dimension of B
 * @param covariance_block: The matrix in which the result is stored
 * @param c_is_block_diagonal: Whether C is block diagonal, as opposed to a general sparse matrix.
 *   If true, the solver used is specialized for the block structure.
 */
template <typename Scalar>
[[nodiscard]] Eigen::ComputationInfo ComputeCovarianceBlockWithSchurComplement(
    Eigen::SparseMatrix<Scalar>& hessian_lower, const size_t block_dim, const Scalar epsilon,
    sym::MatrixX<Scalar>& covariance_block, const bool c_is_block_diagonal) {
  const int marginalized_dim = hessian_lower.rows() - block_dim;

  // Damp the C portion of the hessian, which is the block we need to invert directly
  hessian_lower.diagonal().tail(marginalized_dim).array() += epsilon;

  // Compute the inverse of the Schur complement
  // TODO(aaron): Cache the solver and the sparsity pattern for this. Similarly this doesn't handle
  // numerical vs symbolic nonzeros, which shouldn't be an issue if we aren't saving the sparsity
  // pattern
  if (c_is_block_diagonal) {
    sym::SparseSchurSolver<Eigen::SparseMatrix<Scalar>> schur_solver{};
    schur_solver.ComputeSymbolicSparsity(hessian_lower, marginalized_dim);
    schur_solver.Factorize(hessian_lower);
    covariance_block = sym::MatrixX<Scalar>::Identity(block_dim, block_dim);
    schur_solver.SInvInPlace(covariance_block);
    return Eigen::Success;
  } else {
    return ComputeCovarianceBlockWithSchurComplementFromSparseC(hessian_lower, block_dim,
                                                                covariance_block);
  }
}

/**
 * Computes the top left square block of the covariance matrix. This is the overload for dense
 * matrices. Does more or less the exact same thing.
 *
 * @param hessian_lower: The lower triangular portion of the Hessian.  This will be modified in
 * place
 * @param block_dim: The dimension of computed block of the covariance matrix
 * @param covariance_block: The matrix in which the result is stored
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
