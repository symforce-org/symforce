/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>

#include "./cholesky/sparse_cholesky_solver.h"

namespace sym {

// A solver for factorizing and solving positive definite matrices with a large block-diagonal
// component
//
// This includes matrices typically encountered in SfM, where the landmarks-to-landmarks block of
// the Hessian is block diagonal (there are no cross-terms between different landmarks)
//
// We should have a matrix A of structure
//
// A = ( B    E )
//     ( E^T  C )
//
// with C in R^{C_dim x C_dim} block diagonal.
//
// The full problem is A x = b, which we can break down the same way as
//
//  ( B    E ) ( y ) = ( v )
//  ( E^T  C ) ( z )   ( w )
//
//  We then have two equations:
//
//  B y + E z = v
//  E^T y + C z = w
//
//  C is block diagonal and therefore easy to invert, so we can write
//
//  z = C^{-1} (w - E^T y)
//
//  Plugging this into the first equation, we can eliminate z:
//
//  B y + E C^{-1} (w - E^T y) = v
//  => B y + E C^{-1} w - E C^{-1} E^T y = v
//  => (B - E C^{-1} E^T) y = v - E C^{-1} w
//
// Defining the Schur complement S = B - E C^{-1} E^T, we have
//
// S y = v - E C^{-1} w
//
// So, we can form S and use the above equation to solve for y.  Once we have
// y, we can use the equation for z above to solve for z, and we're done.
//
// See http://ceres-solver.org/nnls_solving.html#dense-schur-sparse-schur
template <typename _MatrixType>
class SparseSchurSolver {
 public:
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;

  static_assert(static_cast<int>(MatrixType::Options) == Eigen::ColMajor,
                "Matrix must be column major");

  using SMatrixSolverType = SparseCholeskySolver<MatrixType, Eigen::Lower>;

  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  SparseSchurSolver(const typename SMatrixSolverType::Ordering& ordering =
                        Eigen::MetisOrdering<typename SMatrixSolverType::StorageIndex>())
      : is_initialized_(false), S_solver_(ordering) {}

  bool IsInitialized() const {
    return is_initialized_;
  }

  // Analyzes A and precomputes/allocates some things (some additional initialization is also done
  // on the first call to Factorize)
  //
  // `A` should be lower triangular
  void ComputeSymbolicSparsity(const MatrixType& A, const int C_dim);

  void Factorize(const MatrixType& A);

  // Solve A x = rhs, return x
  // Requires a call to Factorize(A) first
  template <typename RhsType>
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Solve(
      const Eigen::MatrixBase<RhsType>& rhs) const;

  // Solves in place for S^{-1}, which is equal to the top left block of A^{-1}
  //
  // Args:
  //     x_and_rhs: This matrix stores both the input and output of the function; when the function
  //                is called, this should be set to rhs; when the function returns, the solution x
  //                is stored here
  void SInvInPlace(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* const x_and_rhs) const;

 private:
  bool is_initialized_;

  // Data that depends only on the structure of A, not on the values
  struct SparsityInformation {
    // Information about a single block on the diagonal of C
    struct CBlock {
      int start_idx;
      int dim;
      std::vector<int> col_starts_in_C_inv;
    };

    int total_dim_;
    int B_dim_;
    int C_dim_;
    std::vector<CBlock> C_blocks_;
  };

  // Data that depends on the structure and values in A.  Not cleared on calls to Factorize however,
  // because then we'd have to reallocate and recompute sparsity for S_solver
  struct FactorizationData {
    Eigen::SparseMatrix<Scalar> C_inv_lower;
    Eigen::SparseMatrix<Scalar> E_transpose;
    Eigen::SparseMatrix<Scalar> S_lower;
  };

  SparsityInformation sparsity_information_;
  FactorizationData factorization_data_;
  SMatrixSolverType S_solver_;
};

}  // namespace sym

#include "./sparse_schur_solver.tcc"
