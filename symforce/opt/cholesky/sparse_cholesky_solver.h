/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the LGPL license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include "../assert.h"

// Needed for Metis
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>

namespace sym {

// Efficiently solves A * x = b, where A is a sparse matrix and b is a dense vector or matrix,
// using the LDLT cholesky factorization A = L * D * L^T, where L is a unit triangular matrix
// and D is a diagonal matrix.
//
// When repeatedly solving systems where A changes but its sparsity pattern remains identical,
// this class can analyze the sparsity pattern once and use it to more efficiently factorize
// and solve on subsequent calls.
template <typename _MatrixType, int _UpLo = Eigen::Lower>
class SparseCholeskySolver {
 public:
  // Save template args for external reference
  using MatrixType = _MatrixType;
  enum { UpLo = _UpLo };

  // Helper types
  using Scalar = typename MatrixType::Scalar;
  using StorageIndex = typename MatrixType::StorageIndex;
  using CholMatrixType = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
  using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using RhsType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using PermutationMatrixType =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, StorageIndex>;
  using Ordering = std::function<void(const MatrixType&, PermutationMatrixType&)>;

 public:
  // Default constructor
  //
  // Args:
  //     ordering: Functor to compute the variable ordering to use.  Can be any functor with
  //         signature void(const MatrixType&, PermutationMatrixType&) which takes in the sparsity
  //         pattern of the matrix A and fills out the permutation of variables to use in the second
  //         argument.  The first argument is the full matrix A, not just the upper or lower
  //         triangle; the values may not be the same as in A, but will be nonzero for entries in A
  //         that are nonzero.  Typically this will be an instance of one of the orderings provided
  //         by Eigen, such as Eigen::NaturalOrdering().
  SparseCholeskySolver(const Ordering& ordering = Eigen::MetisOrdering<StorageIndex>())
      : is_initialized_(false), ordering_(ordering) {}

  // Construct with a representative sparse matrix
  //
  // Args:
  //     A: The matrix to be factorized
  //     ordering: Functor to compute the variable ordering to use.  Can be any functor with
  //         signature void(const MatrixType&, PermutationMatrixType&) which takes in the sparsity
  //         pattern of the matrix A and fills out the permutation of variables to use in the second
  //         argument.  The first argument is the full matrix A, not just the upper or lower
  //         triangle; the values may not be the same as in A, but will be nonzero for entries in A
  //         that are nonzero.  Typically this will be an instance of one of the orderings provided
  //         by Eigen, such as Eigen::NaturalOrdering().
  explicit SparseCholeskySolver(const MatrixType& A,
                                const Ordering& ordering = Eigen::MetisOrdering<StorageIndex>())
      : SparseCholeskySolver(ordering) {
    ComputeSymbolicSparsity(A);
    Factorize(A);
  }

  ~SparseCholeskySolver() {}

  // Whether we have computed a symbolic sparsity and
  // are ready to factorize/solve.
  bool IsInitialized() const {
    return is_initialized_;
  }

  // Compute an efficient permutation matrix (ordering) for A and store internally.
  void ComputePermutationMatrix(const MatrixType& A);

  // Compute symbolic sparsity pattern for A and store internally.
  void ComputeSymbolicSparsity(const MatrixType& A);

  // Decompose A into A = L * D * L^T and store internally.
  // A must have the same sparsity as the matrix used for construction.
  void Factorize(const MatrixType& A);

  // Returns x for A x = b, where x and b are dense
  template <typename Rhs>
  RhsType Solve(const Eigen::MatrixBase<Rhs>& b) const;

  // Solves in place for x in A x = b, where x and b are dense
  template <typename Rhs>
  void SolveInPlace(Eigen::MatrixBase<Rhs>* const b) const;

 protected:
  // Whether we have computed a symbolic sparsity and
  // are ready to factorize/solve.
  bool is_initialized_;

  // The ordering function
  Ordering ordering_;

  // The unit triangular cholesky decomposition L
  // and the diagonal coefficients D
  // These are computed from Factorize()
  // This `L_` here only stores the lower triangular part,
  // we later call `Eigen::UnitLower` to get the actual
  // unit triangular matrix
  CholMatrixType L_;
  VectorType D_;

  // Forward and inverse permutation matrices
  // These are computed from ComputePermutationMatrix()
  PermutationMatrixType permutation_;
  PermutationMatrixType inv_permutation_;

  // Helpers for efficient factorization
  // These are computed from ComputeSymbolicSparsity()
  Eigen::Matrix<StorageIndex, Eigen::Dynamic, 1> parent_;
  Eigen::Matrix<StorageIndex, Eigen::Dynamic, 1> nnz_per_col_;

  // Internal storage for factorization helpers
  CholMatrixType A_permuted_;
  Eigen::Matrix<StorageIndex, Eigen::Dynamic, 1> visited_;
  Eigen::Matrix<StorageIndex, Eigen::Dynamic, 1> L_k_pattern_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> D_agg_;
};

}  // namespace sym

// Include implementation, yay templates.
#define SYM_SPARSE_CHOLESKY_SOLVER_H
#include "./sparse_cholesky_solver.tcc"
