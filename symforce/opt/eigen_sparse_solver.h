/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace sym {

/**
 * A thin wrapper around Eigen's Sparse Solver interface for use in nonlinear solver classes like
 * sym::LevenbergMarquardtSolver.
 *
 * Can be specialized with anything satisfying the SparseSolver concept.
 *
 * For example, can be used like:
 *
 *     using LinearSolver =
 *         sym::EigenSparseSolver<double, Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>>>;
 *     using NonlinearSolver = sym::LevenbergMarquardtSolver<double, LinearSolver>;
 *     using Optimizer = sym::Optimizer<double, NonlinearSolver>;
 *
 *     Optimizer optimizer{...};
 */
template <typename Scalar, typename EigenSolver>
class EigenSparseSolver {
 public:
  using MatrixType = Eigen::SparseMatrix<Scalar>;

 private:
  EigenSolver solver_;

 public:
  using RhsType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  /**
   * Factorize A and store internally.
   * @param A a symmetric positive definite matrix.
   * @returns true if factorization succeeded, and false if failed.
   */
  bool Factorize(const MatrixType& A) {
    solver_.compute(A);
    return solver_.info() == Eigen::Success;
  }

  /**
   * @returns x for A x = b, where x and b are dense
   * @pre this->Factorize has already been called and succeeded.
   */
  template <typename Rhs>
  RhsType Solve(const Eigen::MatrixBase<Rhs>& b) const {
    return solver_.solve(b);
  }

  /**
   * Solves in place for x in A x = b, where x and b are dense
   *
   * Eigen solvers cannot actually solve in place, so this solves, then copies back into the
   * argument.
   *
   * @pre this->Factorize has already been called and succeeded.
   */
  template <typename Rhs>
  void SolveInPlace(Eigen::MatrixBase<Rhs>& b) const {
    b = solver_.solve(b);
  }

  /**
   * @returns the lower triangular matrix L such that P^T * L * D * L^T * P = A, where A is the
   * last matrix to have been factorized with this->Factorize and D is a diagonal matrix
   * with positive diagonal entries, and P is a permutation matrix.
   *
   * @pre this->Factorize has already been called and succeeded.
   */
  MatrixType L() const {
    return {};
  }

  /**
   * @returns the diagonal matrix D such that P^T * L * D * L^T * P = A, where A is the
   * last matrix to have been factorized with this->Factorize, L is lower triangular with
   * unit diagonal, and P is a permutation matrix
   * @pre this->Factorize has already been called and succeeded.
   */
  MatrixType D() const {
    return {};
  }

  /**
   * @returns the permutation matrix P such that P^T * L * D * L^T * P = A, where A is the
   * last matrix to have been factorized with this->Factorize, L is lower triangular with
   * unit diagonal, and D is a diagonal matrix
   * @pre this->Factorize has already been called and succeeded.
   */
  Eigen::PermutationMatrix<Eigen::Dynamic> Permutation() const {
    return {};
  }

  /**
   * Defined to satisfy interface. No analysis is needed so is a no-op.
   */
  void AnalyzeSparsityPattern(const MatrixType& A) {
    solver_.analyzePattern(A);
  }
};

}  // namespace sym
