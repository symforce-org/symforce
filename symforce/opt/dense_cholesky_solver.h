/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>

namespace sym {

/**
 * A thin wrapper around Eigen::LDLT for use in nonlinear solver classes like
 * sym::LevenbergMarquardtSolver.
 */
template <typename Scalar>
class DenseCholeskySolver {
 public:
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

 private:
  Eigen::LDLT<MatrixType> ldlt_;

 public:
  using RhsType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  /**
   * Decompose A into A = P^T * L * D * L^T * P and store internally.
   * @pre A is a symmetric positive definite matrix.
   * @returns true if factorization succeeded, and false if failed.
   */
  bool Factorize(const MatrixType& A) {
    ldlt_.compute(A);
    return ldlt_.info() == Eigen::Success;
  }

  /**
   * @returns x for A x = b, where x and b are dense
   * @pre this->Factorize has already been called and succeeded.
   */
  template <typename Rhs>
  RhsType Solve(const Eigen::MatrixBase<Rhs>& b) const {
    return ldlt_.solve(b);
  }

  /**
   * Solves in place for x in A x = b, where x and b are dense
   * @pre this->Factorize has already been called and succeeded.
   */
  template <typename Rhs>
  void SolveInPlace(Eigen::MatrixBase<Rhs>& b) const {
    b = ldlt_.solve(b);
  }

  /**
   * @returns the lower triangular matrix L such that P^T * L * D * L^T * P = A, where A is the
   * last matrix to have been factorized with this->Factorize and D is a diagonal matrix
   * with positive diagonal entries, and P is a permutation matrix.
   * @pre this->Factorize has already been called and succeeded.
   */
  auto L() const {
    return ldlt_.matrixL();
  }

  /**
   * @returns the diagonal matrix D such that P^T * L * D * L^T * P = A, where A is the
   * last matrix to have been factorized with this->Factorize, L is lower triangular with
   * unit diagonal, and P is a permutation matrix
   * @pre this->Factorize has already been called and succeeded.
   */
  auto D() const {
    return ldlt_.vectorD();
  }

  /**
   * @returns the permutation matrix P such that P^T * L * D * L^T * P = A, where A is the
   * last matrix to have been factorized with this->Factorize, L is lower triangular with
   * unit diagonal, and D is a diagonal matrix
   * @pre this->Factorize has already been called and succeeded.
   */
  auto Permutation() const {
    return ldlt_.transpositionsP();
  }

  /**
   * Defined to satisfy interface. No analysis is needed so is a no-op.
   */
  template <typename Derived>
  void AnalyzeSparsityPattern(const Eigen::EigenBase<Derived>&) {}
};

}  // namespace sym
