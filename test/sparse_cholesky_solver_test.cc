/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

// Enable Eigen LGPL code only here, for comparison.
#undef EIGEN_MPL2_ONLY

// Required by MetisSupport
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <catch2/catch_test_macros.hpp>

#include <sym/ops/storage_ops.h>
#include <symforce/opt/sparse_cholesky/sparse_cholesky_solver.h>
#include <symforce/opt/tic_toc.h>

using SparseMatrix = Eigen::SparseMatrix<double>;
using DenseVector = Eigen::VectorXd;

Eigen::SparseMatrix<double> MakeRandomSymmetricSparseMatrix(int dim, std::mt19937& gen) {
  Eigen::SparseMatrix<double> mat(dim, dim);

  // Set diagonal to be random
  for (int i = 0; i < dim; ++i) {
    mat.insert(i, i) = sym::Random<double>(gen);
  }

  // Randomly set some values to be nonzero
  for (int i = 0; i < dim * 5; ++i) {
    auto row = std::uniform_int_distribution<>(0, dim - 1)(gen);
    auto col = std::uniform_int_distribution<>(0, dim - 1)(gen);

    mat.insert(row, col) = sym::Random<double>(gen);
  }

  // Make symmetric
  mat = mat.transpose() * mat;

  // Compress the representation
  mat.makeCompressed();

  return mat;
}

template <typename Solver, typename Matrix>
void TestEigenSolver(const std::string& name, Solver& solver, const Matrix& A) {
  SYM_TIME_SCOPE("[{}]", name);
  solver.factorize(A);
  solver.solve(A);
}

TEST_CASE("Test and profile solving a bunch of random sparse systems", "[sparse_cholesky]") {
  // Size of the linear system
  constexpr int dim = 300;

  // Set random seed
  std::mt19937 gen(42);

  const int num_random_problems = 10;
  for (int i = 0; i < num_random_problems; ++i) {
    // Generate random sparse matrix for structure
    const SparseMatrix A = MakeRandomSymmetricSparseMatrix(dim, gen);

    // Create solvers
    sym::SparseCholeskySolver<SparseMatrix> solver_sparse_ac(A);
    Eigen::SparseLU<SparseMatrix> solver_eigen_sparse_lu(A);
    Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper,
                          Eigen::MetisOrdering<SparseMatrix::StorageIndex>>
        solver_eigen_simplicial_ldlt_metis(A);
    Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper,
                          Eigen::AMDOrdering<SparseMatrix::StorageIndex>>
        solver_eigen_simplicial_ldlt_amd(A);

    // Time
    DenseVector x_sparse_ac, x_eigen_sparse_lu, x_eigen_simplicial_ldlt_metis,
        x_eigen_simplicial_ldlt_amd, x_eigen_dense_ldlt;
    for (int i = 0; i < 10; ++i) {
      // Generate random RHS
      const DenseVector b = sym::Random<Eigen::Matrix<double, dim, 1>>(gen);

      // Mess with A by setting the diagonal randomly
      SparseMatrix A_modified = A;
      for (int inx = 0; inx < A_modified.rows(); ++inx) {
        A_modified.coeffRef(inx, inx) = sym::Random<double>(gen);
      }
      const Eigen::MatrixXd A_modified_dense(A_modified);

      {
        SYM_TIME_SCOPE("[solver_sparse_ac]");
        solver_sparse_ac.Factorize(A_modified);
        x_sparse_ac = solver_sparse_ac.Solve(b);
      }

      {
        SYM_TIME_SCOPE("[solver_eigen_sparse_lu]");
        solver_eigen_sparse_lu.factorize(A_modified);
        x_eigen_sparse_lu = solver_eigen_sparse_lu.solve(b);
      }

      {
        SYM_TIME_SCOPE("[solver_eigen_simplicial_ldlt_metis]");
        solver_eigen_simplicial_ldlt_metis.factorize(A_modified);
        x_eigen_simplicial_ldlt_metis = solver_eigen_simplicial_ldlt_metis.solve(b);
      }

      {
        SYM_TIME_SCOPE("[solver_eigen_simplicial_ldlt_amd]");
        solver_eigen_simplicial_ldlt_amd.factorize(A_modified);
        x_eigen_simplicial_ldlt_amd = solver_eigen_simplicial_ldlt_amd.solve(b);
      }

      {
        SYM_TIME_SCOPE("[solver_eigen_dense_ldlt]");
        Eigen::LDLT<Eigen::MatrixXd> solver_eigen_dense_ldlt(A_modified_dense);
        x_eigen_dense_ldlt = solver_eigen_dense_ldlt.solve(b);
      }

      const double tolerance = 1e-5;
      CHECK(x_sparse_ac.isApprox(x_eigen_sparse_lu, tolerance));
      CHECK(x_sparse_ac.isApprox(x_eigen_simplicial_ldlt_metis, tolerance));
      CHECK(x_sparse_ac.isApprox(x_eigen_simplicial_ldlt_amd, tolerance));
      CHECK(x_sparse_ac.isApprox(x_eigen_dense_ldlt, tolerance));
    }
  }
}

TEST_CASE("Make sure solving with a matrix RHS works", "[sparse_cholesky]") {
  // Size of the linear system
  constexpr int dim = 300;

  // Set random seed
  std::mt19937 gen(42);

  const SparseMatrix A = MakeRandomSymmetricSparseMatrix(dim, gen);

  // Create solvers
  sym::SparseCholeskySolver<SparseMatrix> solver_ac(A);
  Eigen::SparseLU<SparseMatrix> solver_eigen(A);

  const Eigen::MatrixXd b = sym::Random<Eigen::Matrix<double, dim, 11>>(gen);

  solver_ac.Factorize(A);
  const Eigen::MatrixXd x_ac = solver_ac.Solve(b);

  solver_eigen.factorize(A);
  const Eigen::MatrixXd x_eigen = solver_eigen.solve(b);

  CHECK(x_ac.rows() == x_eigen.rows());
  CHECK(x_ac.cols() == x_eigen.cols());
  CHECK(x_ac.isApprox(x_eigen, 1e-6));
}

TEST_CASE("Make sure solving in place works", "[sparse_cholesky]") {
  // Size of the linear system
  constexpr int dim = 300;

  // Set random seed
  std::mt19937 gen(42);

  const SparseMatrix A = MakeRandomSymmetricSparseMatrix(dim, gen);

  // Create solvers
  sym::SparseCholeskySolver<SparseMatrix> solver_ac(A);
  Eigen::SparseLU<SparseMatrix> solver_eigen(A);

  const Eigen::MatrixXd b = sym::Random<Eigen::Matrix<double, dim, 11>>(gen);

  solver_ac.Factorize(A);
  Eigen::MatrixXd x_ac = b;
  solver_ac.SolveInPlace(x_ac);

  solver_eigen.factorize(A);
  const Eigen::MatrixXd x_eigen = solver_eigen.solve(b);

  CHECK(x_ac.rows() == x_eigen.rows());
  CHECK(x_ac.cols() == x_eigen.cols());
  CHECK(x_ac.isApprox(x_eigen, 1e-6));
}
