// Enable Eigen LGPL code only here, for comparison.
#undef EIGEN_MPL2_ONLY

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <ac_sparse_math/sparse_cholesky_solver.h>
#include <gtest/gtest.h>

#include "math_shared/rand.h"
#include "util/common/constants.h"
#include "util/common/macros.h"
#include "util/gtest_util/eigen.h"
#include "util/tic_toc/tic_toc.h"

using SparseMatrix = Eigen::SparseMatrix<double>;
using DenseVector = Eigen::VectorXd;

Eigen::SparseMatrix<double> MakeRandomSymmetricSparseMatrix(int dim) {
  Eigen::SparseMatrix<double> mat(dim, dim);

  // Set diagonal to be random
  for (int i = 0; i < dim; ++i) {
    mat.insert(i, i) = math::RandomScalar();
  }

  // Randomly set some values to be nonzero
  for (int i = 0; i < dim * 5; ++i) {
    auto row = math::RandomInteger() % dim;
    auto col = math::RandomInteger() % dim;

    mat.insert(row, col) = math::RandomScalar();
  }

  // Make symmetric
  mat = mat.transpose() * mat;

  // Compress the representation
  mat.makeCompressed();

  return mat;
}

template <typename Solver, typename Matrix>
void TestEigenSolver(const std::string& name, Solver& solver, const Matrix& A) {
  TIC_TOC_SCOPE("[{}]", name);
  solver.factorize(A);
  solver.solve(A);
}

// Test and profile solving a bunch of random sparse systems.
TEST(SparseCholeskySolverTest, TestCholesky) {
  // Size of the linear system
  const int dim = 300;

  // Set random seed
  const int random_seed = 42;
  srand(random_seed);
  math::RandomSetSeed(random_seed);

  const int num_random_problems = 10;
  for (int i = 0; i < num_random_problems; ++i) {
    // Generate random sparse matrix for structure
    const SparseMatrix A = MakeRandomSymmetricSparseMatrix(dim);

    // Create solvers
    math::SparseCholeskySolver<SparseMatrix> solver_sparse_ac(A);
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
      const DenseVector b = math::RandomNormal(dim);

      // Mess with A by setting the diagonal randomly
      SparseMatrix A_modified = A;
      for (int inx = 0; inx < A_modified.rows(); ++inx) {
        A_modified.coeffRef(inx, inx) = math::RandomNormalScalar();
      }
      const Eigen::MatrixXd A_modified_dense(A_modified);

      {
        TIC_TOC_SCOPE("[solver_sparse_ac]");
        solver_sparse_ac.Factorize(A_modified);
        x_sparse_ac = solver_sparse_ac.Solve(b);
      }

      {
        TIC_TOC_SCOPE("[solver_eigen_sparse_lu]");
        solver_eigen_sparse_lu.factorize(A_modified);
        x_eigen_sparse_lu = solver_eigen_sparse_lu.solve(b);
      }

      {
        TIC_TOC_SCOPE("[solver_eigen_simplicial_ldlt_metis]");
        solver_eigen_simplicial_ldlt_metis.factorize(A_modified);
        x_eigen_simplicial_ldlt_metis = solver_eigen_simplicial_ldlt_metis.solve(b);
      }

      {
        TIC_TOC_SCOPE("[solver_eigen_simplicial_ldlt_amd]");
        solver_eigen_simplicial_ldlt_amd.factorize(A_modified);
        x_eigen_simplicial_ldlt_amd = solver_eigen_simplicial_ldlt_amd.solve(b);
      }

      {
        TIC_TOC_SCOPE("[solver_eigen_dense_ldlt]");
        Eigen::LDLT<Eigen::MatrixXd> solver_eigen_dense_ldlt(A_modified_dense);
        x_eigen_dense_ldlt = solver_eigen_dense_ldlt.solve(b);
      }

      const double tolerance = 10 * kMicroTol;
      EXPECT_EIGEN_NEAR_RELATIVE(x_sparse_ac, x_eigen_sparse_lu, tolerance);
      EXPECT_EIGEN_NEAR_RELATIVE(x_sparse_ac, x_eigen_simplicial_ldlt_metis, tolerance);
      EXPECT_EIGEN_NEAR_RELATIVE(x_sparse_ac, x_eigen_simplicial_ldlt_amd, tolerance);
      EXPECT_EIGEN_NEAR_RELATIVE(x_sparse_ac, x_eigen_dense_ldlt, tolerance);
    }
  }
}

// Make sure solving with a matrix RHS works.
TEST(SparseCholeskySolverTest, TestMatrixMatrixSolve) {
  // Size of the linear system
  const int dim = 300;

  // Set random seed
  const int random_seed = 42;
  srand(random_seed);
  math::RandomSetSeed(random_seed);

  const SparseMatrix A = MakeRandomSymmetricSparseMatrix(dim);

  // Create solvers
  math::SparseCholeskySolver<SparseMatrix> solver_ac(A);
  Eigen::SparseLU<SparseMatrix> solver_eigen(A);

  const Eigen::MatrixXd b = Eigen::MatrixXd::Random(A.rows(), 11);

  solver_ac.Factorize(A);
  const Eigen::MatrixXd x_ac = solver_ac.Solve(b);

  solver_eigen.factorize(A);
  const Eigen::MatrixXd x_eigen = solver_eigen.solve(b);

  EXPECT_EQ(x_ac.rows(), x_eigen.rows());
  EXPECT_EQ(x_ac.cols(), x_eigen.cols());
  EXPECT_EIGEN_NEAR_RELATIVE(x_ac, x_eigen, kMicroTol);
}

// Make sure solving in place works.
TEST(SparseCholeskySolverTest, TestSolveInPlace) {
  // Size of the linear system
  const int dim = 300;

  // Set random seed
  const int random_seed = 42;
  srand(random_seed);
  math::RandomSetSeed(random_seed);

  const SparseMatrix A = MakeRandomSymmetricSparseMatrix(dim);

  // Create solvers
  math::SparseCholeskySolver<SparseMatrix> solver_ac(A);
  Eigen::SparseLU<SparseMatrix> solver_eigen(A);

  const Eigen::MatrixXd b = Eigen::MatrixXd::Random(A.rows(), 11);

  solver_ac.Factorize(A);
  Eigen::MatrixXd x_ac = b;
  solver_ac.SolveInPlace(&x_ac);

  solver_eigen.factorize(A);
  const Eigen::MatrixXd x_eigen = solver_eigen.solve(b);

  EXPECT_EQ(x_ac.rows(), x_eigen.rows());
  EXPECT_EQ(x_ac.cols(), x_eigen.cols());
  EXPECT_EIGEN_NEAR_RELATIVE(x_ac, x_eigen, kMicroTol);
}
