/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

// Enable Eigen LGPL code only here, for comparison.
#undef EIGEN_MPL2_ONLY

#include <fstream>
#include <random>

// Required by MetisSupport
#include <iostream>

#include <Eigen/Core>
#include <Eigen/MetisSupport>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/sparse_cholesky/sparse_cholesky_solver.h>
#include <symforce/opt/sparse_schur_solver.h>
#include <symforce/opt/tic_toc.h>

// Include factorization methods which are very slow
static constexpr const bool kIncludeSlowTests = false;

std::pair<int, Eigen::SparseMatrix<double>> BuildSmallMatrix() {
  int nonlandmarks_dim = 10;
  int landmarks_dim = 30;
  int total_dim = nonlandmarks_dim + landmarks_dim;
  Eigen::SparseMatrix<double> J(landmarks_dim, total_dim);
  std::vector<Eigen::Triplet<double>> triplets;
  int i = 1;
  for (int col = 0; col < total_dim; col++) {
    for (int row = 0; row < landmarks_dim; row++) {
      if (col < nonlandmarks_dim) {
        // in poses block
        triplets.emplace_back(row, col, i++);
      } else {
        // in landmarks block
        int C_col = col - nonlandmarks_dim;
        if (C_col / 2 == row / 2) {
          // in the landmark-to-landmark block
          triplets.emplace_back(row, col, i++);
        }
      }
    }
  }
  J.setFromTriplets(triplets.begin(), triplets.end());

  const Eigen::SparseMatrix<double> A = (J.transpose() * J).triangularView<Eigen::Lower>();

  return std::make_pair(landmarks_dim, A);
}

std::pair<int, Eigen::SparseMatrix<double>> LoadMatrix() {
#define _SYMFORCE_STRINGIFY(s) #s
#define SYMFORCE_STRINGIFY(s) _SYMFORCE_STRINGIFY(s)
  static const std::string filename =
      std::string(SYMFORCE_STRINGIFY(SYMFORCE_DIR)) + "/test/test_data/schur_test_matrix.txt";
#undef SYMFORCE_STRINGIFY
#undef _SYMFORCE_STRINGIFY

  std::ifstream file(filename);

  int rows, cols, landmarks_dim;
  {
    std::string l;
    std::getline(file, l);
    std::istringstream line_stream(l);
    line_stream >> landmarks_dim;
  }

  {
    std::string l;
    std::getline(file, l);
    std::istringstream line_stream(l);
    line_stream >> rows;
    line_stream >> cols;
  }

  std::vector<Eigen::Triplet<double>> triplets;
  for (std::string l; std::getline(file, l);) {
    std::istringstream line_stream(l);
    int row, col;
    double value;
    line_stream >> row;
    line_stream >> col;
    line_stream >> value;
    triplets.emplace_back(row, col, value);
  }

  CAPTURE(rows, cols, filename);

  Eigen::SparseMatrix<double> A(rows, cols);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return std::make_pair(landmarks_dim, A);
}

template <typename Scalar>
void TestSchur(const Eigen::SparseMatrix<Scalar>& A, const int landmarks_dim) {
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  using StorageIndex = typename SparseMatrix::StorageIndex;
  using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using DenseVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  const int total_dim = A.rows();

  const DenseMatrix rhs = DenseMatrix::Random(total_dim, 1);

  // Create solvers
  sym::SparseSchurSolver<Eigen::SparseMatrix<Scalar>> solver_schur_ac;
  sym::SparseCholeskySolver<SparseMatrix, Eigen::Lower> solver_sparse_ac_natural{
      Eigen::NaturalOrdering<StorageIndex>()};
  sym::SparseCholeskySolver<SparseMatrix, Eigen::Lower> solver_sparse_ac_metis{
      Eigen::MetisOrdering<StorageIndex>()};
  sym::SparseCholeskySolver<SparseMatrix, Eigen::Lower> solver_sparse_ac_colamd{
      Eigen::COLAMDOrdering<StorageIndex>()};
  sym::SparseCholeskySolver<SparseMatrix, Eigen::Lower> solver_sparse_ac_amd{
      Eigen::AMDOrdering<StorageIndex>()};
  Eigen::SparseLU<SparseMatrix> solver_eigen_sparse_lu(A);
  Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower,
                        Eigen::MetisOrdering<typename Eigen::SparseMatrix<Scalar>::StorageIndex>>
      solver_eigen_simplicial_ldlt_metis(A);
  Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower,
                        Eigen::AMDOrdering<typename Eigen::SparseMatrix<Scalar>::StorageIndex>>
      solver_eigen_simplicial_ldlt_amd(A);

  {
    SYM_TIME_SCOPE("[solver_schur_ac_sparsity]");
    solver_schur_ac.ComputeSymbolicSparsity(A, landmarks_dim);
  }

  {
    SYM_TIME_SCOPE("[solver_sparse_ac_natural_sparsity]");
    solver_sparse_ac_natural.ComputeSymbolicSparsity(A);
  }

  {
    SYM_TIME_SCOPE("[solver_sparse_ac_metis_sparsity]");
    solver_sparse_ac_metis.ComputeSymbolicSparsity(A);
  }

  {
    SYM_TIME_SCOPE("[solver_sparse_ac_colamd_sparsity]");
    solver_sparse_ac_colamd.ComputeSymbolicSparsity(A);
  }

  {
    SYM_TIME_SCOPE("[solver_sparse_ac_amd_sparsity]");
    solver_sparse_ac_amd.ComputeSymbolicSparsity(A);
  }

  DenseVector x_schur_ac, x_sparse_ac_natural, x_sparse_ac_metis, x_sparse_ac_colamd,
      x_sparse_ac_amd, x_eigen_sparse_lu, x_eigen_simplicial_ldlt_metis,
      x_eigen_simplicial_ldlt_amd, x_eigen_dense_ldlt;

  std::mt19937 gen(12345);
  std::uniform_real_distribution<Scalar> dist{1, 5};
  for (int i = 0; i < 5; i++) {
    // Mess with A by setting the diagonal randomly
    SparseMatrix A_modified = A;
    for (int inx = 0; inx < A_modified.rows(); ++inx) {
      A_modified.coeffRef(inx, inx) *= dist(gen);
    }
    const SparseMatrix A_modified_symmetric = A_modified.template selfadjointView<Eigen::Lower>();
    const DenseMatrix A_modified_dense(A_modified);

    {
      SYM_TIME_SCOPE("[solver_schur_ac]");
      solver_schur_ac.Factorize(A_modified);
      x_schur_ac = solver_schur_ac.Solve(rhs);
    }

    if (kIncludeSlowTests) {
      SYM_TIME_SCOPE("[solver_sparse_ac_natural]");
      solver_sparse_ac_natural.Factorize(A_modified);
      x_sparse_ac_natural = solver_sparse_ac_natural.Solve(rhs);
    }

    {
      SYM_TIME_SCOPE("[solver_sparse_ac_metis]");
      solver_sparse_ac_metis.Factorize(A_modified);
      x_sparse_ac_metis = solver_sparse_ac_metis.Solve(rhs);
    }

    if (kIncludeSlowTests) {
      SYM_TIME_SCOPE("[solver_sparse_ac_colamd]");
      solver_sparse_ac_colamd.Factorize(A_modified);
      x_sparse_ac_colamd = solver_sparse_ac_colamd.Solve(rhs);
    }

    {
      SYM_TIME_SCOPE("[solver_sparse_ac_amd]");
      solver_sparse_ac_amd.Factorize(A_modified);
      x_sparse_ac_amd = solver_sparse_ac_amd.Solve(rhs);
    }

    if (kIncludeSlowTests) {
      SYM_TIME_SCOPE("[solver_eigen_sparse_lu]");
      solver_eigen_sparse_lu.factorize(A_modified_symmetric);
      x_eigen_sparse_lu = solver_eigen_sparse_lu.solve(rhs);
    }

    {
      SYM_TIME_SCOPE("[solver_eigen_simplicial_ldlt_metis]");
      solver_eigen_simplicial_ldlt_metis.factorize(A_modified);
      x_eigen_simplicial_ldlt_metis = solver_eigen_simplicial_ldlt_metis.solve(rhs);
    }

    {
      SYM_TIME_SCOPE("[solver_eigen_simplicial_ldlt_amd]");
      solver_eigen_simplicial_ldlt_amd.factorize(A_modified);
      x_eigen_simplicial_ldlt_amd = solver_eigen_simplicial_ldlt_amd.solve(rhs);
    }

    const Scalar tolerance = std::is_same<Scalar, double>::value ? 1e-3 : 1e-1;
    CHECK(x_schur_ac.isApprox(x_sparse_ac_metis, tolerance));
    CHECK(x_schur_ac.isApprox(x_sparse_ac_amd, tolerance));
    CHECK(x_schur_ac.isApprox(x_eigen_simplicial_ldlt_metis, tolerance));
    CHECK(x_schur_ac.isApprox(x_eigen_simplicial_ldlt_amd, tolerance));

    if (kIncludeSlowTests) {
      CHECK(x_schur_ac.isApprox(x_sparse_ac_natural, tolerance));
      CHECK(x_schur_ac.isApprox(x_sparse_ac_colamd, tolerance));
      CHECK(x_schur_ac.isApprox(x_eigen_sparse_lu, tolerance));
    }
  }
}

TEMPLATE_TEST_CASE("Test Schur complement with small matrix", "[schur_solver]", float, double) {
  using Scalar = TestType;

  int landmarks_dim;
  Eigen::SparseMatrix<double> A;
  std::tie(landmarks_dim, A) = BuildSmallMatrix();
  TestSchur<Scalar>(A.cast<Scalar>(), landmarks_dim);
}

TEMPLATE_TEST_CASE("Test Schur complement with real matrix", "[schur_solver]", float, double) {
  using Scalar = TestType;

  int landmarks_dim;
  Eigen::SparseMatrix<double> A;
  std::tie(landmarks_dim, A) = LoadMatrix();
  TestSchur<Scalar>(A.cast<Scalar>(), landmarks_dim);
}
