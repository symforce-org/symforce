// Enable Eigen LGPL code only here, for comparison.
#undef EIGEN_MPL2_ONLY

#include <fstream>
#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <Eigen/MetisSupport>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <ac_sparse_math/assert.h>
#include <ac_sparse_math/sparse_cholesky_solver.h>
#include <ac_sparse_math/sparse_schur_solver.h>
#include <gtest/gtest.h>

#include "util/gtest_util/eigen.h"
#include "util/path_util/path_util.h"
#include "util/tic_toc/tic_toc.h"

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
  static const std::string filename =
      path_util::JoinPath(path_util::RootPath(), "third_party_modules", "ac_sparse_math", "test",
                          "data", "schur_test_matrix.txt");
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

  Eigen::SparseMatrix<double> A(rows, cols);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return std::make_pair(landmarks_dim, A);
}

class SparseSchurSolverTest : public ::testing::Test {
 public:
  SparseSchurSolverTest() {
    std::tie(small_landmarks_dim_, small_A_) = BuildSmallMatrix();
    std::tie(large_landmarks_dim_, large_A_) = LoadMatrix();
  }

 protected:
  int small_landmarks_dim_;
  Eigen::SparseMatrix<double> small_A_;
  int large_landmarks_dim_;
  Eigen::SparseMatrix<double> large_A_;

  template <typename Scalar>
  void TestSchur(const Eigen::SparseMatrix<Scalar>& A, const int landmarks_dim) {
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;
    using StorageIndex = typename SparseMatrix::StorageIndex;
    using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    const int total_dim = A.rows();

    const DenseMatrix rhs = DenseMatrix::Random(total_dim, 1);

    // Create solvers
    math::SparseSchurSolver<Eigen::SparseMatrix<Scalar>> solver_schur_ac;
    math::SparseCholeskySolver<SparseMatrix, Eigen::Lower> solver_sparse_ac_natural{
        Eigen::NaturalOrdering<StorageIndex>()};
    math::SparseCholeskySolver<SparseMatrix, Eigen::Lower> solver_sparse_ac_metis{
        Eigen::MetisOrdering<StorageIndex>()};
    math::SparseCholeskySolver<SparseMatrix, Eigen::Lower> solver_sparse_ac_colamd{
        Eigen::COLAMDOrdering<StorageIndex>()};
    math::SparseCholeskySolver<SparseMatrix, Eigen::Lower> solver_sparse_ac_amd{
        Eigen::AMDOrdering<StorageIndex>()};
    Eigen::SparseLU<SparseMatrix> solver_eigen_sparse_lu(A);
    Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower,
                          Eigen::MetisOrdering<typename Eigen::SparseMatrix<Scalar>::StorageIndex>>
        solver_eigen_simplicial_ldlt_metis(A);
    Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower,
                          Eigen::AMDOrdering<typename Eigen::SparseMatrix<Scalar>::StorageIndex>>
        solver_eigen_simplicial_ldlt_amd(A);

    {
      TIC_TOC_SCOPE("[solver_schur_ac_sparsity]");
      solver_schur_ac.ComputeSymbolicSparsity(A, landmarks_dim);
    }

    {
      TIC_TOC_SCOPE("[solver_sparse_ac_natural_sparsity]");
      solver_sparse_ac_natural.ComputeSymbolicSparsity(A);
    }

    {
      TIC_TOC_SCOPE("[solver_sparse_ac_metis_sparsity]");
      solver_sparse_ac_metis.ComputeSymbolicSparsity(A);
    }

    {
      TIC_TOC_SCOPE("[solver_sparse_ac_colamd_sparsity]");
      solver_sparse_ac_colamd.ComputeSymbolicSparsity(A);
    }

    {
      TIC_TOC_SCOPE("[solver_sparse_ac_amd_sparsity]");
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
        TIC_TOC_SCOPE("[solver_schur_ac]");
        solver_schur_ac.Factorize(A_modified);
        x_schur_ac = solver_schur_ac.Solve(rhs);
      }

      if (kIncludeSlowTests) {
        TIC_TOC_SCOPE("[solver_sparse_ac_natural]");
        solver_sparse_ac_natural.Factorize(A_modified);
        x_sparse_ac_natural = solver_sparse_ac_natural.Solve(rhs);
      }

      {
        TIC_TOC_SCOPE("[solver_sparse_ac_metis]");
        solver_sparse_ac_metis.Factorize(A_modified);
        x_sparse_ac_metis = solver_sparse_ac_metis.Solve(rhs);
      }

      if (kIncludeSlowTests) {
        TIC_TOC_SCOPE("[solver_sparse_ac_colamd]");
        solver_sparse_ac_colamd.Factorize(A_modified);
        x_sparse_ac_colamd = solver_sparse_ac_colamd.Solve(rhs);
      }

      {
        TIC_TOC_SCOPE("[solver_sparse_ac_amd]");
        solver_sparse_ac_amd.Factorize(A_modified);
        x_sparse_ac_amd = solver_sparse_ac_amd.Solve(rhs);
      }

      if (kIncludeSlowTests) {
        TIC_TOC_SCOPE("[solver_eigen_sparse_lu]");
        solver_eigen_sparse_lu.factorize(A_modified_symmetric);
        x_eigen_sparse_lu = solver_eigen_sparse_lu.solve(rhs);
      }

      {
        TIC_TOC_SCOPE("[solver_eigen_simplicial_ldlt_metis]");
        solver_eigen_simplicial_ldlt_metis.factorize(A_modified);
        x_eigen_simplicial_ldlt_metis = solver_eigen_simplicial_ldlt_metis.solve(rhs);
      }

      {
        TIC_TOC_SCOPE("[solver_eigen_simplicial_ldlt_amd]");
        solver_eigen_simplicial_ldlt_amd.factorize(A_modified);
        x_eigen_simplicial_ldlt_amd = solver_eigen_simplicial_ldlt_amd.solve(rhs);
      }

      const Scalar tolerance = std::is_same<Scalar, double>::value ? 1e-3 : 1e-1;
      EXPECT_EIGEN_NEAR_RELATIVE(x_schur_ac, x_sparse_ac_metis, tolerance);
      EXPECT_EIGEN_NEAR_RELATIVE(x_schur_ac, x_sparse_ac_amd, tolerance);
      EXPECT_EIGEN_NEAR_RELATIVE(x_schur_ac, x_eigen_simplicial_ldlt_metis, tolerance);
      EXPECT_EIGEN_NEAR_RELATIVE(x_schur_ac, x_eigen_simplicial_ldlt_amd, tolerance);

      if (kIncludeSlowTests) {
        EXPECT_EIGEN_NEAR_RELATIVE(x_schur_ac, x_sparse_ac_natural, tolerance);
        EXPECT_EIGEN_NEAR_RELATIVE(x_schur_ac, x_sparse_ac_colamd, tolerance);
        EXPECT_EIGEN_NEAR_RELATIVE(x_schur_ac, x_eigen_sparse_lu, tolerance);
      }
    }
  }
};

TEST_F(SparseSchurSolverTest, TestSmallFloat) {
  TestSchur<float>(small_A_.cast<float>(), small_landmarks_dim_);
}

TEST_F(SparseSchurSolverTest, TestSmallDouble) {
  TestSchur<double>(small_A_, small_landmarks_dim_);
}

TEST_F(SparseSchurSolverTest, TestLargeFloat) {
  TestSchur<float>(large_A_.cast<float>(), large_landmarks_dim_);
}

TEST_F(SparseSchurSolverTest, TestLargeDouble) {
  TestSchur<double>(large_A_, large_landmarks_dim_);
}
