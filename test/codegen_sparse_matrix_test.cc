/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <catch2/catch_test_macros.hpp>

#include <symforce/codegen_sparse_matrix_test/get_diagonal_sparse.h>
#include <symforce/codegen_sparse_matrix_test/get_multiple_dense_and_sparse.h>
#include <symforce/codegen_sparse_matrix_test/update_sparse_mat.h>

const int DIM = 100;

TEST_CASE("Sparse Matrix codegen works", "[codegen_sparse_matrix]") {
  // Create and return sparse matrix
  Eigen::Matrix<double, DIM, DIM> dense_mat = Eigen::Matrix<double, DIM, DIM>::Identity();
  Eigen::SparseMatrix<double> sparse_mat =
      codegen_sparse_matrix_test::GetDiagonalSparse<double>(dense_mat);
  CHECK(sparse_mat.nonZeros() == DIM);
  for (int i = 0; i < DIM; ++i) {
    CHECK(sparse_mat.coeff(i, i) == 1);
  }

  // Create and return multiple sparse and dense matrices
  Eigen::SparseMatrix<double> sparse_mat2;
  Eigen::Matrix4d dense_mat2;
  Eigen::SparseMatrix<double> sparse_mat3;
  const Eigen::Matrix3d dense_mat3 = codegen_sparse_matrix_test::GetMultipleDenseAndSparse(
      dense_mat, &sparse_mat2, &dense_mat2, &sparse_mat3);
  CHECK(sparse_mat2.nonZeros() == DIM);
  for (int i = 0; i < DIM; ++i) {
    CHECK(sparse_mat2.coeff(i, i) == 2);
  }
  CHECK(dense_mat2 == 3 * Eigen::Matrix4d::Identity());
  CHECK(sparse_mat3.nonZeros() == DIM);
  for (int i = 0; i < DIM; ++i) {
    CHECK(sparse_mat3.coeff(i, i) == 4);
  }
  CHECK(dense_mat3 == Eigen::Matrix3d::Zero());

  // If a pre-constructed sparse matrix of the correct dimensions is passed, we should only update
  // its data
  codegen_sparse_matrix_test::UpdateSparseMat<double>(dense_mat, &sparse_mat);
  CHECK(sparse_mat.nonZeros() == DIM);
  for (int i = 0; i < DIM; ++i) {
    CHECK(sparse_mat.coeff(i, i) == 2);
  }

  // If an uninitialized sparse matrix pointer is passed, we should return a sparse matrix of the
  // correct dimensions
  Eigen::SparseMatrix<double> new_sparse_mat;
  codegen_sparse_matrix_test::UpdateSparseMat<double>(dense_mat, &new_sparse_mat);
  CHECK(new_sparse_mat.nonZeros() == DIM);
  for (int i = 0; i < DIM; ++i) {
    CHECK(new_sparse_mat.coeff(i, i) == 2);
  }
}
