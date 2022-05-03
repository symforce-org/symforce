/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include "./assert.h"
#include "./sparse_schur_solver.h"

namespace sym {

template <typename _MatrixType>
void SparseSchurSolver<_MatrixType>::ComputeSymbolicSparsity(const MatrixType& A, const int C_dim) {
  // A must be square
  SYM_ASSERT(A.rows() == A.cols());

  sparsity_information_.total_dim_ = A.rows();
  sparsity_information_.B_dim_ = sparsity_information_.total_dim_ - C_dim;
  sparsity_information_.C_dim_ = C_dim;

  // Iterate over blocks along the diagonal of C
  bool currently_in_block = false;
  int offset_in_C = 0;
  for (int col = sparsity_information_.B_dim_; col < A.outerSize(); ++col) {
    int start_row = -1;
    int prev_row = -1;
    const int col_start_in_C = offset_in_C;

    // Check that C is block diagonal and that the blocks are dense
    // TODO(aaron): Maybe support sparse diagonal blocks?
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(A, col); it; ++it) {
      offset_in_C++;

      if (start_row == -1) {
        start_row = it.row();
      }

      SYM_ASSERT(prev_row == -1 || it.row() == prev_row + 1);
      prev_row = it.row();
    }

    // Make sure that (1) the column wasn't empty and (2) the first nonzero was on the diagonal
    SYM_ASSERT(start_row != -1 && start_row == col);

    const int nonzeros_in_col = prev_row - start_row + 1;

    // Are we currently in the middle of a diagonal block?
    if (currently_in_block) {
      // Make sure we have exactly as many nonzeros in this column as we expect
      const auto& block = sparsity_information_.C_blocks_.back();
      const int block_offset = col - block.start_idx;
      const int expected_nonzeros = block.dim - block_offset;
      SYM_ASSERT(nonzeros_in_col == expected_nonzeros);

      // Are we at the end of the block?
      if (block_offset == block.dim - 1) {
        currently_in_block = false;
      }
    } else {
      // Create the new CBlock
      sparsity_information_.C_blocks_.emplace_back();
      sparsity_information_.C_blocks_.back().start_idx = col;
      sparsity_information_.C_blocks_.back().dim = nonzeros_in_col;

      // If the block had dim 1, we're already out of the block
      if (sparsity_information_.C_blocks_.back().dim > 1) {
        currently_in_block = true;
      }
    }

    sparsity_information_.C_blocks_.back().col_starts_in_C_inv.push_back(col_start_in_C);
  }

  // Allocate C_inv
  std::vector<Eigen::Triplet<Scalar>> triplets;
  for (const auto& block : sparsity_information_.C_blocks_) {
    const int block_offset_in_C = block.start_idx - sparsity_information_.B_dim_;
    for (int block_col = 0; block_col < block.dim; block_col++) {
      for (int block_row = block_col; block_row < block.dim; block_row++) {
        triplets.emplace_back(block_offset_in_C + block_row, block_offset_in_C + block_col, 1);
      }
    }
  }

  Eigen::SparseMatrix<Scalar>& C_inv_lower = factorization_data_.C_inv_lower;
  C_inv_lower = Eigen::SparseMatrix<Scalar>(C_dim, C_dim);
  C_inv_lower.setFromTriplets(triplets.begin(), triplets.end());
}

// TODO(aaron): Record conditioning information here, and have a way for the user to get it
template <typename _MatrixType>
void SparseSchurSolver<_MatrixType>::Factorize(const MatrixType& A) {
  // Compute C_inv
  // NOTE(aaron): Doing this with dense block-wise inversions is faster than a full sparse inversion
  Eigen::SparseMatrix<Scalar>& C_inv_lower = factorization_data_.C_inv_lower;
  for (const typename SparsityInformation::CBlock& block : sparsity_information_.C_blocks_) {
    const MatrixX dense_block = A.block(block.start_idx, block.start_idx, block.dim, block.dim);

    // TODO(aaron): Check conditioning explicitly here
    const MatrixX dense_block_inv =
        dense_block.template selfadjointView<Eigen::Lower>().llt().solve(
            MatrixX::Identity(block.dim, block.dim));

    for (int block_col = 0; block_col < block.dim; block_col++) {
      const int col_size = block.dim - block_col;
      Eigen::Map<VectorX>(C_inv_lower.valuePtr(), C_inv_lower.nonZeros())
          .segment(block.col_starts_in_C_inv[block_col], col_size) =
          dense_block_inv.block(block_col, block_col, col_size, 1);
    }
  }

  Eigen::SparseMatrix<Scalar>& E_transpose = factorization_data_.E_transpose;
  E_transpose = A.block(sparsity_information_.B_dim_, 0, sparsity_information_.C_dim_,
                        sparsity_information_.B_dim_);

  const auto B = A.topLeftCorner(sparsity_information_.B_dim_, sparsity_information_.B_dim_);

  Eigen::SparseMatrix<Scalar>& S_lower = factorization_data_.S_lower;
  S_lower.template selfadjointView<Eigen::Lower>() =
      (B -
       E_transpose.transpose() * C_inv_lower.template selfadjointView<Eigen::Lower>() * E_transpose)
          .template selfadjointView<Eigen::Lower>();

  if (!S_solver_.IsInitialized()) {
    S_solver_.ComputeSymbolicSparsity(S_lower);
  }

  S_solver_.Factorize(S_lower);
}

template <typename _MatrixType>
template <typename RhsType>
Eigen::Matrix<typename _MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>
SparseSchurSolver<_MatrixType>::Solve(const Eigen::MatrixBase<RhsType>& rhs) const {
  const auto v = rhs.topRows(sparsity_information_.B_dim_);
  const auto w = rhs.bottomRows(sparsity_information_.C_dim_);

  const Eigen::SparseMatrix<Scalar>& C_inv_lower = factorization_data_.C_inv_lower;
  const Eigen::SparseMatrix<Scalar>& E_transpose = factorization_data_.E_transpose;

  const auto C_inv = C_inv_lower.template selfadjointView<Eigen::Lower>();
  const auto E = E_transpose.transpose();

  const MatrixX schur_rhs = v - E * C_inv * w;

  const MatrixX y = S_solver_.Solve(schur_rhs);
  const MatrixX z = C_inv * (w - E_transpose * y);

  MatrixX yz(y.rows() + z.rows(), y.cols());
  yz << y, z;

  return yz;
}

template <typename _MatrixType>
void SparseSchurSolver<_MatrixType>::SInvInPlace(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* const x_and_rhs) const {
  *x_and_rhs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(
      sparsity_information_.B_dim_, sparsity_information_.B_dim_);
  S_solver_.SolveInPlace(x_and_rhs);
}

}  // namespace sym
