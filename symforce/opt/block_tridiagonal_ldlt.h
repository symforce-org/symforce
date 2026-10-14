/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "./assert.h"

namespace sym {

// LDL^T factorization of a symmetric block tridiagonal matrix
//
//        A_0     C_0^T    0       ...
//   A =  C_0     A_1      C_1^T   ...
//        0       C_1      A_2     ...
//        ...     ...      ...     ...
//
// whose first diagonal block A_0 is kFirstBlock wide and whose remaining kNumBlocks - 1 diagonal
// blocks are kBlock wide. This matches the hessian of a trajectory optimization problem where the
// variables of each timestep only couple to the neighboring timesteps, with possibly fewer free
// variables in the first timestep.
//
// The matrix is given as the lower triangle of an Eigen::SparseMatrix, exactly as
// sym::LevenbergMarquardtSolver hands it to its linear solver, and the class exposes the same
// interface as sym::SparseCholeskySolver so it can be used as the LinearSolverType of the LM
// solver. AnalyzeSparsityPattern() builds a scatter map from the sparse value array into dense
// block storage, so Factorize() only does a gather plus dense block operations, and the sparse
// matrix never needs to be permuted or walked.
//
// The factorization is the plain (unpivoted) LDL^T in natural order, so L and D are the same as
// those of sym::SparseCholeskySolver with Eigen::NaturalOrdering up to rounding.
template <typename ScalarType, int kFirstBlock, int kBlock, int kNumBlocks>
class BlockTridiagonalLdlt {
 public:
  static_assert(kNumBlocks >= 2, "Need at least two diagonal blocks");
  static_assert(kFirstBlock > 0 && kBlock > 0, "Block sizes must be positive");

  using Scalar = ScalarType;
  using MatrixType = Eigen::SparseMatrix<Scalar>;
  using StorageIndex = typename MatrixType::StorageIndex;
  using CholMatrixType = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>;
  using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using PermutationMatrixType =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, StorageIndex>;

  static constexpr int kDim = kFirstBlock + kBlock * (kNumBlocks - 1);

  // A panel holds one diagonal block stacked on top of the sub-diagonal block below it. After
  // factorization the strictly lower triangle of the top square holds the unit lower triangular
  // factor of the diagonal block and the bottom rows hold the corresponding block row of L.
  using FirstPanel = Eigen::Matrix<Scalar, kFirstBlock + kBlock, kFirstBlock>;
  using Panel = Eigen::Matrix<Scalar, 2 * kBlock, kBlock>;
  using DiagVector = Eigen::Matrix<Scalar, kDim, 1>;

  bool IsInitialized() const {
    return is_initialized_;
  }

  // Build the scatter map from the lower triangle of A into dense block storage. Every nonzero
  // must lie on the block tridiagonal band; the values of A are irrelevant.
  void AnalyzeSparsityPattern(const MatrixType& A_lower) {
    SYM_ASSERT_EQ(A_lower.rows(), kDim);
    SYM_ASSERT_EQ(A_lower.cols(), kDim);
    SYM_ASSERT(A_lower.isCompressed(), "BlockTridiagonalLdlt requires a compressed matrix");

    const StorageIndex* const outer = A_lower.outerIndexPtr();
    const StorageIndex* const inner = A_lower.innerIndexPtr();
    scatter_destination_.resize(A_lower.nonZeros());
    for (int col = 0; col < kDim; col++) {
      for (int value_index = outer[col]; value_index < outer[col + 1]; value_index++) {
        const int row = inner[value_index];
        SYM_ASSERT(row >= col, "Only the lower triangle may be stored, found entry ({}, {})", row,
                   col);
        // Get which block this row/col of the overall matrix is in
        const int row_block = BlockOf(row);
        const int col_block = BlockOf(col);
        SYM_ASSERT(row_block == col_block || row_block == col_block + 1,
                   "Entry ({}, {}) lies outside the block tridiagonal band", row, col);
        // Get the row of this entry in the panel (top square + off-diagonal below)
        const int panel_row =
            OffsetInBlock(row) + (row_block == col_block ? 0 : TopSize(col_block));
        // Map this entry in the original sparse matrix nonzeros to an index in the dense panel
        // storage. The dense storage is in column-major, and stores the entries for each panel
        // sequentially.
        scatter_destination_[value_index] =
            PanelOffset(col_block) + OffsetInBlock(col) * PanelRows(col_block) + panel_row;
      }
    }

    storage_.resize(kStorageSize);
    is_initialized_ = true;
  }

  // Factorize A, which must have the same storage layout as the matrix that was analyzed.
  bool Factorize(const MatrixType& A_lower) {
    SYM_ASSERT(is_initialized_);
    SYM_ASSERT(A_lower.isCompressed() &&
                   A_lower.nonZeros() == static_cast<Eigen::Index>(scatter_destination_.size()),
               "Matrix layout differs from the analyzed one");

    // Copy the values of the sparse matrix into the dense storage so that we can do dense
    // factorization on the blocks in place.
    storage_.setZero();
    const Scalar* const values = A_lower.valuePtr();
    Scalar* const input = storage_.data();
    const int* const destination = scatter_destination_.data();
    const int nnz = static_cast<int>(scatter_destination_.size());
    for (int value_index = 0; value_index < nnz; value_index++) {
      input[destination[value_index]] = values[value_index];
    }

    // We want A = L * D * L^T, which are structured as:
    //
    //        A_0   C_0^T   0     ...          L_0   0     0     ...         D_0   0     0    ...
    //   A =  C_0   A_1     C_1^T ...     L =  W_0   L_1   0     ...    D =  0     D_1   0    ...
    //        0     C_1     A_2   ...          0     W_1   L_2   ...         0     0     D_2  ...
    //        ...   ...     ...   ...          ...   ...   ...   ...         ...   ...   ...  ...
    //
    // with each L_t unit lower triangular. Multiplying out L * D * L^T and matching blocks gives
    //
    //   A_t = W_{t-1} D_{t-1} W_{t-1}^T + L_t D_t L_t^T
    //   C_t = W_t D_t L_t^T
    //
    // so going down the block rows, where we initialize S_0 = A_0:
    //
    //   L_t D_t L_t^T = S_t                              (dense LDL^T of S_t)
    //   W_t           = C_t L_t^{-T} D_t^{-1}
    //   S_{t+1}       = A_{t+1} - W_t D_t W_t^T          (Schur complement of blocks 0..t)
    //
    // Each panel starts out as [A_t; C_t] from the copy above. PanelLdlt() turns a panel holding
    // [S_t; C_t] into [L_t; W_t] in place and writes D_t, and SchurUpdate() turns the next panel's
    // A_{t+1} into S_{t+1}. FirstFactor() is the panel of block 0 and Factor(t) that of block t
    // + 1.

    // Block 0: S_0 = A_0, so the first panel can be factored right away
    Eigen::Map<FirstPanel> first_factor = FirstFactor();
    Eigen::Matrix<Scalar, kFirstBlock, 1> d_first;
    PanelLdlt(first_factor, kFirstBlock + kBlock, d_first, d_inv_.template head<kFirstBlock>());

    // Block 1: replace A_1 with S_1 = A_1 - W_0 D_0 W_0^T
    SchurUpdate(first_factor.template bottomRows<kBlock>(), d_first, Factor(0));

    Eigen::Matrix<Scalar, kBlock, 1> d;
    for (int t = 0; t < kNumBlocks - 1; t++) {
      // Factor block t + 1. The last block has no C below it, so only its top square is factored.
      const bool is_last = t == kNumBlocks - 2;
      Eigen::Map<Panel> factor = Factor(t);
      PanelLdlt(factor, is_last ? kBlock : 2 * kBlock, d,
                d_inv_.template segment<kBlock>(kFirstBlock + kBlock * t));

      // Form the Schur complement of block t + 2 from W_{t+1} and D_{t+1}
      if (!is_last) {
        SchurUpdate(factor.template bottomRows<kBlock>(), d, Factor(t + 1));
      }
    }
    return true;
  }

  // Returns x for A x = b, where b is a dense vector or matrix
  template <typename Rhs>
  Eigen::Matrix<Scalar, kDim, Rhs::ColsAtCompileTime> Solve(const Eigen::MatrixBase<Rhs>& b) const {
    Eigen::Matrix<Scalar, kDim, Rhs::ColsAtCompileTime> x = b;
    SolveInPlace(x);
    return x;
  }

  // Solves A x = b in place for a dense vector or matrix.
  template <typename Rhs>
  void SolveInPlace(Eigen::MatrixBase<Rhs>& b) const {
    SYM_ASSERT(is_initialized_);
    SYM_ASSERT_EQ(b.rows(), kDim);

    // With A = L * D * L^T, solve A x = b as L y = b, then D z = y, then L^T x = z, overwriting b
    // with y, z and x in turn. In block form (block rows of b written b_t):
    //
    //   L_0   0     0    ...     y_0       b_0
    //   W_0   L_1   0    ...     y_1   =   b_1       =>   y_t = L_t^{-1} (b_t - W_{t-1} y_{t-1})
    //   0     W_1   L_2  ...     y_2       b_2
    //   ...   ...   ...  ...     ...       ...
    //
    //   L_0^T  W_0^T  0      ...     x_0       z_0
    //   0      L_1^T  W_1^T  ...     x_1   =   z_1   =>   x_t = L_t^{-T} (z_t - W_t^T x_{t+1})
    //   0      0      L_2^T  ...     x_2       z_2
    //   ...    ...    ...    ...     ...       ...
    //
    // so forward substitution runs from the top block row down and back substitution from the
    // bottom block row up.

    // Forward substitution L y = b
    UnitLowerOf(FirstFactor()).solveInPlace(b.template topRows<kFirstBlock>());
    b.template middleRows<kBlock>(kFirstBlock).noalias() -=
        FirstFactor().template bottomRows<kBlock>() * b.template topRows<kFirstBlock>();
    for (int t = 0; t < kNumBlocks - 1; t++) {
      const int start = kFirstBlock + kBlock * t;
      UnitLowerOf(Factor(t)).solveInPlace(b.template middleRows<kBlock>(start));
      if (t < kNumBlocks - 2) {
        b.template middleRows<kBlock>(start + kBlock).noalias() -=
            Factor(t).template bottomRows<kBlock>() * b.template middleRows<kBlock>(start);
      }
    }

    // Diagonal D z = y
    b = d_inv_.asDiagonal() * b;

    // Back substitution L^T x = z
    for (int t = kNumBlocks - 2; t >= 0; t--) {
      const int start = kFirstBlock + kBlock * t;
      if (t < kNumBlocks - 2) {
        b.template middleRows<kBlock>(start).noalias() -=
            Factor(t).template bottomRows<kBlock>().transpose() *
            b.template middleRows<kBlock>(start + kBlock);
      }
      UnitLowerOf(Factor(t)).adjoint().solveInPlace(b.template middleRows<kBlock>(start));
    }
    b.template topRows<kFirstBlock>().noalias() -=
        FirstFactor().template bottomRows<kBlock>().transpose() *
        b.template middleRows<kBlock>(kFirstBlock);
    UnitLowerOf(FirstFactor()).adjoint().solveInPlace(b.template topRows<kFirstBlock>());
  }

  // The strictly lower triangle of the unit lower triangular factor as a sparse matrix. This is
  // assembled on demand for debugging and is not used by Factorize() or Solve().
  //
  // Every entry of the dense blocks is stored, including exact zeros, so the sparsity pattern is
  // the full block band that this solver computes rather than the symbolic pattern of L that
  // sym::SparseCholeskySolver would report.
  CholMatrixType L() const {
    SYM_ASSERT(is_initialized_);
    std::vector<Eigen::Triplet<Scalar, StorageIndex>> triplets;
    const auto add_panel = [&](const auto& panel, const int rows, const int start, const int cols) {
      for (int c = 0; c < cols; c++) {
        for (int r = c + 1; r < rows; r++) {
          triplets.emplace_back(start + r, start + c, panel(r, c));
        }
      }
    };
    add_panel(FirstFactor(), kFirstBlock + kBlock, 0, kFirstBlock);
    for (int t = 0; t < kNumBlocks - 1; t++) {
      add_panel(Factor(t), t == kNumBlocks - 2 ? kBlock : 2 * kBlock, kFirstBlock + kBlock * t,
                kBlock);
    }
    CholMatrixType L(kDim, kDim);
    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
  }

  DiagVector D() const {
    SYM_ASSERT(is_initialized_);
    return d_inv_.cwiseInverse();
  }

  // This solver does not permute the matrix, so return an identity permutation.
  PermutationMatrixType Permutation() const {
    SYM_ASSERT(is_initialized_);
    PermutationMatrixType identity(kDim);
    identity.setIdentity();
    return identity;
  }

 private:
  // Number of entries in the first panel, in every other panel, and in all of storage_.
  static constexpr int kFirstPanelSize = (kFirstBlock + kBlock) * kFirstBlock;
  static constexpr int kPanelSize = 2 * kBlock * kBlock;
  static constexpr int kStorageSize = kFirstPanelSize + kPanelSize * (kNumBlocks - 1);

  // Returns the block index for a given row/column index in the overall matrix
  static constexpr int BlockOf(const int index) {
    return index < kFirstBlock ? 0 : 1 + (index - kFirstBlock) / kBlock;
  }

  // Returns the offset of a given row/column index within its block
  static constexpr int OffsetInBlock(const int index) {
    return index < kFirstBlock ? index : (index - kFirstBlock) % kBlock;
  }

  // Returns the number of rows/cols in the top square of the panel of the given block
  static constexpr int TopSize(const int block) {
    return block == 0 ? kFirstBlock : kBlock;
  }

  // Returns the number of rows in the panel of the given block (top square + off-diagonal)
  static constexpr int PanelRows(const int block) {
    return TopSize(block) + kBlock;
  }

  // Offset of a block's panel in storage_. The last block has no sub-diagonal block, but gets a
  // full panel anyway so the layout stays uniform.
  static constexpr int PanelOffset(const int block) {
    return block == 0 ? 0 : kFirstPanelSize + kPanelSize * (block - 1);
  }

  // The panel of block 0
  Eigen::Map<FirstPanel> FirstFactor() {
    return Eigen::Map<FirstPanel>(storage_.data());
  }
  Eigen::Map<const FirstPanel> FirstFactor() const {
    return Eigen::Map<const FirstPanel>(storage_.data());
  }
  // The panel of block t + 1
  Eigen::Map<Panel> Factor(const int t) {
    return Eigen::Map<Panel>(storage_.data() + PanelOffset(t + 1));
  }
  Eigen::Map<const Panel> Factor(const int t) const {
    return Eigen::Map<const Panel>(storage_.data() + PanelOffset(t + 1));
  }

  template <typename PanelType>
  static auto UnitLowerOf(const PanelType& panel) {
    constexpr int kCols = PanelType::ColsAtCompileTime;
    return panel.template topRows<kCols>().template triangularView<Eigen::UnitLower>();
  }

  // Left-looking LDL^T of the first num_rows rows of a panel [S_t; C_t], where S_t is the Schur
  // complement of the diagonal block. On exit the top square holds L_t (strictly lower part,
  // diagonal implied) and the bottom rows hold W_t = C_t L_t^{-T} D_t^{-1}. D_t and its inverse are
  // written to d and d_inv. Factoring the bottom rows along with the top square works because each
  // row of W_t is computed exactly like a row of L_t.
  //
  // Column j is computed from the previous columns with one matrix-vector product, which is
  // where nearly all of the arithmetic happens; the upper triangle of the top square is never
  // read, so whatever it contains is irrelevant.
  template <typename PanelType>
  static void PanelLdlt(PanelType& panel, const int num_rows,
                        Eigen::Ref<Eigen::Matrix<Scalar, PanelType::ColsAtCompileTime, 1>> d,
                        Eigen::Ref<Eigen::Matrix<Scalar, PanelType::ColsAtCompileTime, 1>> d_inv) {
    constexpr int kCols = PanelType::ColsAtCompileTime;
    Eigen::Matrix<Scalar, kCols, 1> scaled_row;
    for (int j = 0; j < kCols; j++) {
      // Subtract the contributions of the already factored columns 0..j-1,
      //   panel(j:, j) -= panel(j:, 0:j) * D(0:j) * panel(j, 0:j)^T
      // after which panel(j, j) is D(j) and the entries below it are L(j+1:, j) * D(j).
      if (j > 0) {
        scaled_row.head(j) = panel.row(j).head(j).transpose().cwiseProduct(d.head(j));
        panel.col(j).segment(j, num_rows - j).noalias() -=
            panel.block(j, 0, num_rows - j, j) * scaled_row.head(j);
      }
      // Store D(j) and divide it out of the column to get L(j+1:, j)
      d[j] = panel(j, j);
      d_inv[j] = Scalar(1.0) / d[j];
      panel.col(j).segment(j + 1, num_rows - j - 1) *= d_inv[j];
    }
  }

  // Turns the top square of the next panel from A_{t+1} into S_{t+1} = A_{t+1} - W_t D_t W_t^T,
  // where d holds the diagonal of D_t.
  template <typename WType, typename DType>
  static void SchurUpdate(const Eigen::MatrixBase<WType>& w, const Eigen::MatrixBase<DType>& d,
                          Eigen::Map<Panel> next_panel) {
    constexpr int kWidth = WType::ColsAtCompileTime;
    const Eigen::Matrix<Scalar, kBlock, kWidth> w_times_d = w * d.asDiagonal();
    next_panel.template topRows<kBlock>().noalias() -= w_times_d * w.transpose();
  }

  bool is_initialized_ = false;

  // Map from the sparse matrix entries to the indices of the dense storage_
  std::vector<int> scatter_destination_;

  // One panel per block, column major. Factorize() copies the input blocks in and factors them in
  // place, so after factorization this holds the factors.
  VectorType storage_;
  DiagVector d_inv_;
};

}  // namespace sym
