/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the LGPL license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#ifndef SYM_SPARSE_CHOLESKY_SOLVER_H
#error __FILE__ should only be included from sparse_cholesky_solver.h
#endif  // SYM_SPARSE_CHOLESKY_SOLVER_H

namespace sym {

template <typename MatrixType, int UpLo>
void SparseCholeskySolver<MatrixType, UpLo>::ComputePermutationMatrix(const MatrixType& A) {
  SYM_ASSERT(A.rows() == A.cols());

  // Invoke the ordering object
  const MatrixType A_selfadjoint = A.template selfadjointView<UpLo>();
  ordering_(A_selfadjoint, inv_permutation_);

  // "Invert" to get the reverse mapping
  if (inv_permutation_.size() != 0) {
    permutation_ = inv_permutation_.inverse();
  }
}

template <typename MatrixType, int UpLo>
void SparseCholeskySolver<MatrixType, UpLo>::ComputeSymbolicSparsity(const MatrixType& A) {
  SYM_ASSERT(A.rows() == A.cols());

  // Update permutation matrix
  ComputePermutationMatrix(A);

  // Apply permutation matrix (twist A)
  const Eigen::Index N = A.cols();
  A_permuted_.resize(N, N);
  if (permutation_.size() > 0) {
    A_permuted_.template selfadjointView<Eigen::Upper>() =
        A.template selfadjointView<UpLo>().twistedBy(permutation_);
  } else {
    A_permuted_.template selfadjointView<Eigen::Upper>() = A.template selfadjointView<UpLo>();
  }

  // Everything not visited
  visited_.resize(N);
  visited_.setConstant(-1);

  // Set unknown root
  parent_.resize(N);
  parent_.setConstant(-1);

  // Nonzero counts start empty
  nnz_per_col_.resize(N);
  nnz_per_col_.setZero();

  // See Chapter 6:
  // https://www.tau.ac.il/~stoledo/Support/chapter-direct.pdf

  // Iterate through each dim k and for L(k, :), touch all nodes reachable in elimination
  // tree from nonzero entries in A(0:k-1, k)
  for (StorageIndex k = 0; k < N; ++k) {
    // Mark k as visited
    visited_[k] = k;

    for (typename CholMatrixType::InnerIterator it(A_permuted_, k); it; ++it) {
      StorageIndex i = it.index();

      // Skip if not in the upper triangle
      if (i >= k) {
        continue;
      }

      // Follow path from i to root, stop when hit previously visited node
      while (visited_[i] != k) {
        // Set parent
        if (parent_[i] == -1) {
          parent_[i] = k;
        }

        // L(k, i) is nonzero
        nnz_per_col_[i]++;

        // Mark i as visited
        visited_[i] = k;

        // Follow to parent
        i = parent_[i];
      }
    }
  }

  // Allocate memory for cholesky factorization using nonzero counts
  L_.resize(N, N);
  StorageIndex* L_outer = L_.outerIndexPtr();
  L_outer[0] = 0;
  for (StorageIndex k = 0; k < N; ++k) {
    L_outer[k + 1] = L_outer[k] + nnz_per_col_[k];
  }
  L_.resizeNonZeros(L_outer[N]);

  // Allocate other memory used for subsequent factorization and solve calls
  D_.resize(N);
  L_k_pattern_.resize(N);
  D_agg_.resize(N);

  is_initialized_ = true;
}

template <typename MatrixType, int UpLo>
void SparseCholeskySolver<MatrixType, UpLo>::Factorize(const MatrixType& A) {
  const Eigen::Index N = A.rows();

  // Check some invariants
  SYM_ASSERT(N == L_.rows());
  SYM_ASSERT(N == A.cols());

  // Apply twist
  if (permutation_.size() > 0) {
    A_permuted_.template selfadjointView<Eigen::Upper>() =
        A.template selfadjointView<UpLo>().twistedBy(permutation_);
  } else {
    A_permuted_.template selfadjointView<Eigen::Upper>() = A.template selfadjointView<UpLo>();
  }

  // Get the sparse storage arrays. For details see:
  // https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
  const StorageIndex* L_outer = L_.outerIndexPtr();
  StorageIndex* L_inner = L_.innerIndexPtr();
  Scalar* L_value = L_.valuePtr();

  // See "Modified Cholesky Factorization", page 145:
  // http://www.bioinfo.org.cn/~wangchao/maa/Numerical_Optimization.pdf

  // Initialize helpers
  nnz_per_col_.setZero();
  D_agg_.setZero();

  // For each row of L, compute nonzero pattern in topo order
  for (StorageIndex k = 0; k < N; ++k) {
    // Mark k as visited
    visited_[k] = k;

    // Reverse counter
    StorageIndex top_inx = N;

    for (typename CholMatrixType::InnerIterator it(A_permuted_, k); it; ++it) {
      StorageIndex i = it.index();

      if (i > k) {
        continue;
      }

      // Sum A(i, k) into D_agg
      D_agg_[i] += it.value();

      Eigen::Index depth = 0;
      while (visited_[i] != k) {
        // L(k,i) is nonzero
        L_k_pattern_[depth] = i;

        // Mark i as visited
        visited_[i] = k;

        // Follow to parent
        i = parent_[i];

        // Increment depth
        depth += 1;
      }

      // Update pattern
      while (depth > 0) {
        top_inx -= 1;
        depth -= 1;
        L_k_pattern_[top_inx] = L_k_pattern_[depth];
      }
    }

    // Get D(k, k) and clear D_agg(k)
    Scalar D_k = D_agg_[k];
    D_agg_[k] = 0.0;

    // NOTE(hayk): This is a double loop in a loop and is ~O(N^3 / 6)
    for (; top_inx < N; ++top_inx) {
      // L_k_pattern_[top_inx:] is the pattern of L(:, k)
      const Eigen::Index i = L_k_pattern_[top_inx];

      // Compute the nonzero L(k, i)
      const Scalar D_agg_i = D_agg_[i];
      const Scalar L_ki = D_agg_i / D_[i];

      // Get the range for i
      const Eigen::Index ptr_start = L_outer[i];
      const Eigen::Index ptr_end = ptr_start + nnz_per_col_[i];

      // Update D_agg
      Eigen::Index ptr;
      D_agg_[i] = 0.0;
      for (ptr = ptr_start; ptr < ptr_end; ++ptr) {
        D_agg_[L_inner[ptr]] -= L_value[ptr] * D_agg_i;
      }

      // Save L(k, i)
      L_inner[ptr] = k;
      L_value[ptr] = L_ki;

      // Update D(k)
      D_k -= L_ki * D_agg_i;

      // Increment nonzeros in column i
      nnz_per_col_[i] += 1;
    }

    // Save D(k)
    D_[k] = D_k;
  }
}

template <typename MatrixType, int UpLo>
template <typename Rhs>
typename SparseCholeskySolver<MatrixType, UpLo>::RhsType
SparseCholeskySolver<MatrixType, UpLo>::Solve(const Eigen::MatrixBase<Rhs>& b) const {
  RhsType x = b;
  SolveInPlace(&x);
  return x;
}

template <typename MatrixType, int UpLo>
template <typename Rhs>
void SparseCholeskySolver<MatrixType, UpLo>::SolveInPlace(Eigen::MatrixBase<Rhs>* const b) const {
  // Sanity checks
  SYM_ASSERT(is_initialized_);
  SYM_ASSERT(b != nullptr);
  SYM_ASSERT(L_.rows() == b->rows());
  SYM_ASSERT(D_.size() > 0);

  // Pre-computed cholesky decomposition
  const Eigen::TriangularView<const CholMatrixType, Eigen::UnitLower> L(L_);

  Eigen::MatrixBase<Rhs>& x = *b;

  // Twist
  if (permutation_.size() > 0) {
    x = permutation_ * x;
  }

  // A * x = b
  // (L * D * L^T) * x = b
  // x = L^-T * D^-1 * L^-1 * b
  L.solveInPlace(x);
  x = D_.asDiagonal().inverse() * x;
  L.adjoint().solveInPlace(x);

  // Untwist
  if (permutation_.size() > 0) {
    x = inv_permutation_ * x;
  }
}

// Explicit template instantiations
extern template class SparseCholeskySolver<Eigen::SparseMatrix<double>, Eigen::Upper>;
extern template class SparseCholeskySolver<Eigen::SparseMatrix<double>, Eigen::Lower>;
extern template class SparseCholeskySolver<Eigen::SparseMatrix<float>, Eigen::Upper>;
extern template class SparseCholeskySolver<Eigen::SparseMatrix<float>, Eigen::Lower>;

}  // namespace sym
