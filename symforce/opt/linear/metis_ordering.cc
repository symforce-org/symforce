#include <ac_sparse_math/metis_ordering.h>

#include <iostream>

namespace math {

using namespace Eigen;

template <typename MatrixType>
void MetisGraphOrdering<MatrixType>::ComputeGraph(const MatrixType& A) {
  eigen_assert(A.rows() == A.cols());

  // Get the transpose of the input matrix
  MatrixType At = A.transpose();

  // Get the number of nonzeros elements in each row/col of At+A
  Index total_nonzero = 0;
  IndexVector visited(A.cols());
  visited.setConstant(-1);
  for (int j = 0; j < A.cols(); j++) {
    // Compute the union structure of of A(j,:) and At(j,:)
    // Do not include the diagonal element
    visited(j) = j;

    // Get the nonzeros in row/column j of A
    for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
      // Get the row index (for column major) or column index (for row major)
      Index idx = it.index();
      if (visited(idx) != j) {
        visited(idx) = j;
        ++total_nonzero;
      }
    }
    // Get the nonzeros in row/column j of At
    for (typename MatrixType::InnerIterator it(At, j); it; ++it) {
      Index idx = it.index();
      if (visited(idx) != j) {
        visited(idx) = j;
        ++total_nonzero;
      }
    }
  }

  // Reserve place for A + At
  index_ptr_.resize(A.cols() + 1);
  inner_indices_.resize(total_nonzero);

  // Now compute the real adjacency list of each column/row
  visited.setConstant(-1);
  Index current_nonzero = 0;
  for (int j = 0; j < A.cols(); j++) {
    index_ptr_(j) = current_nonzero;

    // Do not include the diagonal element
    visited(j) = j;

    // Add the pattern of row/column j of A to A+At
    for (typename MatrixType::InnerIterator it(A, j); it; ++it) {
      // Get the row index (for column major) or column index (for row major)
      Index idx = it.index();
      if (visited(idx) != j) {
        visited(idx) = j;
        inner_indices_(current_nonzero) = idx;
        ++current_nonzero;
      }
    }
    // Add the pattern of row/column j of At to A+At
    for (typename MatrixType::InnerIterator it(At, j); it; ++it) {
      Index idx = it.index();
      if (visited(idx) != j) {
        visited(idx) = j;
        inner_indices_(current_nonzero) = idx;
        ++current_nonzero;
      }
    }
  }

  index_ptr_(A.cols()) = current_nonzero;
}

template <typename MatrixType>
void MetisGraphOrdering<MatrixType>::operator()(const MatrixType& A,
                                                PermutationType& permutation_matrix) {
  // First, symmetrize the matrix graph.
  ComputeGraph(A);

  // Invoke the fill-reducing routine from METIS
  // TODO(hayk): Pass options and try to make better.
  int output_error;
  Index cols = A.cols();
  IndexVector perm(A.cols());
  IndexVector iperm(A.cols());
  output_error = METIS_NodeND(&cols, index_ptr_.data(), inner_indices_.data(), NULL, NULL,
                              perm.data(), iperm.data());
  if (output_error != METIS_OK) {
    std::cerr << "Error calling the METIS package." << std::endl;
    return;
  }

  // Get the fill-reducing permutation
  // NOTE:  If Ap is the permuted matrix then perm and iperm vectors are defined as follows
  // Row (column) i of Ap is the perm(i) row(column) of A, and row (column) i of A is the iperm(i)
  // row(column) of Ap

  permutation_matrix.resize(A.cols());
  for (int j = 0; j < A.cols(); j++) {
    permutation_matrix.indices()(iperm(j)) = j;
  }
}

// Explicit template instantiations
template class MetisGraphOrdering<Eigen::SparseMatrix<double>>;
template class MetisGraphOrdering<Eigen::SparseMatrix<float>>;

}  // namespace math
