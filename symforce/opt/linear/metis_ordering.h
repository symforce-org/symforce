#pragma once

#include <metis.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace math {

// Uses the METIS package to compute a permutation for a sparse matrix that increases
// the sparsity of its factorization (reduce fill-in).
// Adapted from Eigen's own adapter around METIS.
template <typename MatrixType>
class MetisGraphOrdering {
 public:
  using Index = Eigen::SparseMatrix<double>::StorageIndex;
  using PermutationType = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index>;
  using IndexVector = Eigen::Matrix<Index, Eigen::Dynamic, 1>;

  void operator()(const MatrixType& A, PermutationType& permutation_matrix);

 protected:
  void ComputeGraph(const MatrixType& A);

  // Pointers to adjacency list of each row/col
  IndexVector index_ptr_;

  // Adjacency list
  IndexVector inner_indices_;
};

}  // namespace math
