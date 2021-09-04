#include <ac_sparse_math/sparse_cholesky_solver.h>

namespace math {

// Explicit template instantiations
template class SparseCholeskySolver<Eigen::SparseMatrix<double>, Eigen::Upper>;
template class SparseCholeskySolver<Eigen::SparseMatrix<double>, Eigen::Lower>;
template class SparseCholeskySolver<Eigen::SparseMatrix<float>, Eigen::Upper>;
template class SparseCholeskySolver<Eigen::SparseMatrix<float>, Eigen::Lower>;

}  // namespace math
