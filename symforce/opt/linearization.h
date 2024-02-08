/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <lcmtypes/sym/sparse_matrix_structure_t.hpp>

#include <sym/util/type_ops.h>
#include <sym/util/typedefs.h>

#include "./assert.h"

namespace sym {

/**
 * Class for storing a problem linearization evaluated at a Values (i.e. a residual, jacobian,
 * hessian, and rhs)
 *
 * MatrixType is expected to be an Eigen MatrixX or SparseMatrix.
 */
template <typename MatrixType>
struct Linearization {
  using Scalar = typename MatrixType::Scalar;
  using Vector = VectorX<Scalar>;
  using Matrix = MatrixType;

  /**
   * Set to invalid
   */
  void Reset() {
    initialized_ = false;
  }

  /**
   * Returns whether the linearization is currently valid for the corresponding values. Accessing
   * any of the members when this is false could result in unexpected behavior
   */
  bool IsInitialized() const {
    return initialized_;
  }

  void SetInitialized(const bool initialized = true) {
    initialized_ = initialized;
  }

  inline Scalar Error() const {
    SYM_ASSERT(IsInitialized());
    return Scalar{0.5} * residual.squaredNorm();
  }

  /**
   * Returns the change in error predicted by the Linearization at the given update
   *
   * @param x_update The update to the values
   * @param damping_vector The vector added to the diagonal of the hessian during the linear solve
   */
  inline Scalar LinearDeltaError(const Vector& x_update, const Vector& damping_vector) const {
    SYM_ASSERT(IsInitialized());
    // See Section 3.2 of "Methods For Non-Linear Least Squares Problems" 2nd Edition.
    // http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
    return Scalar{0.5} * x_update.dot(rhs - damping_vector.cwiseProduct(x_update));
  }

  // Sparse storage
  Vector residual;
  Matrix hessian_lower;
  Matrix jacobian;
  Vector rhs;

 private:
  bool initialized_{false};
};

// Shorthand instantiations
template <typename Scalar>
using SparseLinearization = Linearization<Eigen::SparseMatrix<Scalar>>;
using SparseLinearizationd = SparseLinearization<double>;
using SparseLinearizationf = SparseLinearization<float>;

template <typename Scalar>
using DenseLinearization = Linearization<MatrixX<Scalar>>;
using DenseLinearizationd = DenseLinearization<double>;
using DenseLinearizationf = DenseLinearization<float>;

/**
 * Returns the sparse matrix structure of matrix.
 */
template <typename Scalar>
sparse_matrix_structure_t GetSparseStructure(const Eigen::SparseMatrix<Scalar>& matrix) {
  return {Eigen::Map<const VectorX<typename Eigen::SparseMatrix<Scalar>::StorageIndex>>(
              matrix.innerIndexPtr(), matrix.nonZeros()),
          Eigen::Map<const VectorX<typename Eigen::SparseMatrix<Scalar>::StorageIndex>>(
              matrix.outerIndexPtr(), matrix.outerSize()),
          {matrix.rows(), matrix.cols()}};
}

/**
 * Return a default initialized sparse structure because arg is dense.
 */
template <typename Derived>
sparse_matrix_structure_t GetSparseStructure(const Eigen::EigenBase<Derived>& matrix) {
  return {{}, {}, {matrix.rows(), matrix.cols()}};
}

/**
 * Returns the reconstructed matrix from the stored values and sparse_matrix_structure_t
 *
 * Useful for reconstructing the jacobian for an iteration in the OptimizationStats
 *
 * The result contains references to structure and values, so its lifetime must be shorter than
 * those.
 *
 * @param rows The number of rows in the matrix (must match structure.shape for sparse matrices)
 * @param cols The number of columns in the matrix (must match structure.shape for sparse matrices)
 * @param structure The sparsity pattern of the matrix, as obtained from GetSparseStructure or
 * stored in OptimizationStats::iterations.  For dense matrices, this should be empty.
 * @param values The coefficients of the matrix; flat for sparse matrices, 2d for dense
 * @tparam MatrixType The type of the matrix used to decide if the result is sparse or dense.  This
 * is not otherwise used
 * @tparam Scalar The scalar type of the result (must match the scalar type of values)
 */
template <typename MatrixType, typename Scalar,
          typename std::enable_if_t<kIsSparseEigenType<MatrixType>, bool> = true>
Eigen::Map<const Eigen::SparseMatrix<Scalar>> MatrixViewFromSparseStructure(
    const sparse_matrix_structure_t& structure, const MatrixX<Scalar>& values) {
  SYM_ASSERT_EQ(structure.shape.size(), 2, "Invalid shape for sparse matrix: {}", structure.shape);
  SYM_ASSERT_EQ(structure.column_pointers.size(), structure.shape.at(1));
  SYM_ASSERT_EQ(structure.row_indices.size(), values.rows());
  SYM_ASSERT_EQ(values.cols(), 1);
  return {structure.shape.at(0),        structure.shape.at(1),
          structure.row_indices.size(), structure.column_pointers.data(),
          structure.row_indices.data(), values.data()};
}

/**
 * Returns the reconstructed matrix from the stored values and sparse_matrix_structure_t
 *
 * Useful for reconstructing the jacobian for an iteration in the OptimizationStats
 *
 * The result contains references to structure and values, so its lifetime must be shorter than
 * those.
 *
 * @param rows The number of rows in the matrix (must match structure.shape for sparse matrices)
 * @param cols The number of columns in the matrix (must match structure.shape for sparse matrices)
 * @param structure The sparsity pattern of the matrix, as obtained from GetSparseStructure or
 * stored in OptimizationStats::iterations.  For dense matrices, this should be empty.
 * @param values The coefficients of the matrix; flat for sparse matrices, 2d for dense
 * @tparam MatrixType The type of the matrix used to decide if the result is sparse or dense.  This
 * is not otherwise used
 * @tparam Scalar The scalar type of the result (must match the scalar type of values)
 */
template <typename MatrixType, typename Scalar,
          typename std::enable_if_t<kIsEigenType<MatrixType>, bool> = true>
Eigen::Map<const MatrixX<Scalar>> MatrixViewFromSparseStructure(
    const sparse_matrix_structure_t& structure, const MatrixX<Scalar>& values) {
  SYM_ASSERT_EQ(structure.shape.size(), 2, "Invalid shape for dense matrix: {}", structure.shape);
  SYM_ASSERT_EQ(structure.column_pointers.size(), 0,
                "column_pointers must be empty for dense matrix");
  SYM_ASSERT_EQ(structure.row_indices.size(), 0, "row_indices must be empty for dense matrix");
  SYM_ASSERT_EQ(values.rows(), structure.shape.at(0));
  SYM_ASSERT_EQ(values.cols(), structure.shape.at(1));
  return {values.data(), values.rows(), values.cols()};
}

/**
 * Returns coefficients of matrix. Overloads exist for both dense and sparse matrices
 * to make writing generic code easier.
 *
 * This version returns the non-zero values of an Eigen::SparseMatrix
 *
 * Note: it returns a map, so be careful about mutating or disposing of matrix before
 * you are finished with the output.
 */
template <typename Scalar>
Eigen::Map<const VectorX<Scalar>> JacobianValues(const Eigen::SparseMatrix<Scalar>& matrix) {
  return Eigen::Map<const VectorX<Scalar>>(matrix.valuePtr(), matrix.nonZeros());
}

/**
 * Returns coefficients of matrix. Overloads exist for both dense and sparse matrices
 * to make writing generic code easier.
 *
 * Returns a const-ref to the argument.
 */
template <typename Scalar>
const MatrixX<Scalar>& JacobianValues(const MatrixX<Scalar>& matrix) {
  return matrix;
}

}  // namespace sym

// Explicit instantiation declarations
extern template struct sym::Linearization<Eigen::SparseMatrix<double>>;
extern template struct sym::Linearization<Eigen::SparseMatrix<float>>;
extern template struct sym::Linearization<sym::MatrixX<double>>;
extern template struct sym::Linearization<sym::MatrixX<float>>;
