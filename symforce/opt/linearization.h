/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <lcmtypes/sym/sparse_matrix_structure_t.hpp>

#include <sym/util/typedefs.h>

#include "./assert.h"

namespace sym {

/**
 * Class for storing a problem linearization evaluated at a Values (i.e. a residual, jacobian,
 * hessian, and rhs)
 */
template <typename ScalarType>
struct Linearization {
  using Scalar = ScalarType;
  using VectorType = VectorX<Scalar>;
  using MatrixType = Eigen::SparseMatrix<Scalar>;

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

  inline double Error() const {
    SYM_ASSERT(IsInitialized());
    return 0.5 * residual.squaredNorm();
  }

  inline double LinearError(const VectorType& x_update) const {
    SYM_ASSERT(jacobian.cols() == x_update.size());
    const auto linear_residual_new = -jacobian * x_update + residual;
    return 0.5 * linear_residual_new.squaredNorm();
  }

  // Sparse storage
  VectorType residual;
  MatrixType hessian_lower;
  MatrixType jacobian;
  VectorType rhs;

 private:
  bool initialized_{false};
};

// Shorthand instantiations
using Linearizationd = Linearization<double>;
using Linearizationf = Linearization<float>;

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
template <typename Scalar>
sparse_matrix_structure_t GetSparseStructure(const MatrixX<Scalar>&) {
  return {};
}

/**
 * Returns coefficients of matrix. Overloads exist for both dense and sparse matrices
 * to make writing generic code easier.
 * This version returns the non-zero values of an Eigen::SparseMatrix
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
 * Returns a const-ref to the argument.
 */
template <typename Scalar>
const MatrixX<Scalar>& JacobianValues(const MatrixX<Scalar>& matrix) {
  return matrix;
}

}  // namespace sym

// Explicit instantiation declarations
extern template struct sym::Linearization<double>;
extern template struct sym::Linearization<float>;
