/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

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

  Eigen::Map<const VectorX<Scalar>> JacobianValuesMap() const {
    return Eigen::Map<const VectorX<Scalar>>(jacobian.valuePtr(), jacobian.nonZeros());
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

}  // namespace sym
