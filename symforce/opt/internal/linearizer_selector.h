/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include "../dense_linearizer.h"
#include "../linearizer.h"

namespace sym {

namespace internal {

template <typename MatrixType>
struct LinearizerSelector;

template <typename Scalar>
struct LinearizerSelector<Eigen::SparseMatrix<Scalar>> {
  using type = Linearizer<Scalar>;
};

template <typename Scalar>
struct LinearizerSelector<MatrixX<Scalar>> {
  using type = DenseLinearizer<Scalar>;
};

template <typename MatrixType>
using LinearizerSelector_t = typename LinearizerSelector<MatrixType>::type;

}  // namespace internal

}  // namespace sym
