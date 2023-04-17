/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./sparse_schur_solver.h"

// Explicit instantiation
template class sym::SparseSchurSolver<Eigen::SparseMatrix<double>>;
template class sym::SparseSchurSolver<Eigen::SparseMatrix<float>>;
