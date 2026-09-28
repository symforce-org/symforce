/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

/// Compile-time traits for Eigen types.  This only forward declares Eigen, so it can be used
/// without pulling in Eigen or any of the generated sym types.

#include <type_traits>
#include <utility>

namespace Eigen {
template <typename Derived>
class DenseBase;
template <typename Derived>
class MatrixBase;
template <typename Scalar, int Options, typename StorageIndex>
class SparseMatrix;
}  // namespace Eigen

namespace sym {
namespace internal {

// Each trait overloads on a pointer to the Eigen base, so a type matches if it derives from that
// base for any template arguments.  This also covers subclasses of Eigen types, which derive from
// e.g. MatrixBase<Matrix<...>> rather than MatrixBase<Subclass>.
template <typename Derived>
std::true_type IsEigenTypeImpl(const Eigen::MatrixBase<Derived>*);
std::false_type IsEigenTypeImpl(...);

template <typename Derived>
std::true_type IsDenseEigenTypeImpl(const Eigen::DenseBase<Derived>*);
std::false_type IsDenseEigenTypeImpl(...);

// Only the default options (column-major, int indices).
template <typename Scalar>
std::true_type IsSparseEigenTypeImpl(const Eigen::SparseMatrix<Scalar, 0, int>*);
std::false_type IsSparseEigenTypeImpl(...);

}  // namespace internal

/// True for Eigen matrix and vector types and expressions, not arrays
template <typename T>
static constexpr const bool kIsEigenType =
    decltype(internal::IsEigenTypeImpl(std::declval<T*>()))::value;

/// True for all dense Eigen types and expressions, including arrays
template <typename T>
static constexpr const bool kIsDenseEigenType =
    decltype(internal::IsDenseEigenTypeImpl(std::declval<T*>()))::value;

/// True for Eigen::SparseMatrix<Scalar> and types derived from it
template <typename T>
static constexpr const bool kIsSparseEigenType =
    decltype(internal::IsSparseEigenTypeImpl(std::declval<T*>()))::value;

}  // namespace sym
