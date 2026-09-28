/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

// Always keep this header.  The specializations it defines are not recognized by IWYU
// IWYU pragma: always_keep

/// Formatter definitions for Eigen types

#include <type_traits>
#include <utility>

#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "./eigen_type_ops.h"

namespace Eigen {
template <typename T>
class DenseBase;
template <typename T>
class MatrixBase;
template <typename T>
class SparseMatrixBase;
template <typename ExpressionType>
class WithFormat;
template <typename VectorType, int Size>
class VectorBlock;
}  // namespace Eigen

template <typename Derived>
struct fmt::formatter<Eigen::DenseBase<Derived>> : ostream_formatter {};

template <typename Derived>
struct fmt::formatter<Derived,
                      std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<Derived>, Derived>, char>>
    : ostream_formatter {};

template <typename Derived>
struct fmt::formatter<Eigen::MatrixBase<Derived>> : ostream_formatter {};

template <typename Derived>
struct fmt::formatter<Eigen::SparseMatrixBase<Derived>> : ostream_formatter {};

template <typename Derived>
struct fmt::formatter<
    Derived, std::enable_if_t<std::is_base_of_v<Eigen::SparseMatrixBase<Derived>, Derived>, char>>
    : ostream_formatter {};

template <typename ExpressionType>
struct fmt::formatter<Eigen::WithFormat<ExpressionType>> : ostream_formatter {};
template <typename VectorType, int Size>
struct fmt::formatter<Eigen::VectorBlock<VectorType, Size>> : ostream_formatter {};

// Eigen 3.4 gave vector expressions STL iterators, so fmt's range formatter is an equally
// specialized candidate against the ostream formatters above.  Opt out of range formatting to leave
// the ostream formatters as the only match.  This also covers subclasses of Eigen types (e.g.
// eigen_lcm types) whose own format_as would otherwise be ambiguous with the range formatter.
template <typename T, typename Char>
struct fmt::range_format_kind<T, Char, std::enable_if_t<sym::kIsDenseEigenType<T>>>
    : std::integral_constant<fmt::range_format, fmt::range_format::disabled> {};
