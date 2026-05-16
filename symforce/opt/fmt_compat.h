/* ----------------------------------------------------------------------------
 * fmt 11.x compatibility header for symforce
 *
 * fmt >= 11 no longer auto-discovers operator<< for formatting.  This header
 * restores that behaviour for every class type that provides operator<< but
 * is not already handled by fmt (strings, arithmetic, …).
 *
 * Two C++20 concepts do all the work:
 *
 *  1. EigenDerived  – disables fmt/ranges.h's range formatter for Eigen
 *     expression types (Eigen 3.4+ exposes begin()/end()).
 *
 *  2. The formatter partial specialization – a single, universal catch-all
 *     that delegates to fmt::ostream_formatter for any remaining class type
 *     with operator<<.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <concepts>
#include <ostream>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

// ---------------------------------------------------------------------------
// 1.  Disable fmt's range formatter for Eigen types.
//
//     Eigen 3.4+ types expose begin()/end() which makes fmt/ranges.h treat
//     them as iterable ranges.  We want the human-readable matrix output from
//     operator<< instead, so we mark them as range_format::disabled.
// ---------------------------------------------------------------------------
template <typename T>
concept EigenDerived = requires(const T& t) { t.derived(); };

template <EigenDerived T>
struct fmt::range_format_kind<T, char>
    : std::integral_constant<fmt::range_format, fmt::range_format::disabled> {};

// ---------------------------------------------------------------------------
// 2.  Universal ostream-based formatter for class types with operator<<.
//
//     Guards:
//       • std::is_class_v           – excludes arithmetic, pointers, enums
//                                     that fmt already handles natively.
//       • !convertible to string_view – excludes std::string / string_view
//                                     which have their own fmt formatter.
//       • requires { os << t; }     – the type must actually support operator<<.
// ---------------------------------------------------------------------------
template <typename T>
  requires std::is_class_v<T>
        && (!std::is_convertible_v<const T&, fmt::string_view>)
        && requires(std::ostream& os, const T& t) { os << t; }
struct fmt::formatter<T, char> : fmt::ostream_formatter {};
