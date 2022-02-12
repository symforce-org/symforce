/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <functional>

namespace sym {
namespace internal {

/**
 * Implementation of tuple hashing.
 *
 * Reference:
 *     https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
 */
inline void hash_combine(std::size_t& seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

}  // namespace internal
}  // namespace sym
