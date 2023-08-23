/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <ostream>
#include <vector>

#include <fmt/format.h>

#include "./factor.h"
#include "./key.h"

namespace sym {

/**
 * Write the factor graph in .dot format to the given stream
 *
 * The factor graph is represented as `keys` and `factors`.  For example, these can be obtained from
 * an `Optimizer` as `optimizer.Keys()` and `optimizer.Factors()`.
 *
 * @param[in] name The name of the graph (e.g. `optimizer.GetName()`)
 * @param[in] keys The keys in the graph (e.g. `optimizer.Keys()`)
 * @param[in] factors The factors in the graph (e.g. `optimizer.Factors()`)
 * @param[out] out The stream to write the graph to
 */
template <typename Scalar>
void DumpGraph(const std::string& name, const std::vector<Key>& keys,
               const std::vector<Factor<Scalar>>& factors, std::ostream& out) {
  fmt::print(out, "graph \"{}\" {{\n", name);
  for (const auto& key : keys) {
    fmt::print(out, "  {};\n", key);
  }

  for (int i = 0; i < static_cast<int>(factors.size()); i++) {
    const auto& factor = factors[i];
    fmt::print(out, "  factor_{} [shape=point];\n", i);
    for (const auto& key : factor.OptimizedKeys()) {
      fmt::print(out, "  {} -- factor_{};\n", key, i);
    }
  }

  fmt::print(out, "}}\n");
}

extern template void DumpGraph(const std::string& name, const std::vector<Key>& keys,
                               const std::vector<Factor<double>>& factors, std::ostream& out);
extern template void DumpGraph(const std::string& name, const std::vector<Key>& keys,
                               const std::vector<Factor<float>>& factors, std::ostream& out);

}  // namespace sym
