/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./dump_graph.h"

namespace sym {

template void DumpGraph(const std::string& name, const std::vector<Key>& keys,
                        const std::vector<Factor<double>>& factors, std::ostream& out);
template void DumpGraph(const std::string& name, const std::vector<Key>& keys,
                        const std::vector<Factor<float>>& factors, std::ostream& out);

}  // namespace sym
