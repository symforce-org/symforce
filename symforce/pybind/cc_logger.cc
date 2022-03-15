/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_logger.h"

#include <spdlog/spdlog.h>

#include <symforce/opt/internal/logging_configure.h>

namespace py = pybind11;

namespace sym {

void AddLoggerWrapper(pybind11::module_ module) {
  module.def("set_log_level", [](const std::string& log_level) {
    const bool success = internal::SetLogLevel(log_level);
    if (!success) {
      // This gets translated to ValueError
      throw std::invalid_argument(fmt::format("Invalid log level: {}", log_level));
    }
  });
}

}  // namespace sym
