#pragma once

#include <pybind11/pybind11.h>

namespace sym {

void AddOptimizationStatsWrapper(pybind11::module_ module);

}
