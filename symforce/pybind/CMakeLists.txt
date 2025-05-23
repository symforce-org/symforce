# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ==============================================================================
# Third Party Dependencies
# ==============================================================================

include(FetchContent)

find_package(pybind11 2.13.6 QUIET)
if (NOT pybind11_FOUND)
  message(STATUS "pybind11 not found, adding with FetchContent")
  # NOTE(brad): Set PYTHON_EXECUTABLE to ensure pybind11 uses the same
  # python as the rest of symforce.
  set(PYTHON_EXECUTABLE ${SYMFORCE_PYTHON})
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.13.6.zip
    URL_HASH SHA256=d0a116e91f64a4a2d8fb7590c34242df92258a61ec644b79127951e821b47be6
  )
  FetchContent_MakeAvailable(pybind11)
else()
  message(STATUS "pybind11 found")
endif()

# ==============================================================================
# SymForce Targets
# ==============================================================================

# ------------------------------------------------------------------------------
# cc_sym

pybind11_add_module(
  cc_sym
  SHARED
  cc_factor.cc
  cc_key.cc
  cc_linearization.cc
  cc_logger.cc
  cc_optimization_stats.cc
  cc_optimizer.cc
  cc_slam.cc
  cc_sym.cc
  cc_values.cc
  sym_type_casters.cc
)

target_compile_options(cc_sym PRIVATE ${SYMFORCE_COMPILE_OPTIONS})

if(SYMFORCE_PY_EXTENSION_MODULE_OUTPUT_PATH)
  set_target_properties(cc_sym PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${SYMFORCE_PY_EXTENSION_MODULE_OUTPUT_PATH}
  )
endif()

target_link_libraries(
  cc_sym
  PRIVATE
  symforce_opt
  symforce_gen
  symforce_slam
)

# TODO(aaron): Should there be an install action here?  This is currently installed by setup.py if
# you do `pip install .`
