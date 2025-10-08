# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ==============================================================================
# Third Party Dependencies
# ==============================================================================

include(FetchContent)

# NOTE(aaron): cc_sym builds with pybind 2.13.6 and up, but the stubs are generated using pybind
# 3 (and pybind 3.x does produce different stubs from 2.x).  We'll use pybind 2.x if we find it, but
# for CI (or for reproducing the stubs locally), pybind 3.x is required.
find_package(pybind11 2.13.6 QUIET)
if (NOT pybind11_FOUND)
  message(STATUS "pybind11 not found, adding with FetchContent")
  # NOTE(brad): Set PYTHON_EXECUTABLE to ensure pybind11 uses the same
  # python as the rest of symforce.
  set(PYTHON_EXECUTABLE ${SYMFORCE_PYTHON})
  # Disable FindPython.  This causes issues in cibuildwheel when the Python it finds is different
  # from the one we want, and we've specified the python we want above.  We could maybe only disable
  # if SYMFORCE_PYTHON is set?
  set(PYBIND11_FINDPYTHON OFF)
  if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE not set - cannot configure pybind11")
  endif()
  message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v3.0.1.zip
    URL_HASH SHA256=20fb420fe163d0657a262a8decb619b7c3101ea91db35f1a7227e67c426d4c7e
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
