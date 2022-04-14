# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

file(GLOB_RECURSE SYMFORCE_EXAMPLES_SOURCES CONFIGURE_DEPENDS **/*.cc)
file(GLOB_RECURSE SYMFORCE_EXAMPLES_HEADERS CONFIGURE_DEPENDS **/*.h **/*.tcc)
add_library(
  symforce_examples
  ${SYMFORCE_LIBRARY_TYPE}
  ${SYMFORCE_EXAMPLES_SOURCES}
  ${SYMFORCE_EXAMPLES_HEADERS}
)
target_link_libraries(
  symforce_examples
  symforce_gen
  symforce_opt
  ${SYMFORCE_EIGEN_TARGET}
)
target_include_directories(
  symforce_examples
  PUBLIC bundle_adjustment_fixed_size/gen/cpp
)