# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# TODO(aaron): Split this into targets per example
file(GLOB_RECURSE SYMFORCE_EXAMPLES_SOURCES CONFIGURE_DEPENDS
  bundle_adjustment/*.cc
  bundle_adjustment_fixed_size/*.cc
  robot_2d_localization/**.cc
  robot_3d_localization/**.cc
)
file(GLOB_RECURSE SYMFORCE_EXAMPLES_HEADERS CONFIGURE_DEPENDS **/*.h **/*.tcc)

add_library(
  symforce_examples
  ${SYMFORCE_LIBRARY_TYPE}
  ${SYMFORCE_EXAMPLES_SOURCES}
  ${SYMFORCE_EXAMPLES_HEADERS}
)
target_compile_options(symforce_examples PRIVATE ${SYMFORCE_COMPILE_OPTIONS})
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

# ------------------------------------------------------------------------------

add_subdirectory(bundle_adjustment_in_the_large)
