# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

add_executable(
    bundle_adjustment_in_the_large_example
    bundle_adjustment_in_the_large.cc
)

target_link_libraries(
    bundle_adjustment_in_the_large_example
    symforce_gen
    symforce_opt
    symforce_examples
)

set_target_properties(bundle_adjustment_in_the_large_example
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples
)
