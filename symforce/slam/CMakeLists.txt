# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ==============================================================================
# SymForce Targets
# ==============================================================================

file(GLOB_RECURSE SYMFORCE_SLAM_SOURCES CONFIGURE_DEPENDS *.cc **/*.cc)
file(GLOB_RECURSE SYMFORCE_SLAM_HEADERS CONFIGURE_DEPENDS *.h **/*.h *.tcc **/*.tcc)

add_library(
  symforce_slam
  ${SYMFORCE_LIBRARY_TYPE}
  ${SYMFORCE_SLAM_SOURCES}
  ${SYMFORCE_SLAM_HEADERS}
)
target_compile_options(symforce_slam PRIVATE ${SYMFORCE_COMPILE_OPTIONS})
target_link_libraries(symforce_slam
  symforce_gen
  symforce_opt
  ${SYMFORCE_EIGEN_TARGET}
)

install(TARGETS symforce_slam DESTINATION lib)
install(DIRECTORY ./ DESTINATION include/symforce/slam FILES_MATCHING PATTERN "*.h" PATTERN "*.tcc")
