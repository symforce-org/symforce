# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ==============================================================================
# Third Party Dependencies
# ==============================================================================

include(FetchContent)

# ------------------------------------------------------------------------------
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(metis REQUIRED)

# ==============================================================================
# SymForce Targets
# ==============================================================================

# ------------------------------------------------------------------------------
# symforce_cholesky

file(GLOB SYMFORCE_CHOLESKY_SOURCES CONFIGURE_DEPENDS sparse_cholesky/*.cc)
file(GLOB SYMFORCE_CHOLESKY_HEADERS CONFIGURE_DEPENDS sparse_cholesky/*.h sparse_cholesky/*.tcc)
add_library(
  symforce_cholesky
  ${SYMFORCE_LIBRARY_TYPE}
  ${SYMFORCE_CHOLESKY_SOURCES}
  ${SYMFORCE_CHOLESKY_HEADERS}
)
target_compile_options(symforce_cholesky PRIVATE ${SYMFORCE_COMPILE_OPTIONS})
target_link_libraries(symforce_cholesky
  fmt::fmt
  METIS::METIS
  ${SYMFORCE_EIGEN_TARGET}
)
target_include_directories(symforce_cholesky PUBLIC ../..)

# ------------------------------------------------------------------------------
# symforce_opt

file(GLOB_RECURSE SYMFORCE_OPT_SOURCES CONFIGURE_DEPENDS *.cc **/*.cc)
file(GLOB_RECURSE SYMFORCE_OPT_HEADERS CONFIGURE_DEPENDS *.h **/*.h *.tcc **/*.tcc)
if(SYMFORCE_CUSTOM_TIC_TOCS)
  list(REMOVE_ITEM SYMFORCE_OPT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/internal/tic_toc.cc")
  list(REMOVE_ITEM SYMFORCE_OPT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/internal/tic_toc.h")
endif(SYMFORCE_CUSTOM_TIC_TOCS)
add_library(
  symforce_opt
  ${SYMFORCE_LIBRARY_TYPE}
  ${SYMFORCE_OPT_SOURCES}
  ${SYMFORCE_OPT_HEADERS}
)
target_compile_options(symforce_opt PRIVATE ${SYMFORCE_COMPILE_OPTIONS})
target_link_libraries(symforce_opt
  symforce_gen
  symforce_cholesky
  fmt::fmt
  spdlog::spdlog
  ${SYMFORCE_EIGEN_TARGET}
)

if(SYMFORCE_CUSTOM_TIC_TOCS)
  target_link_libraries(symforce_opt ${SYMFORCE_TIC_TOC_TARGET})
  target_compile_definitions(symforce_opt
    PUBLIC SYMFORCE_TIC_TOC_HEADER=${SYMFORCE_TIC_TOC_HEADER}
    PUBLIC SYM_TIME_SCOPE=${SYMFORCE_TIC_TOC_MACRO}
  )
endif(SYMFORCE_CUSTOM_TIC_TOCS)


# ------------------------------------------------------------------------------
# install

install(TARGETS symforce_cholesky DESTINATION lib)
install(TARGETS symforce_opt DESTINATION lib)
install(DIRECTORY ./ DESTINATION include/symforce/opt FILES_MATCHING PATTERN "*.h" PATTERN "*.tcc")
