# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ==============================================================================
# Third Party Dependencies
# ==============================================================================

include(FetchContent)

# ------------------------------------------------------------------------------
# spdlog

find_package(spdlog 1.17...<1.18 QUIET)
if (NOT spdlog_FOUND)
  message(STATUS "spdlog not found, adding with FetchContent")
  function(add_spdlog)
    set(SPDLOG_INSTALL ON CACHE INTERNAL "spdlog should create an install target")
    set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "spdlog shouldn't use its bundled fmtlib")
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    FetchContent_Declare(
      spdlog
      URL https://github.com/gabime/spdlog/archive/v1.17.0.zip
      URL_HASH SHA256=b11912a82d149792fef33fabd0503b13d54aeac25c1464755461d4108ea71fc2
      # spdlog 1.17.0 relies on an fmt::fstring conversion that fmt 12 deprecated. These are
      # upstream spdlog 1685e694 and d5af52d9, which landed after the 1.17.0 release.
      PATCH_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/patch_spdlog.sh"
        "${FETCHCONTENT_BASE_DIR}/spdlog-src"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/spdlog-0001-Fix-deprecated-copy-constructor-usage-of-fmt-format_.patch"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/spdlog-0002-Fix-format_string-propagation-3543.patch"
    )
    FetchContent_MakeAvailable(spdlog)
  endfunction()

  add_spdlog()
else()
  message(STATUS "spdlog found: ${spdlog_VERSION}")
endif()

# ------------------------------------------------------------------------------
# METIS

function(add_metis)
  set(CMAKE_POSITION_INDEPENDENT_CODE True)
  # CMake 4.0 removed compatibility with CMake 3.5: https://cmake.org/cmake/help/v4.0/release/4.0.html#deprecated-and-removed-features
  # See the note below on METIS versioning; newest GKlib at https://github.com/KarypisLab/GKlib does
  # support newer CMake, but we'd need to make sure the memory bugs mentioned below are resolved to
  # upgrade.  The current copy of GKlib only declares it supports CMake 2.8.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  FetchContent_Declare(
    metis
    # METIS does not have releases recently.  Previously were using nearly the initial commit on
    # github, which is newer than the release on the METIS website. All of the releases on github
    # seem to have memory bugs, which do not appear in this release:
    URL https://symforce-org.github.io/downloads/metis-5.1.0.tar.gz
    URL_HASH SHA256=76faebe03f6c963127dbb73c13eab58c9a3faeae48779f049066a21c087c5db2
    # GKlib builds some test executables we can't disable without patching
    PATCH_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/patch_metis.sh" "${FETCHCONTENT_BASE_DIR}/metis-src"
  )

  # Tell metis where to find GKlib
  set(GKLIB_PATH
    "${FETCHCONTENT_BASE_DIR}/metis-src/GKlib" CACHE PATH "Path to GKlib for METIS" FORCE
  )

  set(METIS_LIBRARY_TYPE "SHARED" CACHE STRING "Always build METIS as a shared library" FORCE)

  FetchContent_MakeAvailable(metis)

  # Metis doesn't put its main header (metis.h) on the metis target
  FetchContent_GetProperties(metis SOURCE_DIR metis_SOURCE_DIR)
  target_include_directories(metis INTERFACE ${metis_SOURCE_DIR}/include)
endfunction()

add_metis()

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
  metis
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
