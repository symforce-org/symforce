# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ==============================================================================
# Third Party Dependencies
# ==============================================================================

include(FetchContent)

# ------------------------------------------------------------------------------
# fmtlib

find_package(fmt QUIET)
if (NOT fmt_FOUND)
  message(STATUS "fmt not found, adding with FetchContent")
  function(add_fmt)
    set(FMT_INSTALL ON CACHE INTERNAL "fmt should create an install target")
    FetchContent_Declare(
      fmtlib
      GIT_REPOSITORY https://github.com/fmtlib/fmt
      GIT_TAG d141cdbeb0fb422a3fb7173b285fd38e0d1772dc # release 8.0.1
      GIT_SHALLOW TRUE
    )
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    FetchContent_MakeAvailable(fmtlib)
  endfunction()

  add_fmt()
else()
  message(STATUS "fmt found")
endif()

# ------------------------------------------------------------------------------
# spdlog

find_package(spdlog 1.9.0 QUIET)
if (NOT spdlog_FOUND)
  message(STATUS "spdlog not found, adding with FetchContent")
  function(add_spdlog)
    set(SPDLOG_INSTALL ON CACHE INTERNAL "spdlog should create an install target")
    set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "spdlog shouldn't use its bundled fmtlib")
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog
      GIT_TAG v1.9.2
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(spdlog)
  endfunction()

  add_spdlog()
else()
  message(STATUS "spdlog found")
endif()

# ------------------------------------------------------------------------------
# tl::optional

find_package(tl-optional QUIET)
if (NOT tl_optional_FOUND)
  message(STATUS "tl::optional not found, adding with FetchContent")
  function(add_tl_optional)
    set(OPTIONAL_ENABLE_TESTS OFF CACHE INTERNAL "tl::optional shouldn't build tests")
    FetchContent_Declare(
      tl_optional
      GIT_REPOSITORY https://github.com/TartanLlama/optional
      GIT_TAG v1.0.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(tl_optional)

    # tl::optional does this correctly for installs, but not direct includes with FetchContent
    add_library(tl::optional ALIAS optional)
  endfunction()

  add_tl_optional()
else()
  message(STATUS "tl::optional found")
endif()

# ------------------------------------------------------------------------------
# METIS

function(add_metis)
  set(CMAKE_POSITION_INDEPENDENT_CODE True)
  FetchContent_Declare(
    metis
    # METIS does not have releases recently.  Previously were using nearly the initial commit on
    # github, which is newer than the release on the METIS website. All of the releases on github
    # seem to have memory bugs, which do not appear in this release:
    URL https://symforce-org.github.io/downloads/metis-5.1.0.tar.gz
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

# NOTE(hayk): For licensing reasons keep this as a shared library and do not link statically
file(GLOB SYMFORCE_CHOLESKY_SOURCES CONFIGURE_DEPENDS cholesky/*.cc)
file(GLOB SYMFORCE_CHOLESKY_HEADERS CONFIGURE_DEPENDS cholesky/*.h cholesky/*.tcc)
add_library(
  symforce_cholesky
  SHARED
  ${SYMFORCE_CHOLESKY_SOURCES}
  ${SYMFORCE_CHOLESKY_HEADERS}
)
target_compile_options(symforce_cholesky PRIVATE ${SYMFORCE_COMPILE_OPTIONS})
target_link_libraries(symforce_cholesky metis ${SYMFORCE_EIGEN_TARGET})
target_include_directories(symforce_cholesky PUBLIC ../..)

# ------------------------------------------------------------------------------
# symforce_opt

file(GLOB_RECURSE SYMFORCE_OPT_SOURCES CONFIGURE_DEPENDS *.cc **/*.cc)
file(GLOB_RECURSE SYMFORCE_OPT_HEADERS CONFIGURE_DEPENDS *.h **/*.h *.tcc **/*.tcc)
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
  tl::optional
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
