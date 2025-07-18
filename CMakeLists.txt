# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# NOTE(aaron): This will use NEW policies up to CMake 4.0; this should
# be the maximum tested CMake version, matching requirements/dev_py38.txt
cmake_minimum_required(VERSION 3.19...4.0)

project(symforce)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# User-Configurable Options
# ==============================================================================

# ------------------------------------------------------------------------------
# Enable/Disable Components

option(SYMFORCE_BUILD_OPT "Build symforce_opt and related targets for optimization" ON)
option(SYMFORCE_BUILD_CC_SYM "Build cc_sym and related targets for python wrapping" ON)
option(SYMFORCE_BUILD_EXAMPLES "Build the examples" ON)
option(SYMFORCE_BUILD_TESTS "Build and run the tests" ON)
option(SYMFORCE_ADD_PYTHON_TESTS "Include the Python tests" ON)
option(SYMFORCE_BUILD_SYMENGINE "Build symengine[py]" ON)
option(SYMFORCE_GENERATE_MANIFEST "Generate build manifest" ON)
option(SYMFORCE_BUILD_BENCHMARKS "Generate examples for alternative libraries" OFF)

# ------------------------------------------------------------------------------
# TicTocs

option(SYMFORCE_CUSTOM_TIC_TOCS
  "Use the user-provided TicTocs instead of the builtin implementation"
  OFF
)
set(SYMFORCE_TIC_TOC_TARGET "" CACHE STRING
  "If SYMFORCE_CUSTOM_TIC_TOCS is on, this should be a target to depend on that provides TicTocs"
)
set(SYMFORCE_TIC_TOC_HEADER "" CACHE STRING
  "If SYMFORCE_CUSTOM_TIC_TOCS is on, this should be the name of a header to include that provides the TicTocs implementation"
)
set(SYMFORCE_TIC_TOC_MACRO "" CACHE STRING
  "If SYMFORCE_CUSTOM_TIC_TOCS is on, this should be the implementation of the SYM_TIME_SCOPE macro (typically another macro, or a function)"
)

# ------------------------------------------------------------------------------
# LCM

option(SYMFORCE_USE_EXTERNAL_LCM
  "Use external LCM and generated bindings instead of generating them ourselves"
  OFF
)

set(SYMFORCE_LCMTYPES_TARGET "" CACHE STRING
  "If SYMFORCE_LCMTYPES_EXTERNAL is on, this should be the name of a target we can depend on to get the already generated LCM types"
)

option(SYMFORCE_SKYMARSHAL_PRINTING
  "Define SKYMARSHAL_PRINTING_ENABLED so that LCM types can be printed in C++"
  ON
)

# ------------------------------------------------------------------------------
# Build Customization

option(SYMFORCE_BUILD_STATIC_LIBRARIES "Build libraries as static" OFF)

set(SYMFORCE_COMPILE_OPTIONS -Wall;-Wextra CACHE STRING
  "Extra flags to pass to the compiler; defaults to enabling warnings"
)

option(SYMFORCE_NO_FAST_MATH "Don't automatically add -ffast-math to some targets" OFF)

set(SYMFORCE_EIGEN_TARGET "Eigen3::Eigen" CACHE STRING
  "Target to depend on for Eigen. Default is to find Eigen with find_package."
)

# ------------------------------------------------------------------------------
# Misc

set(SYMFORCE_PYTHON_OVERRIDE "" CACHE STRING
  "Python executable to use - if empty (the default), find python in the environment"
)
if(SYMFORCE_PYTHON_OVERRIDE STREQUAL "")
  find_program(SYMFORCE_DEFAULT_PYTHON python3 NO_CACHE)
  set(SYMFORCE_PYTHON ${SYMFORCE_DEFAULT_PYTHON})
else()
  set(SYMFORCE_PYTHON ${SYMFORCE_PYTHON_OVERRIDE})
endif()

set(SYMFORCE_SYMENGINE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/symengine_install CACHE STRING
  "Directory to install symengine"
)

set(SYMENGINEPY_INSTALL_ENV "" CACHE STRING "Options for symenginepy install step")

set(SYMFORCE_PY_EXTENSION_MODULE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/pybind CACHE PATH
  "Location to place generated python extension modules"
)

# ==============================================================================
# Setup
# ==============================================================================

# If we're cross-compiling macOS arm64 wheels, set the architecture appropriately
if("$ENV{CIBW_ARCHS_MACOS}" STREQUAL "arm64")
  set(CMAKE_OSX_ARCHITECTURES arm64)
endif()

if(SYMFORCE_BUILD_STATIC_LIBRARIES)
  set(SYMFORCE_LIBRARY_TYPE STATIC)
else()
  set(SYMFORCE_LIBRARY_TYPE SHARED)
endif()

if(SYMFORCE_BUILD_CC_SYM
   OR SYMFORCE_BUILD_EXAMPLES
   OR SYMFORCE_BUILD_TESTS
   OR SYMFORCE_GENERATE_MANIFEST)
  if(NOT SYMFORCE_BUILD_OPT)
    message(FATAL_ERROR
      "Attempting to build targets that depend on symforce_opt, without building symforce_opt"
    )
  endif()
endif()

if(SYMFORCE_BUILD_TESTS AND NOT SYMFORCE_BUILD_EXAMPLES)
  message(FATAL_ERROR "Attempting to build tests without building examples, which is not supported")
endif()

# ==============================================================================
# Third Party Dependencies (needed for build)
# ==============================================================================

# NOTE(aaron): Generally, for third party dependencies we will use already-installed versions if
# available via find_package, and if not pull with FetchContent.  In CMake 3.24, FetchContent can
# do this check automatically, so this logic could be simplified:
# https://www.kitware.com/cmake-3-24-0-is-available-for-download/
# We use URL downloads with FetchContent, because GIT downloads do not have a way to do shallow
# clones with specific commits
include(FetchContent)

# ------------------------------------------------------------------------------
# eigen3

if(SYMFORCE_EIGEN_TARGET STREQUAL "Eigen3::Eigen")
  find_package(Eigen3 QUIET)
  if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, adding with FetchContent")
    function(add_eigen)
      FetchContent_Declare(
        eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
      )

      set(EIGEN_BUILD_DOC OFF CACHE BOOL "Don't build Eigen docs")
      set(BUILD_TESTING OFF CACHE BOOL "Don't build Eigen tests")
      set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Don't build Eigen pkg-config")
      FetchContent_MakeAvailable(eigen3)
    endfunction()

    add_eigen()

    # Enable use of Eigen3_ROOT, which is necessary for sophus to be able to find Eigen when
    # included this way
    # See https://cmake.org/cmake/help/latest/policy/CMP0074.html
    set(CMAKE_POLICY_DEFAULT_CMP0074 NEW CACHE STRING "" FORCE)
    set(Eigen3_ROOT "${FETCHCONTENT_BASE_DIR}/eigen3-src" CACHE PATH "Phooey" FORCE)
  else()
    message(STATUS "Eigen found at ${Eigen3_DIR}")
  endif()
endif()

# ------------------------------------------------------------------------------
# catch2

if(SYMFORCE_BUILD_BENCHMARKS OR SYMFORCE_BUILD_TESTS)
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found, adding with FetchContent")
    function(add_catch)
      FetchContent_Declare(
        Catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.zip
        URL_HASH SHA256=cd175f5b7e62c29558d4c17d2b94325ee0ab6d0bf1a4b3d61bc8dbcc688ea3c2
      )

      FetchContent_MakeAvailable(Catch2)
    endfunction()

    add_catch()
  else()
    message(STATUS "Catch2 found at ${Catch2_DIR}")
  endif()
endif()

# ==============================================================================
# SymForce Targets
# ==============================================================================

if(SYMFORCE_USE_EXTERNAL_LCM)
  add_library(symforce_lcmtypes_cpp INTERFACE)
  target_link_libraries(symforce_lcmtypes_cpp ${SYMFORCE_LCMTYPES_TARGET})
else()
  add_subdirectory(third_party/skymarshal)
  include(third_party/skymarshal/cmake/skymarshal.cmake)

  add_skymarshal_bindings(symforce_lcmtypes
    ${CMAKE_CURRENT_BINARY_DIR}/lcmtypes
    ${CMAKE_CURRENT_SOURCE_DIR}/lcmtypes
  )

  target_include_directories(symforce_lcmtypes_cpp
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen_lcm/lcmtypes/eigen_lcm_lcm/cpp
  )

  add_skymarshal_bindings(eigen_lcm
    ${CMAKE_CURRENT_BINARY_DIR}/lcmtypes
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen_lcm/lcmtypes
    LANGUAGES python
  )

  # Install eigen_lcm headers.  Probably only want this when SYMFORCE_USE_EXTERNAL_LCM==OFF, but not
  # sure
  # TODO(aaron): Move this into skymarshal when we move eigen_lcm into skymarshal
  install(DIRECTORY third_party/eigen_lcm/lcmtypes/eigen_lcm_lcm/cpp/ DESTINATION include)

  add_custom_target(symforce_eigen_lcm_py ALL DEPENDS eigen_lcm_py)

  file(WRITE
    ${CMAKE_CURRENT_BINARY_DIR}/lcmtypes/python2.7/pyproject.toml
    "
[build-system]
requires = ['setuptools', 'setuptools-scm>=8']
build-backend = 'setuptools.build_meta'

[project]
name = 'lcmtypes'
description='lcmtype python bindings (installed by SymForce)'
authors = [{ name = 'Skydio, Inc.', email = 'hayk@skydio.com' }]
license = { text = 'Apache 2.0' }
requires-python = '>=3.5'
dynamic = ['version']

[tool.setuptools_scm]
root = '../../..'
    "
  )
endif()

if(SYMFORCE_SKYMARSHAL_PRINTING)
  target_compile_definitions(symforce_lcmtypes_cpp INTERFACE SKYMARSHAL_PRINTING_ENABLED)
endif()

# ------------------------------------------------------------------------------
# symforce_gen

file(GLOB_RECURSE SYMFORCE_GEN_SOURCES CONFIGURE_DEPENDS gen/cpp/**/*.cc)
file(GLOB_RECURSE SYMFORCE_GEN_HEADERS CONFIGURE_DEPENDS gen/cpp/**/*.tcc gen/cpp/**/*.h)
add_library(
  symforce_gen
  ${SYMFORCE_LIBRARY_TYPE}
  ${SYMFORCE_GEN_SOURCES}
  ${SYMFORCE_GEN_HEADERS}
)
target_compile_options(symforce_gen PRIVATE ${SYMFORCE_COMPILE_OPTIONS})
if(NOT SYMFORCE_NO_FAST_MATH)
  target_compile_options(symforce_gen PRIVATE -ffast-math)
endif()
target_link_libraries(symforce_gen ${SYMFORCE_EIGEN_TARGET} symforce_lcmtypes_cpp)
target_include_directories(
  symforce_gen
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gen/cpp
)

install(TARGETS symforce_gen DESTINATION lib)
install(DIRECTORY gen/cpp/ DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN "*.tcc")

if(SYMFORCE_BUILD_OPT)
  add_subdirectory(symforce/opt)
  add_subdirectory(symforce/slam)
endif()

if(SYMFORCE_BUILD_CC_SYM)
  add_subdirectory(symforce/pybind)
endif()


# ==============================================================================
# Examples, Benchmarks, and Tests
# ==============================================================================

if(SYMFORCE_BUILD_EXAMPLES)
  add_subdirectory(symforce/examples)
endif()

if (SYMFORCE_BUILD_BENCHMARKS)
  add_subdirectory(symforce/benchmarks)
endif()

if(SYMFORCE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)

  get_directory_property(have_parent_scope PARENT_DIRECTORY)
  if(have_parent_scope)
    set(SYMFORCE_CC_TEST_TARGETS "${SYMFORCE_CC_TEST_TARGETS}" PARENT_SCOPE)
  endif()
endif()

# ==============================================================================
# Manifest
# ==============================================================================

# The manifest contains various paths which are known to CMake and used later by other parts of
# SymForce, such as include paths of libraries we compile generated code against, and the paths to
# the lcm-gen executable and symengine.

if(SYMFORCE_GENERATE_MANIFEST)
  function(generate_manifest)
    execute_process(
      COMMAND ${SYMFORCE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_manifest.py
                                --symenginepy_install_dir "${SYMFORCE_SYMENGINE_INSTALL_PREFIX}"
                                --cc_sym_install_dir "${SYMFORCE_PY_EXTENSION_MODULE_OUTPUT_PATH}"
                                --binary_output_dir "${CMAKE_BINARY_DIR}"
                                # TODO(aaron): Put this somewhere smarter
                                --manifest_path ${CMAKE_CURRENT_SOURCE_DIR}/build/manifest.json
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE GENERATE_MANIFEST_OUTPUT
      ERROR_VARIABLE GENERATE_MANIFEST_OUTPUT
    )

    if(STATUS AND NOT STATUS EQUAL 0)
      message(FATAL_ERROR
        "Failed generating manifest with exit code ${STATUS} and output ${GENERATE_MANIFEST_OUTPUT}"
      )
    endif()
  endfunction()

  generate_manifest()
endif()

# ==============================================================================
# Third Party Dependencies (not needed for build)
# ==============================================================================

if(SYMFORCE_BUILD_SYMENGINE)

  # ------------------------------------------------------------------------------
  # SymEngine
  string(REPLACE ";" "$<SEMICOLON>" EXTERNAL_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
  include(ExternalProject)
  ExternalProject_Add(symengine
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/symengine
    INSTALL_DIR ${SYMFORCE_SYMENGINE_INSTALL_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DBUILD_TESTS=OFF
               -DBUILD_BENCHMARKS=OFF
               -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
               -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
               -DCMAKE_PREFIX_PATH=${EXTERNAL_PREFIX_PATH}
               -DWITH_COTIRE=OFF
               -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  )

  # ------------------------------------------------------------------------------
  # symenginepy

  ExternalProject_Add(symenginepy
    DEPENDS symengine
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/symenginepy
    CONFIGURE_COMMAND ""
    # TODO(aaron): This depends on SYMFORCE_PYTHON, which can change between builds (e.g. when
    # running `python -m build` multiple times on the same build directory), currently if you're
    # doing that you need to clean this manually
    BUILD_COMMAND ${SYMFORCE_PYTHON} ${PROJECT_SOURCE_DIR}/third_party/symenginepy/setup.py
        --verbose
        build_ext
        --build-type=Release
        --symengine-dir ${SYMFORCE_SYMENGINE_INSTALL_PREFIX}/lib/
        --define CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
        --define CMAKE_POLICY_VERSION_MINIMUM=3.5
    INSTALL_COMMAND env ${SYMENGINEPY_INSTALL_ENV} ${SYMFORCE_PYTHON}
        ${PROJECT_SOURCE_DIR}/third_party/symenginepy/setup.py
        --verbose
        install
        --prefix ${SYMFORCE_SYMENGINE_INSTALL_PREFIX}
        --define CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
        --define CMAKE_POLICY_VERSION_MINIMUM=3.5
  )

  # Rebuild symengine[py] on changed files
  function(add_sourcechanged_dependency project_name)
    set(cmake_stampdir ${CMAKE_CURRENT_BINARY_DIR}/${project_name}-prefix/src/${project_name}-stamp)
    set(git_stampfile ${CMAKE_CURRENT_BINARY_DIR}/${project_name}.stamp)
    set(git_check_rulename ${project_name}_check_git_clean)
    add_custom_target(${git_check_rulename}
      COMMAND ${SYMFORCE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rerun_if_needed.py
                  --name ${project_name}
                  --path_to_check ${PROJECT_SOURCE_DIR}/third_party/${project_name}
                  --stamp_file ${git_stampfile}
                  --cmake_stampdir=${cmake_stampdir}
      )

    add_dependencies(${project_name} ${git_check_rulename})
  endfunction()

  add_sourcechanged_dependency(symengine)
  add_sourcechanged_dependency(symenginepy)

endif(SYMFORCE_BUILD_SYMENGINE)
