# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ==============================================================================
# Third Party Dependencies
# ==============================================================================

include(FetchContent)

# ------------------------------------------------------------------------------
# Sophus

find_package(Sophus QUIET)
if (NOT Sophus_FOUND)
  message(STATUS "Sophus not found, adding with FetchContent")
  function(add_sophus)
    set(BUILD_SOPHUS_TESTS OFF CACHE INTERNAL "Sophus shouldn't build tests by default")
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    FetchContent_Declare(
      sophus
      URL https://github.com/strasdat/Sophus/archive/refs/tags/1.24.6.zip
      URL_HASH SHA256=149bb58e4f136c25349fde84cc6244d7669a8e8101006fd8543478152cb20f45
    )
    FetchContent_MakeAvailable(sophus)
    find_package(Sophus)
  endfunction()

  add_sophus()
else()
  message(STATUS "Sophus found")
endif()

# ------------------------------------------------------------------------------
# GTSAM

find_package(gtsam QUIET)
if (NOT gtsam_FOUND)
  # NOTE(aaron): gtsam will not build against a copy of Eigen we've downloaded but not installed,
  # because they include the eigen include directories in their install targets, and CMake
  # complains (correctly) about directories inside the build folder in install targets.  They
  # handle this correctly for their bundled version, but we cannot use that because the version of
  # Eigen that GTSAM is built with must match our version (which is what's used when we call GTSAM
  # code)
  message(STATUS "GTSAM not found, adding with FetchContent")
  function(add_gtsam)
    set(GTSAM_BUILD_WITH_MARCH_NATIVE OFF CACHE INTERNAL "GTSAM shouldn't build native by default")
    set(GTSAM_BUILD_TESTS OFF CACHE INTERNAL "Don't build GTSAM tests")
    set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE INTERNAL "Don't build GTSAM examples")
    set(GTSAM_WITH_TBB OFF CACHE INTERNAL "Don't build TBB with GTSAM for accurate profiling")
    # NOTE(aaron): This seems to have no effect on the experiments we're doing
    set(GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR ON CACHE INTERNAL "Use the correct BetweenFactor")
    # NOTE(aaron): This has no effect other than making inverse compose chained slower
    set(GTSAM_USE_QUATERNIONS ON CACHE INTERNAL "Use quaternions for Rot3")
    set(GTSAM_USE_SYSTEM_EIGEN ON CACHE INTERNAL "Use system Eigen")
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    FetchContent_Declare(
      gtsam
      # Need https://github.com/borglab/gtsam/pull/1899 to build with CMake 4.0
      # This isn't in a stable release yet, so we use 4.3a0
      URL https://github.com/borglab/gtsam/archive/refs/tags/4.3a0.zip
      URL_HASH SHA256=fccfffe225fb5e7e9e8a8e0a07f15881da54ecfdbcc3a02c5f090d298228f16b
    )
    FetchContent_MakeAvailable(gtsam)
  endfunction()

  add_gtsam()
else()
  message(STATUS "GTSAM found")
endif()

# ------------------------------------------------------------------------------
# Ceres

find_package(Ceres QUIET)
if (NOT Ceres_FOUND)
  message(STATUS "Ceres not found, adding with FetchContent")
  function(add_ceres)
    set(BUILD_TESTING OFF CACHE INTERNAL "Don't enable tests")
    set(BUILD_EXAMPLES OFF CACHE INTERNAL "Don't build examples")
    set(BUILD_BENCHMARKS OFF CACHE INTERNAL "Don't build Ceres benchmarking suite")
    set(PROVIDE_UNINSTALL_TARGET
      OFF CACHE INTERNAL
      "Ceres shouldn't add uninstall target, gtsam already adds a target with the same name that collides"
      FORCE
    )
    set(
      CERES_THREADING_MODEL "NO_THREADS" CACHE INTERNAL "Don't use threads for benchmarking" FORCE
    )
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    # We don't need this, and Ceres/CMake can have issues with CUDA setup
    set(USE_CUDA OFF)
    FetchContent_Declare(
      ceres
      URL https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.zip
      URL_HASH SHA256=1fc28e22ce190ce4c1db04d0c2ddfdf8dc836609866d09372ea8533c8b6d490b
    )
    FetchContent_MakeAvailable(ceres)
  endfunction()

  add_ceres()
else()
  message(STATUS "Ceres found")
endif()

# ==============================================================================
# Benchmark Targets
# ==============================================================================

function(add_matrix_multiplication_benchmark matrix_name)
    add_executable(
        matrix_multiplication_benchmark_${matrix_name}
        matrix_multiplication/gen/matrix_multiplication_benchmark_${matrix_name}.cc
    )

    target_link_libraries(
        matrix_multiplication_benchmark_${matrix_name}
        Catch2::Catch2WithMain
        symforce_gen
        symforce_opt
    )

    set_target_properties(matrix_multiplication_benchmark_${matrix_name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
    )
endfunction()

add_matrix_multiplication_benchmark(b1_ss)
add_matrix_multiplication_benchmark(Tina_DisCog)
add_matrix_multiplication_benchmark(n3c4_b2)
add_matrix_multiplication_benchmark(bibd_9_3)
add_matrix_multiplication_benchmark(lp_sc105)
add_matrix_multiplication_benchmark(rotor1)

# -----------------------------------------------------------------------------

add_executable(
    inverse_compose_jacobian_benchmark
    inverse_compose_jacobian/inverse_compose_jacobian_benchmark.cc
)

target_link_libraries(
    inverse_compose_jacobian_benchmark
    gtsam
    Catch2::Catch2WithMain
    Sophus::Sophus
    symforce_gen
    symforce_opt
)

set_target_properties(inverse_compose_jacobian_benchmark
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)

# -----------------------------------------------------------------------------

add_executable(
    robot_3d_localization_benchmark
    robot_3d_localization/robot_3d_localization_benchmark.cc
)

target_link_libraries(
    robot_3d_localization_benchmark
    gtsam
    Catch2::Catch2WithMain
    symforce_gen
    symforce_opt
    symforce_examples
    Ceres::ceres
)

set_target_properties(robot_3d_localization_benchmark
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)
