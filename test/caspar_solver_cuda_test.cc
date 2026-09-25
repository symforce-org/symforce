/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "solver.h"

namespace {

constexpr size_t kNumNodes = 4096;

bool HaveDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

struct Solve {
  caspar::SolveResult result;
  std::vector<double> x;
};

// Independent Rosenbrock problems, all started at (-1.2, 1).
Solve Run(const caspar::SolverParams<double>& params) {
  caspar::GraphSolver solver(params, kNumNodes, kNumNodes);

  std::vector<unsigned int> indices(kNumNodes);
  std::vector<double> x(2 * kNumNodes);
  for (size_t i = 0; i < kNumNodes; i++) {
    indices[i] = static_cast<unsigned int>(i);
    x[2 * i + 0] = -1.2;
    x[2 * i + 1] = 1.0;
  }

  solver.SetRosenbrockXIndicesFromHost(indices.data(), kNumNodes);
  solver.SetMatrix21NodesFromStackedHost(x.data(), 0, kNumNodes);
  solver.finish_indices();

  Solve solve;
  solve.result = solver.solve(/* print_progress */ false, /* verbose_logging */ true);
  solve.x.resize(2 * kNumNodes);
  solver.GetMatrix21NodesToStackedHost(solve.x.data(), 0, kNumNodes);
  return solve;
}

// Half the sum of squared residuals, which is what the solver reports.
double ScoreOf(const std::vector<double>& x) {
  double score = 0.0;
  for (size_t i = 0; i < kNumNodes; i++) {
    const double r0 = 10.0 * (x[2 * i + 1] - x[2 * i + 0] * x[2 * i + 0]);
    const double r1 = 1.0 - x[2 * i + 0];
    score += r0 * r0 + r1 * r1;
  }
  return 0.5 * score;
}

caspar::SolverParams<double> ParamsWithIterations(const int solver_iter_max) {
  caspar::SolverParams<double> params;
  params.solver_iter_max = solver_iter_max;
  return params;
}

}  // namespace

TEST_CASE("Solver drives a bank of Rosenbrock problems toward the optimum",
          "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  const Solve solve = Run(ParamsWithIterations(50));

  CHECK(std::isfinite(solve.result.final_score));
  CHECK(solve.result.final_score < solve.result.initial_score);
  CHECK(solve.x[0] > -1.2);
  CHECK(ScoreOf(solve.x) == Catch::Approx(solve.result.final_score).epsilon(1e-9));
}

TEST_CASE("The last iteration makes progress", "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  const Solve one = Run(ParamsWithIterations(1));
  const Solve two = Run(ParamsWithIterations(2));

  CHECK(std::isfinite(two.result.final_score));
  CHECK(two.result.final_score < one.result.final_score);
}

TEST_CASE("step_accepted marks exactly the iterations that moved the score",
          "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  const Solve solve = Run(ParamsWithIterations(8));
  REQUIRE(solve.result.iterations.size() > 1);

  int accepted = 0;
  double previous_best = solve.result.initial_score;
  for (const caspar::IterationData& iteration : solve.result.iterations) {
    CAPTURE(iteration.solver_iter, iteration.score_best, previous_best);
    CHECK(iteration.step_accepted == (iteration.score_best < previous_best));
    accepted += iteration.step_accepted;
    previous_best = iteration.score_best;
  }
  CHECK(accepted > 0);
}

TEST_CASE("A solve that accepts non-monotonically still returns what it reports",
          "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  caspar::SolverParams<double> params = ParamsWithIterations(8);
  params.pcg_rel_decrease_min = 0.5;
  params.solver_rel_decrease_min = 1.5;

  const Solve solve = Run(params);

  CHECK(std::isfinite(solve.result.final_score));
  CHECK(ScoreOf(solve.x) == Catch::Approx(solve.result.final_score).epsilon(1e-9));
}

TEST_CASE("A solver with no inner iterations is rejected", "[caspar_solver_cuda_test]") {
  REQUIRE(HaveDevice());

  caspar::SolverParams<double> params = ParamsWithIterations(8);
  params.pcg_iter_max = 0;

  CHECK_THROWS(caspar::GraphSolver(params, kNumNodes, kNumNodes));
}
