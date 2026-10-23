/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <random>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>

#include "symforce_caspar_tridiagonal_test/solver.h"

namespace {

bool HaveDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// The linear problem of the generated library: chains of ChainA (2D) and ChainB (3D) nodes, each
// node with a prior, consecutive nodes linked by a weighted difference, and optional couplings
// between an A node and a B node.
struct Problem {
  int num_a = 0;
  int num_b = 0;
  std::vector<Eigen::Matrix2d> a_link_weight;
  std::vector<Eigen::Vector2d> a_link_offset;
  std::vector<Eigen::Vector2d> a_target;
  std::vector<Eigen::Matrix3d> b_link_weight;
  std::vector<Eigen::Vector3d> b_link_offset;
  std::vector<Eigen::Vector3d> b_target;
  std::vector<unsigned int> coupling_a;
  std::vector<unsigned int> coupling_b;
  std::vector<Eigen::Matrix<double, 2, 3>> coupling_weight;
  std::vector<Eigen::Vector2d> coupling_offset;
  Eigen::VectorXd initial;  // [a_0, ..., a_{num_a - 1}, b_0, ..., b_{num_b - 1}]
};

Problem MakeProblem(const int num_a, const int num_b, const int num_couplings,
                    const unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  const auto random = [&](auto matrix) {
    for (int i = 0; i < matrix.size(); i++) {
      matrix.data()[i] = uniform(rng);
    }
    return matrix;
  };

  Problem problem;
  problem.num_a = num_a;
  problem.num_b = num_b;
  for (int i = 0; i + 1 < num_a; i++) {
    problem.a_link_weight.push_back(Eigen::Matrix2d::Identity() * 3.0 + random(Eigen::Matrix2d()));
    problem.a_link_offset.push_back(random(Eigen::Vector2d()));
  }
  for (int i = 0; i < num_a; i++) {
    problem.a_target.push_back(random(Eigen::Vector2d()));
  }
  for (int i = 0; i + 1 < num_b; i++) {
    problem.b_link_weight.push_back(Eigen::Matrix3d::Identity() * 3.0 + random(Eigen::Matrix3d()));
    problem.b_link_offset.push_back(random(Eigen::Vector3d()));
  }
  for (int i = 0; i < num_b; i++) {
    problem.b_target.push_back(random(Eigen::Vector3d()));
  }
  for (int i = 0; i < num_couplings; i++) {
    problem.coupling_a.push_back(static_cast<unsigned int>(rng() % num_a));
    problem.coupling_b.push_back(static_cast<unsigned int>(rng() % num_b));
    problem.coupling_weight.push_back(random(Eigen::Matrix<double, 2, 3>()) * 2.0);
    problem.coupling_offset.push_back(random(Eigen::Vector2d()));
  }
  problem.initial = random(Eigen::VectorXd(2 * num_a + 3 * num_b)) * 5.0;
  return problem;
}

// The minimizer of the sum of squared residuals, from a dense solve of J^T J x = J^T c, where the
// residuals are J x - c.
Eigen::VectorXd DenseSolution(const Problem& problem) {
  const int num_rows = 2 * (problem.num_a - 1) + 2 * problem.num_a + 3 * (problem.num_b - 1) +
                       3 * problem.num_b + 2 * static_cast<int>(problem.coupling_a.size());
  const int num_cols = 2 * problem.num_a + 3 * problem.num_b;
  const int b_col = 2 * problem.num_a;
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(num_rows, num_cols);
  Eigen::VectorXd constant = Eigen::VectorXd::Zero(num_rows);
  int row = 0;
  for (int i = 0; i + 1 < problem.num_a; i++, row += 2) {
    jacobian.block<2, 2>(row, 2 * i) = -problem.a_link_weight[i];
    jacobian.block<2, 2>(row, 2 * i + 2) = problem.a_link_weight[i];
    constant.segment<2>(row) = problem.a_link_offset[i];
  }
  for (int i = 0; i < problem.num_a; i++, row += 2) {
    jacobian.block<2, 2>(row, 2 * i).setIdentity();
    constant.segment<2>(row) = problem.a_target[i];
  }
  for (int i = 0; i + 1 < problem.num_b; i++, row += 3) {
    jacobian.block<3, 3>(row, b_col + 3 * i) = -problem.b_link_weight[i];
    jacobian.block<3, 3>(row, b_col + 3 * i + 3) = problem.b_link_weight[i];
    constant.segment<3>(row) = problem.b_link_offset[i];
  }
  for (int i = 0; i < problem.num_b; i++, row += 3) {
    jacobian.block<3, 3>(row, b_col + 3 * i).setIdentity();
    constant.segment<3>(row) = problem.b_target[i];
  }
  for (size_t i = 0; i < problem.coupling_a.size(); i++, row += 2) {
    jacobian.block<2, 2>(row, 2 * problem.coupling_a[i]).setIdentity();
    jacobian.block<2, 3>(row, b_col + 3 * problem.coupling_b[i]) = -problem.coupling_weight[i];
    constant.segment<2>(row) = problem.coupling_offset[i];
  }
  return (jacobian.transpose() * jacobian).ldlt().solve(jacobian.transpose() * constant);
}

template <typename Matrix>
std::vector<double> Stacked(const std::vector<Matrix>& matrices) {
  std::vector<double> stacked;
  for (const Matrix& matrix : matrices) {
    stacked.insert(stacked.end(), matrix.data(), matrix.data() + matrix.size());
  }
  return stacked;
}

std::vector<unsigned int> Range(const int num) {
  std::vector<unsigned int> range(num);
  for (int i = 0; i < num; i++) {
    range[i] = static_cast<unsigned int>(i);
  }
  return range;
}

struct Solve {
  caspar::SolveResult result;
  Eigen::VectorXd x;
};

Solve Run(const Problem& problem, const caspar::SolverParams<double>& params) {
  const int num_a = problem.num_a;
  const int num_b = problem.num_b;
  const int num_couplings = static_cast<int>(problem.coupling_a.size());
  caspar::GraphSolver solver(params, num_a, num_b, num_a - 1, num_a, num_b - 1, num_b,
                             num_couplings);

  solver.SetChainANodesFromStackedHost(problem.initial.data(), 0, num_a);
  solver.SetChainBNodesFromStackedHost(problem.initial.data() + 2 * num_a, 0, num_b);
  if (num_a > 1) {
    solver.SetChainALinkWeightDataFromStackedHost(Stacked(problem.a_link_weight).data(), 0,
                                                  num_a - 1);
    solver.SetChainALinkOffsetDataFromStackedHost(Stacked(problem.a_link_offset).data(), 0,
                                                  num_a - 1);
  }
  solver.SetChainAPriorAIndicesFromHost(Range(num_a).data(), num_a);
  solver.SetChainAPriorTargetDataFromStackedHost(Stacked(problem.a_target).data(), 0, num_a);
  if (num_b > 1) {
    solver.SetChainBLinkWeightDataFromStackedHost(Stacked(problem.b_link_weight).data(), 0,
                                                  num_b - 1);
    solver.SetChainBLinkOffsetDataFromStackedHost(Stacked(problem.b_link_offset).data(), 0,
                                                  num_b - 1);
  }
  solver.SetChainBPriorBIndicesFromHost(Range(num_b).data(), num_b);
  solver.SetChainBPriorTargetDataFromStackedHost(Stacked(problem.b_target).data(), 0, num_b);
  if (num_couplings > 0) {
    solver.SetChainCouplingAIndicesFromHost(problem.coupling_a.data(), num_couplings);
    solver.SetChainCouplingBIndicesFromHost(problem.coupling_b.data(), num_couplings);
    solver.SetChainCouplingWeightDataFromStackedHost(Stacked(problem.coupling_weight).data(), 0,
                                                     num_couplings);
    solver.SetChainCouplingOffsetDataFromStackedHost(Stacked(problem.coupling_offset).data(), 0,
                                                     num_couplings);
  }
  solver.finish_indices();

  Solve solve;
  solve.result = solver.solve(/* print_progress */ false, /* verbose_logging */ true);
  solve.x.resize(2 * num_a + 3 * num_b);
  solver.GetChainANodesToStackedHost(solve.x.data(), 0, num_a);
  solver.GetChainBNodesToStackedHost(solve.x.data() + 2 * num_a, 0, num_b);
  return solve;
}

caspar::SolverParams<double> Params(const int solver_iter_max) {
  caspar::SolverParams<double> params;
  params.solver_iter_max = solver_iter_max;
  params.pcg_iter_max = 200;
  params.pcg_rel_error_exit = 1e-24;
  params.diag_init = 1e-12;
  return params;
}

}  // namespace

TEST_CASE("Uncoupled chains are solved exactly by one preconditioned CG iteration",
          "[caspar_tridiagonal_cuda_test]") {
  REQUIRE(HaveDevice());
  const int num_a = GENERATE(1, 2, 3, 7, 8);
  const int num_b = GENERATE(1, 2, 7, 8);
  CAPTURE(num_a, num_b);

  const Problem problem = MakeProblem(num_a, num_b, /* num_couplings */ 0, num_a * 100 + num_b);
  const Solve solve = Run(problem, Params(/* solver_iter_max */ 1));

  REQUIRE(solve.result.iterations.size() == 1);
  CHECK(solve.result.iterations[0].step_accepted);
  // The block-tridiagonal preconditioner is the inverse of the damped normal matrix.
  CHECK(solve.result.iterations[0].pcg_iter <= 1);
  const Eigen::VectorXd expected = DenseSolution(problem);
  CAPTURE(solve.x.transpose(), expected.transpose());
  CHECK((solve.x - expected).norm() < 1e-8 * (1.0 + expected.norm()));
}

TEST_CASE("Coupled chains converge to the dense solution", "[caspar_tridiagonal_cuda_test]") {
  REQUIRE(HaveDevice());
  const int num_a = GENERATE(1, 2, 3, 7, 8);
  const int num_b = GENERATE(1, 2, 7, 8);
  CAPTURE(num_a, num_b);

  const Problem problem =
      MakeProblem(num_a, num_b, /* num_couplings */ num_a + num_b, num_a * 100 + num_b + 1);
  const Solve solve = Run(problem, Params(/* solver_iter_max */ 5));

  const Eigen::VectorXd expected = DenseSolution(problem);
  CAPTURE(solve.x.transpose(), expected.transpose());
  CHECK((solve.x - expected).norm() < 1e-8 * (1.0 + expected.norm()));
}
