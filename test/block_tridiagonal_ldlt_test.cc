/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/block_tridiagonal_ldlt.h>

namespace {

using SparseMatrix = Eigen::SparseMatrix<double>;

// Largest absolute difference between the entries of a and b
double MaxAbsDiff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

// Builds the lower triangle of a random symmetric matrix with the given block tridiagonal layout.
// Each off-diagonal entry in the band is stored with probability `density` and has magnitude at
// most 1, while every diagonal entry is at least kDim. The matrix is therefore strictly diagonally
// dominant with a positive diagonal, and so positive definite.
template <int kFirstBlock, int kBlock, int kNumBlocks>
SparseMatrix RandomBlockTridiagonalLower(const double density, const int seed) {
  // Helper to get the block index of a given row/column index
  const auto block_of = [](const int i) {
    return i < kFirstBlock ? 0 : 1 + (i - kFirstBlock) / kBlock;
  };
  constexpr int kDim = kFirstBlock + kBlock * (kNumBlocks - 1);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::bernoulli_distribution stored(density);

  std::vector<Eigen::Triplet<double>> triplets;
  for (int c = 0; c < kDim; c++) {
    triplets.emplace_back(c, c, kDim + std::abs(value(rng)));
    for (int r = c + 1; r < kDim && block_of(r) <= block_of(c) + 1; r++) {
      if (stored(rng)) {
        triplets.emplace_back(r, c, value(rng));
      }
    }
  }
  SparseMatrix lower(kDim, kDim);
  lower.setFromTriplets(triplets.begin(), triplets.end());
  return lower;
}

// Factorizes a random matrix with the given layout and checks the results match Eigen's built-in
// LDLT solver.
template <int kFirstBlock, int kBlock, int kNumBlocks>
void CheckMatchesSparseLdlt(const double density, const int seed) {
  using Solver = sym::BlockTridiagonalLdlt<double, kFirstBlock, kBlock, kNumBlocks>;
  constexpr int kDim = Solver::kDim;

  // Build a random, positive definite matrix
  const SparseMatrix lower =
      RandomBlockTridiagonalLower<kFirstBlock, kBlock, kNumBlocks>(density, seed);
  const Eigen::MatrixXd full = Eigen::MatrixXd(lower).selfadjointView<Eigen::Lower>();

  Solver solver;
  solver.AnalyzeSparsityPattern(lower);
  solver.Factorize(lower);

  const Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower,
                              Eigen::NaturalOrdering<SparseMatrix::StorageIndex>>
      reference(lower);
  REQUIRE(reference.info() == Eigen::Success);
  CHECK(MaxAbsDiff(solver.D(), reference.vectorD()) < 1e-9);
  // Both store only the strictly lower part of the unit lower triangular factor
  CHECK(MaxAbsDiff(Eigen::MatrixXd(solver.L()),
                   Eigen::MatrixXd(reference.matrixL().nestedExpression())) < 1e-9);

  // Check that solving A*x = b gives the correct solution
  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(kDim, -1.0, 2.0);
  CHECK(MaxAbsDiff(full * solver.Solve(b), b) < 1e-9);

  // Check that solving A*I gives A^-1
  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(kDim, kDim);
  solver.SolveInPlace(x);
  CHECK(MaxAbsDiff(full * x, Eigen::MatrixXd::Identity(kDim, kDim)) < 1e-9);
}

}  // namespace

TEST_CASE("BlockTridiagonalLdlt matches Eigen on a sparse band", "[block_tridiagonal_ldlt]") {
  CheckMatchesSparseLdlt<3, 5, 4>(0.5, 1);
}

TEST_CASE("BlockTridiagonalLdlt matches Eigen with two blocks and a larger first block",
          "[block_tridiagonal_ldlt]") {
  CheckMatchesSparseLdlt<6, 2, 2>(0.9, 2);
}

TEST_CASE("BlockTridiagonalLdlt matches Eigen on a dense band", "[block_tridiagonal_ldlt]") {
  CheckMatchesSparseLdlt<2, 3, 6>(1.0, 3);
}
