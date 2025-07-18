/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <unsupported/Eigen/SparseExtra>

#include <symforce/opt/internal/covariance_utils.h>
#include <symforce/opt/tic_toc.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error \
    "The matrix loading code only supports little-endian systems, to run it on a big-endian " \
    "system you'll need to add support for that."
#endif

void WriteBinaryMatrix(const Eigen::SparseMatrix<double>& m, const std::string& path) {
  SYM_ASSERT(m.isCompressed());

  std::ofstream file(path, std::ios::binary);

  SYM_ASSERT_LE(m.rows(), std::numeric_limits<int32_t>::max());
  SYM_ASSERT_LE(m.cols(), std::numeric_limits<int32_t>::max());
  SYM_ASSERT_LE(m.nonZeros(), std::numeric_limits<int32_t>::max());
  const auto rows = static_cast<int32_t>(m.rows());
  const auto cols = static_cast<int32_t>(m.cols());
  const auto nnz = static_cast<int32_t>(m.nonZeros());
  file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  file.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));

  static_assert(
      std::is_same_v<std::decay_t<std::remove_pointer_t<decltype(m.innerIndexPtr())>>, int32_t>);
  static_assert(
      std::is_same_v<std::decay_t<std::remove_pointer_t<decltype(m.outerIndexPtr())>>, int32_t>);
  static_assert(
      std::is_same_v<std::decay_t<std::remove_pointer_t<decltype(m.valuePtr())>>, double>);

  file.write(reinterpret_cast<const char*>(m.innerIndexPtr()), m.nonZeros() * sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(m.outerIndexPtr()),
             (m.outerSize() + 1) * sizeof(int32_t));
  file.write(reinterpret_cast<const char*>(m.valuePtr()), m.nonZeros() * sizeof(double));
}

Eigen::SparseMatrix<double> LoadBinaryMatrix(const std::string& path) {
  std::ifstream file(path, std::ios::binary);

  int32_t rows, cols, nnz;
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  file.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

  SYM_ASSERT(!file.eof() && !file.fail());

  std::vector<int32_t> inner_index(nnz);
  std::vector<int32_t> outer_index(cols + 1);
  std::vector<double> values(nnz);

  file.read(reinterpret_cast<char*>(inner_index.data()), nnz * sizeof(int32_t));
  file.read(reinterpret_cast<char*>(outer_index.data()), (cols + 1) * sizeof(int32_t));
  file.read(reinterpret_cast<char*>(values.data()), nnz * sizeof(double));

  SYM_ASSERT(!file.eof() && !file.fail());

  return Eigen::Map<Eigen::SparseMatrix<double>>{
      rows, cols, nnz, outer_index.data(), inner_index.data(), values.data()};
}

template <typename Scalar>
Eigen::SparseMatrix<Scalar> LoadMatrix() {
#define _SYMFORCE_STRINGIFY(s) #s
#define SYMFORCE_STRINGIFY(s) _SYMFORCE_STRINGIFY(s)
  static const std::string filename =
      std::string(SYMFORCE_STRINGIFY(SYMFORCE_DIR)) + "/test/test_data/covariance_test_matrix.bin";
#undef SYMFORCE_STRINGIFY
#undef _SYMFORCE_STRINGIFY

  return LoadBinaryMatrix(filename).cast<Scalar>();
}

TEST_CASE("Test covariance is correct for singular C", "[cov]") {
  using Scalar = double;

  const Eigen::SparseMatrix<Scalar> full_hessian_lower = LoadMatrix<Scalar>();
  const Eigen::SparseMatrix<Scalar> hessian_lower =
      full_hessian_lower.bottomRightCorner(1000, 1000);
  const int block_dim = 30;

  const sym::MatrixX<Scalar> dense_solution = sym::MatrixX<Scalar>(hessian_lower)
                                                  .completeOrthogonalDecomposition()
                                                  .pseudoInverse()
                                                  .topLeftCorner(block_dim, block_dim);

  sym::MatrixX<Scalar> covariance_block;
  const auto info = sym::internal::ComputeCovarianceBlockWithSchurComplementFromSparseC(
      hessian_lower, block_dim, covariance_block);

  CHECK(info == Eigen::Success);

  // TODO(aaron): Can this really not be smaller?
  CHECK(dense_solution.isApprox(covariance_block, 1e-2));
}

TEST_CASE("Test covariance is correct", "[cov]") {
  using Scalar = double;

  const Eigen::SparseMatrix<Scalar> full_hessian_lower = LoadMatrix<Scalar>();
  const Eigen::SparseMatrix<Scalar> hessian_lower = full_hessian_lower.topLeftCorner(1000, 1000);
  const int block_dim = 30;

  const sym::MatrixX<Scalar> dense_solution = sym::MatrixX<Scalar>(hessian_lower)
                                                  .completeOrthogonalDecomposition()
                                                  .pseudoInverse()
                                                  .topLeftCorner(block_dim, block_dim);

  sym::MatrixX<Scalar> covariance_block;
  const auto info = sym::internal::ComputeCovarianceBlockWithSchurComplementFromSparseC(
      hessian_lower, block_dim, covariance_block, /* epsilon */ 0);

  CHECK(info == Eigen::Success);

  // TODO(aaron): Can this really not be smaller?
  CHECK(dense_solution.isApprox(covariance_block, 1e-1));
}
