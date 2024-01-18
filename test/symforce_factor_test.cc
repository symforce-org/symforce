/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <lcmtypes/sym/index_entry_t.hpp>
#include <lcmtypes/sym/linearized_dense_factor_t.hpp>

#include <sym/factors/prior_factor_rot3.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/key.h>

TEST_CASE("Test jacobian constructors", "[factors]") {
  sym::Valuesd values;
  values.Set<double>('x', 1.0);
  values.Set<double>('y', 2.0);
  values.Set<double>('z', -3.0);
  values.Set<sym::Rot3d>({'R', 1}, sym::Rot3d::Identity());
  values.Set<sym::Rot3d>({'R', 2}, sym::Rot3d::FromYawPitchRoll(1.0, 0.3, 0.5));
  values.Set<sym::Pose3d>('P', sym::Pose3d::Identity());
  values.Set<double>('e', 1e-9);

  sym::linearized_sparse_factor_t linearized_sparse_factor{};

  // Unary / v1 output with fixed size
  const sym::Factord unary_v1 = sym::Factord::Jacobian(
      [](double a, Eigen::Matrix<double, 1, 1>* res, Eigen::Matrix<double, 1, 1>* jac) {
        (*res) << a * a;
        (*jac) << 2 * a;
      },
      {'x'});
  INFO(unary_v1.Linearize(values));

  // Unary / v1 output with dynamic size
  const sym::Factord unary_v1_dyn = sym::Factord::Jacobian(
      [](double a, Eigen::VectorXd* res, Eigen::MatrixXd* jac) {
        res->resize(1);
        (*res) << a * a;
        jac->resize(1, 1);
        (*jac) << 2 * a;
      },
      {'x'});
  INFO(unary_v1_dyn.Linearize(values));

  // Unary / v2 output with fixed size
  const sym::Factord unary_v2 = sym::Factord::Jacobian(
      [](double a, Eigen::Matrix<double, 2, 1>* res, Eigen::Matrix<double, 2, 1>* jac) {
        (*res) << a * a, a - 1;
        (*jac) << 2 * a, 1.0;
      },
      {'x'});
  INFO(unary_v2.Linearize(values));

  // Unary / v2 output with dynamic size
  const sym::Factord unary_v2_dyn = sym::Factord::Jacobian(
      [](double a, Eigen::VectorXd* res, Eigen::MatrixXd* jac) {
        res->resize(2);
        (*res) << a * a, a - 1;
        jac->resize(2, 1);
        (*jac) << 2 * a, 1.0;
      },
      {'x'});
  INFO(unary_v2_dyn.Linearize(values));

  // Unary / v3 output with fixed size
  const sym::Factord unary_v3 = sym::Factord::Jacobian(
      [](double a, Eigen::Matrix<double, 3, 1>* res, Eigen::Matrix<double, 3, 1>* jac) {
        (*res) << a * a, a - 1, 4.0;
        (*jac) << 2 * a, 1.0, 0.0;
      },
      {'x'});
  INFO(unary_v3.Linearize(values));

  // Unary / v3 output with dynamic size
  const sym::Factord unary_v3_dyn = sym::Factord::Jacobian(
      [](double a, Eigen::VectorXd* res, Eigen::MatrixXd* jac) {
        res->resize(3);
        (*res) << a * a, a - 1, 4.0;
        jac->resize(3, 1);
        (*jac) << 2 * a, 1.0, 0.0;
      },
      {'x'});
  INFO(unary_v3_dyn.Linearize(values));

  // Binary / v1 output with fixed size
  const sym::Factord binary_v1 = sym::Factord::Jacobian(
      [](double a, double b, Eigen::Matrix<double, 1, 1>* res, Eigen::Matrix<double, 1, 2>* jac) {
        (*res) << a * a + 2 * b;
        (*jac) << 2 * a, 2.0;
      },
      {'x', 'y'});
  INFO(binary_v1.Linearize(values));

  // Binary / v1 output with dynamic size
  const sym::Factord binary_v1_dyn = sym::Factord::Jacobian(
      [](double a, double b, Eigen::VectorXd* res, Eigen::MatrixXd* jac) {
        res->resize(1);
        (*res) << a * a + 2 * b;
        jac->resize(1, 2);
        (*jac) << 2 * a, 2.0;
      },
      {'x', 'y'});
  INFO(binary_v1_dyn.Linearize(values));

  // Binary / v2 output with fixed size
  const sym::Factord binary_v2 = sym::Factord::Jacobian(
      [](double a, double b, Eigen::Matrix<double, 2, 1>* res, Eigen::Matrix<double, 2, 2>* jac) {
        (*res) << a * a, 2 * b;
        (*jac) << 2 * a, 0.0, 0.0, 2.0;
      },
      {'x', 'y'});
  INFO(binary_v2.Linearize(values));

  // Binary / v2 output with dynamic size
  const sym::Factord binary_v2_dyn = sym::Factord::Jacobian(
      [](double a, double b, Eigen::VectorXd* res, Eigen::MatrixXd* jac) {
        res->resize(2);
        (*res) << a * a, 2 * b;
        jac->resize(2, 2);
        (*jac) << 2 * a, 0.0, 0.0, 2.0;
      },
      {'x', 'y'});
  INFO(binary_v2_dyn.Linearize(values));

  // Ternary / v3 output with fixed size
  const sym::Factord ternary_v3 = sym::Factord::Jacobian(
      [](double a, double b, double c, Eigen::Matrix<double, 3, 1>* res,
         Eigen::Matrix<double, 3, 3>* jac) {
        (*res) << a * a, 2 * b, b + c;
        (*jac) << 2 * a, 0.0, 0.0,  //
            0.0, 2.0, 0.0,          //
            0.0, 1.0, 1.0;
      },
      {'x', 'y', 'z'});
  INFO(ternary_v3.Linearize(values));

  // This is not allowed, because we can't deduce the size the Rhs vector should be:
  // Ternary / v3 output with fixed size, and a sparse jacobian
  // const sym::Factord ternary_v3_sparse = sym::Factord::Jacobian(
  //     [](double a, double b, double c, Eigen::Matrix<double, 3, 1>* res,
  //        Eigen::SparseMatrix<double>* jac) {
  //       (*res) << a * a, 2 * b, b + c;
  //       jac->resize(3, 3);
  //       jac->coeffRef(0, 0) = 2 * a;
  //       jac->coeffRef(1, 1) = 2.0;
  //       jac->coeffRef(2, 1) = 1.0;
  //       jac->coeffRef(2, 2) = 1.0;
  //       jac->makeCompressed();
  //     },
  //     {'x', 'y', 'z'});
  // ternary_v3_sparse.Linearize(values, &linearized_sparse_factor);
  // INFO(linearized_sparse_factor);

  // Ternary / v3 output with dynamic size
  const sym::Factord ternary_v3_dyn = sym::Factord::Jacobian(
      [](double a, double b, double c, Eigen::VectorXd* res, Eigen::MatrixXd* jac) {
        res->resize(3);
        (*res) << a * a, 2 * b, b + c;
        jac->resize(3, 3);
        (*jac) << 2 * a, 0.0, 0.0,  //
            0.0, 2.0, 0.0,          //
            0.0, 1.0, 1.0;
      },
      {'x', 'y', 'z'});
  INFO(ternary_v3_dyn.Linearize(values));

  // Ternary / v3 output with dynamic size, and a sparse jacobian
  const sym::Factord ternary_v3_dyn_sparse = sym::Factord::Jacobian(
      [](double a, double b, double c, Eigen::VectorXd* res, Eigen::SparseMatrix<double>* jac) {
        res->resize(3);
        (*res) << a * a, 2 * b, b + c;
        jac->resize(3, 3);
        jac->coeffRef(0, 0) = 2 * a;
        jac->coeffRef(1, 1) = 2.0;
        jac->coeffRef(2, 1) = 1.0;
        jac->coeffRef(2, 2) = 1.0;
        jac->makeCompressed();
      },
      {'x', 'y', 'z'});
  ternary_v3_dyn_sparse.Linearize(values, linearized_sparse_factor);
  INFO(linearized_sparse_factor);

  // Unary with Rot3
  const sym::Factord unary_rot3 = sym::Factord::Jacobian(
      [](const sym::Rot3d& rot, Eigen::Matrix<double, 2, 1>* res,
         Eigen::Matrix<double, 2, 3>* jac) {
        (*res) << rot.ToYawPitchRoll().tail<2>();
        (*jac) << 2.0 * rot.ToTangent().transpose(), -1.0 * rot.ToTangent().transpose();  // fake
      },
      {{'R', 2}});
  INFO(unary_rot3.Linearize(values));

  // Binary with Rot3
  const sym::Factord binary_rot3 = sym::Factord::Jacobian(
      [](const sym::Rot3d& a, const sym::Rot3d& b, Eigen::Matrix<double, 3, 1>* res,
         Eigen::Matrix<double, 3, 6>* jac) {
        (*res) << a.LocalCoordinates(b);
        (*jac) << a.ToRotationMatrix(), b.ToRotationMatrix();  // fake
      },
      {{'R', 1}, {'R', 2}});
  INFO(binary_rot3.Linearize(values));

  // Huge one
  const sym::Factord big_factor = sym::Factord::Jacobian(
      [](double x, const sym::Rot3d& R1, double y, const sym::Rot3d& R2, const sym::Pose3d& P,
         double z, Eigen::Matrix<double, 2, 1>* res, Eigen::Matrix<double, 2, 15>* jac) {
        (*res)[0] = x + 2 * y + 3 * z;
        (*res)[1] = R1.Between(R2).Between(P.Rotation()).ToYawPitchRoll()[0];

        // FAKE
        (*jac).row(0) << 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3;
        (*jac).row(1) << 0, R1.ToTangent(), 0.0, R2.ToTangent(), 0, 0, 0, P.Rotation().ToTangent(),
            0.0;
      },
      {'x', {'R', 1}, 'y', {'R', 2}, 'P', 'z'});
  INFO(big_factor.Linearize(values));

  // An example with a std::bind expression
  using namespace std::placeholders;
  const Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();
  const sym::Factord prior_rot3_bind = sym::Factord::Jacobian(
      std::bind(sym::PriorFactorRot3<double>, _1, _2, sqrt_info, _3, _4, _5, nullptr, nullptr),
      {{'R', 1}, {'R', 2}, 'e'});
  INFO(prior_rot3_bind.Linearize(values));

  // Test keys_to_func != keys_to_optimize - pass in extra epsilon not being optimized
  const std::vector<sym::Key> keys_to_optimize = {{'R', 1}, {'R', 2}};
  std::vector<sym::Key> keys_to_func = keys_to_optimize;
  keys_to_func.push_back('e');
  const sym::Factord binary_rot3_with_epsilon = sym::Factord::Jacobian(
      [](const sym::Rot3d& a, const sym::Rot3d& b, const double epsilon,
         Eigen::Matrix<double, 3, 1>* res, Eigen::Matrix<double, 3, 6>* jac) {
        CHECK(epsilon == 1e-9);
        (*res) << a.LocalCoordinates(b, epsilon);
        (*jac) << a.ToRotationMatrix(), b.ToRotationMatrix();  // fake
      },
      keys_to_func, keys_to_optimize);
  INFO(binary_rot3_with_epsilon.Linearize(values));
}

TEST_CASE("Test hessian constructors", "[factors]") {
  sym::Valuesd values;
  values.Set<double>('x', 1.0);
  values.Set<double>('y', 2.0);
  values.Set<double>('z', -3.0);
  values.Set<sym::Rot3d>({'R', 1}, sym::Rot3d::Identity());
  values.Set<sym::Rot3d>({'R', 2}, sym::Rot3d::FromYawPitchRoll(1.0, 0.3, 0.5));
  values.Set<sym::Pose3d>('P', sym::Pose3d::Identity());
  values.Set<double>('e', 1e-9);

  sym::linearized_sparse_factor_t linearized_sparse_factor{};

  // Unary / v1 output with fixed size
  const sym::Factord unary_v1 = sym::Factord::Hessian(
      [](double a, Eigen::Matrix<double, 1, 1>* res, Eigen::Matrix<double, 1, 1>* jac,
         Eigen::Matrix<double, 1, 1>* hessian, Eigen::Matrix<double, 1, 1>* rhs) {
        (*res) << a * a;
        (*jac) << 2 * a;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x'});
  INFO(unary_v1.Linearize(values));

  // Unary / v1 output with dynamic size
  const sym::Factord unary_v1_dyn = sym::Factord::Hessian(
      [](double a, Eigen::VectorXd* res, Eigen::MatrixXd* jac, Eigen::MatrixXd* hessian,
         Eigen::VectorXd* rhs) {
        res->resize(1);
        (*res) << a * a;
        jac->resize(1, 1);
        (*jac) << 2 * a;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x'});
  INFO(unary_v1_dyn.Linearize(values));

  // Unary / v2 output with fixed size
  const sym::Factord unary_v2 = sym::Factord::Hessian(
      [](double a, Eigen::Matrix<double, 2, 1>* res, Eigen::Matrix<double, 2, 1>* jac,
         Eigen::Matrix<double, 1, 1>* hessian, Eigen::Matrix<double, 1, 1>* rhs) {
        (*res) << a * a, a - 1;
        (*jac) << 2 * a, 1.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x'});
  INFO(unary_v2.Linearize(values));

  // Unary / v2 output with dynamic size
  const sym::Factord unary_v2_dyn = sym::Factord::Hessian(
      [](double a, Eigen::VectorXd* res, Eigen::MatrixXd* jac, Eigen::MatrixXd* hessian,
         Eigen::VectorXd* rhs) {
        res->resize(2);
        (*res) << a * a, a - 1;
        jac->resize(2, 1);
        (*jac) << 2 * a, 1.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x'});
  INFO(unary_v2_dyn.Linearize(values));

  // Unary / v3 output with fixed size
  const sym::Factord unary_v3 = sym::Factord::Hessian(
      [](double a, Eigen::Matrix<double, 3, 1>* res, Eigen::Matrix<double, 3, 1>* jac,
         Eigen::Matrix<double, 1, 1>* hessian, Eigen::Matrix<double, 1, 1>* rhs) {
        (*res) << a * a, a - 1, 4.0;
        (*jac) << 2 * a, 1.0, 0.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x'});
  INFO(unary_v3.Linearize(values));

  // Unary / v3 output with dynamic size
  const sym::Factord unary_v3_dyn = sym::Factord::Hessian(
      [](double a, Eigen::VectorXd* res, Eigen::MatrixXd* jac, Eigen::MatrixXd* hessian,
         Eigen::VectorXd* rhs) {
        res->resize(3);
        (*res) << a * a, a - 1, 4.0;
        jac->resize(3, 1);
        (*jac) << 2 * a, 1.0, 0.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x'});
  INFO(unary_v3_dyn.Linearize(values));

  // Binary / v1 output with fixed size
  const sym::Factord binary_v1 = sym::Factord::Hessian(
      [](double a, double b, Eigen::Matrix<double, 1, 1>* res, Eigen::Matrix<double, 1, 2>* jac,
         Eigen::Matrix<double, 2, 2>* hessian, Eigen::Matrix<double, 2, 1>* rhs) {
        (*res) << a * a + 2 * b;
        (*jac) << 2 * a, 2.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', 'y'});
  INFO(binary_v1.Linearize(values));

  // Binary / v1 output with dynamic size
  const sym::Factord binary_v1_dyn = sym::Factord::Hessian(
      [](double a, double b, Eigen::VectorXd* res, Eigen::MatrixXd* jac, Eigen::MatrixXd* hessian,
         Eigen::VectorXd* rhs) {
        res->resize(1);
        (*res) << a * a + 2 * b;
        jac->resize(1, 2);
        (*jac) << 2 * a, 2.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', 'y'});
  INFO(binary_v1_dyn.Linearize(values));

  // Binary / v2 output with fixed size
  const sym::Factord binary_v2 = sym::Factord::Hessian(
      [](double a, double b, Eigen::Matrix<double, 2, 1>* res, Eigen::Matrix<double, 2, 2>* jac,
         Eigen::Matrix<double, 2, 2>* hessian, Eigen::Matrix<double, 2, 1>* rhs) {
        (*res) << a * a, 2 * b;
        (*jac) << 2 * a, 0.0, 0.0, 2.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', 'y'});
  INFO(binary_v2.Linearize(values));

  // Binary / v2 output with dynamic size
  const sym::Factord binary_v2_dyn = sym::Factord::Hessian(
      [](double a, double b, Eigen::VectorXd* res, Eigen::MatrixXd* jac, Eigen::MatrixXd* hessian,
         Eigen::VectorXd* rhs) {
        res->resize(2);
        (*res) << a * a, 2 * b;
        jac->resize(2, 2);
        (*jac) << 2 * a, 0.0, 0.0, 2.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', 'y'});
  INFO(binary_v2_dyn.Linearize(values));

  // Ternary / v3 output with fixed size
  const sym::Factord ternary_v3 = sym::Factord::Hessian(
      [](double a, double b, double c, Eigen::Matrix<double, 3, 1>* res,
         Eigen::Matrix<double, 3, 3>* jac, Eigen::Matrix<double, 3, 3>* hessian,
         Eigen::Matrix<double, 3, 1>* rhs) {
        (*res) << a * a, 2 * b, b + c;
        (*jac) << 2 * a, 0.0, 0.0,  //
            0.0, 2.0, 0.0,          //
            0.0, 1.0, 1.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', 'y', 'z'});
  INFO(ternary_v3.Linearize(values));

  // Ternary / v3 output with fixed size, and a sparse jacobian and hessian
  const sym::Factord ternary_v3_sparse = sym::Factord::Hessian(
      [](double a, double b, double c, Eigen::Matrix<double, 3, 1>* res,
         Eigen::SparseMatrix<double>* jac, Eigen::SparseMatrix<double>* hessian,
         Eigen::Matrix<double, 3, 1>* rhs) {
        (*res) << a * a, 2 * b, b + c;

        jac->resize(3, 3);
        jac->coeffRef(0, 0) = 2 * a;
        jac->coeffRef(1, 1) = 2.0;
        jac->coeffRef(2, 1) = 1.0;
        jac->coeffRef(2, 2) = 1.0;
        jac->makeCompressed();

        hessian->resize(jac->cols(), jac->cols());
        hessian->selfadjointView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).selfadjointView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', 'y', 'z'});
  ternary_v3_sparse.Linearize(values, linearized_sparse_factor);
  INFO(linearized_sparse_factor);

  // Ternary / v3 output with dynamic size
  const sym::Factord ternary_v3_dyn = sym::Factord::Hessian(
      [](double a, double b, double c, Eigen::VectorXd* res, Eigen::MatrixXd* jac,
         Eigen::MatrixXd* hessian, Eigen::VectorXd* rhs) {
        res->resize(3);
        (*res) << a * a, 2 * b, b + c;
        jac->resize(3, 3);
        (*jac) << 2 * a, 0.0, 0.0,  //
            0.0, 2.0, 0.0,          //
            0.0, 1.0, 1.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', 'y', 'z'});
  INFO(ternary_v3_dyn.Linearize(values));

  // Ternary / v3 output with dynamic size, and a sparse jacobian and hessian
  const sym::Factord ternary_v3_dyn_sparse = sym::Factord::Hessian(
      [](double a, double b, double c, Eigen::VectorXd* res, Eigen::SparseMatrix<double>* jac,
         Eigen::SparseMatrix<double>* hessian, Eigen::VectorXd* rhs) {
        res->resize(3);
        (*res) << a * a, 2 * b, b + c;
        jac->resize(3, 3);
        jac->coeffRef(0, 0) = 2 * a;
        jac->coeffRef(1, 1) = 2.0;
        jac->coeffRef(2, 1) = 1.0;
        jac->coeffRef(2, 2) = 1.0;
        jac->makeCompressed();

        hessian->resize(jac->cols(), jac->cols());
        hessian->selfadjointView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).selfadjointView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', 'y', 'z'});
  ternary_v3_dyn_sparse.Linearize(values, linearized_sparse_factor);
  INFO(linearized_sparse_factor);

  // Unary with Rot3
  const sym::Factord unary_rot3 = sym::Factord::Hessian(
      [](const sym::Rot3d& rot, Eigen::Matrix<double, 2, 1>* res, Eigen::Matrix<double, 2, 3>* jac,
         Eigen::Matrix<double, 3, 3>* hessian, Eigen::Matrix<double, 3, 1>* rhs) {
        (*res) << rot.ToYawPitchRoll().tail<2>();
        (*jac) << 2.0 * rot.ToTangent().transpose(), -1.0 * rot.ToTangent().transpose();  // fake

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {{'R', 2}});
  INFO(unary_rot3.Linearize(values));

  // Binary with Rot3
  const sym::Factord binary_rot3 = sym::Factord::Hessian(
      [](const sym::Rot3d& a, const sym::Rot3d& b, Eigen::Matrix<double, 3, 1>* res,
         Eigen::Matrix<double, 3, 6>* jac, Eigen::Matrix<double, 6, 6>* hessian,
         Eigen::Matrix<double, 6, 1>* rhs) {
        (*res) << a.LocalCoordinates(b);
        (*jac) << a.ToRotationMatrix(), b.ToRotationMatrix();  // fake

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {{'R', 1}, {'R', 2}});
  INFO(binary_rot3.Linearize(values));

  // Huge one
  const sym::Factord big_factor = sym::Factord::Hessian(
      [](double x, const sym::Rot3d& R1, double y, const sym::Rot3d& R2, const sym::Pose3d& P,
         double z, Eigen::Matrix<double, 2, 1>* res, Eigen::Matrix<double, 2, 15>* jac,
         Eigen::Matrix<double, 15, 15>* hessian, Eigen::Matrix<double, 15, 1>* rhs) {
        (*res)[0] = x + 2 * y + 3 * z;
        (*res)[1] = R1.Between(R2).Between(P.Rotation()).ToYawPitchRoll()[0];

        // FAKE
        (*jac).row(0) << 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3;
        (*jac).row(1) << 0, R1.ToTangent(), 0.0, R2.ToTangent(), 0, 0, 0, P.Rotation().ToTangent(),
            0.0;

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      {'x', {'R', 1}, 'y', {'R', 2}, 'P', 'z'});
  INFO(big_factor.Linearize(values));

  // An example with a std::bind expression
  using namespace std::placeholders;
  const Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();
  const sym::Factord prior_rot3_bind = sym::Factord::Hessian(
      std::bind(sym::PriorFactorRot3<double>, _1, _2, sqrt_info, _3, _4, _5, _6, _7),
      {{'R', 1}, {'R', 2}, 'e'});
  INFO(prior_rot3_bind.Linearize(values));

  // Test keys_to_func != keys_to_optimize - pass in extra epsilon not being optimized
  const std::vector<sym::Key> keys_to_optimize = {{'R', 1}, {'R', 2}};
  std::vector<sym::Key> keys_to_func = keys_to_optimize;
  keys_to_func.push_back('e');
  const sym::Factord binary_rot3_with_epsilon = sym::Factord::Hessian(
      [](const sym::Rot3d& a, const sym::Rot3d& b, const double epsilon,
         Eigen::Matrix<double, 3, 1>* res, Eigen::Matrix<double, 3, 6>* jac,
         Eigen::Matrix<double, 6, 6>* hessian, Eigen::Matrix<double, 6, 1>* rhs) {
        CHECK(epsilon == 1e-9);
        (*res) << a.LocalCoordinates(b, epsilon);
        (*jac) << a.ToRotationMatrix(), b.ToRotationMatrix();  // fake

        hessian->resize(jac->cols(), jac->cols());
        hessian->triangularView<Eigen::Lower>() =
            (jac->transpose() * (*jac)).triangularView<Eigen::Lower>();
        (*rhs) = jac->transpose() * (*res);
      },
      keys_to_func, keys_to_optimize);
  INFO(binary_rot3_with_epsilon.Linearize(values));
}

template <typename MatrixType>
struct LinearizedFactor;
template <>
struct LinearizedFactor<Eigen::MatrixXd> {
  using type = sym::linearized_dense_factor_t;
};
template <>
struct LinearizedFactor<Eigen::SparseMatrix<double>> {
  using type = sym::linearized_sparse_factor_t;
};
template <typename MatrixType>
using LinearizedFactor_t = typename LinearizedFactor<MatrixType>::type;

template <typename Matrix, typename... Keys, typename ResidualLambda, typename JacobianLambda>
void TestJacobianFuncHelper2Args(const ResidualLambda& eval_residual,
                                 const JacobianLambda& eval_jacobian,
                                 const std::vector<Keys>&... keys) {
  const sym::Factord factor =
      sym::Factord(sym::Factord::JacobianFunc<Matrix>(
                       [&eval_residual, &eval_jacobian](
                           const sym::Valuesd& values, const std::vector<sym::index_entry_t>& keys,
                           Eigen::VectorXd* residual, Matrix* jacobian) {
                         const double x = values.At<double>(keys[0]);
                         const double y = values.At<double>(keys[1]);
                         if (residual != nullptr) {
                           *residual = eval_residual(x, y);
                         }
                         if (jacobian != nullptr) {
                           *jacobian = eval_jacobian(x, y);
                         }
                       }),
                   {'x', 'y'}, keys...);

  const double x = 3.0;
  const double y = -1.0;
  sym::Valuesd values;
  values.Set<double>('x', x);
  values.Set<double>('y', y);

  LinearizedFactor_t<Matrix> linearization{};
  factor.Linearize(values, linearization);

  const Eigen::VectorXd res = eval_residual(x, y);
  const Matrix jac = eval_jacobian(x, y);
  CHECK(linearization.residual.isApprox(res));
  CHECK(linearization.jacobian.isApprox(jac));
  CHECK(Matrix(linearization.hessian.template selfadjointView<Eigen::Lower>())
            .isApprox(jac.transpose() * jac));
  CHECK(linearization.rhs.isApprox(jac.transpose() * res));
}

TEST_CASE("Test Factord(DenseJacobianFunc, keys) and linearization", "[factors]") {
  const auto eval_residual = [](const double x, const double y) {
    return (Eigen::VectorXd(2) << x * x, x * y).finished();
  };
  const auto eval_jacobian = [](const double x, const double y) {
    return (Eigen::MatrixXd(2, 2) << 2 * x, 0, y, x).finished();
  };

  TestJacobianFuncHelper2Args<Eigen::MatrixXd>(eval_residual, eval_jacobian);
}

TEST_CASE("Test Factord(DenseJacobianFunc, keys_to_func, keys_to_optimize", "[factors]") {
  const auto eval_residual = [](const double x, const double y) {
    return (Eigen::VectorXd(2) << x * x, x * y).finished();
  };
  const auto eval_jacobian = [](const double x, const double y) {
    return (Eigen::MatrixXd(2, 1) << 2 * x, y).finished();
  };

  TestJacobianFuncHelper2Args<Eigen::MatrixXd, sym::Key>(eval_residual, eval_jacobian, {'x'});
}

TEST_CASE("Test Factord(SparseJacobianFunc, keys) and linearization", "[factors]") {
  using Sparse = Eigen::SparseMatrix<double>;

  const auto eval_residual = [](const double x, const double y) -> Sparse {
    return (Eigen::VectorXd(2) << x * x, x * y).finished().sparseView();
  };
  const auto eval_jacobian = [](const double x, const double y) -> Sparse {
    return (Eigen::MatrixXd(2, 2) << 2 * x, 0, y, x).finished().sparseView();
  };

  TestJacobianFuncHelper2Args<Sparse>(eval_residual, eval_jacobian);
}

TEST_CASE("Test Factord(SparseJacobianFunc, keys_to_func, keys_to_opt", "[factors]") {
  using Sparse = Eigen::SparseMatrix<double>;

  const auto eval_residual = [](const double x, const double y) -> Sparse {
    return (Eigen::VectorXd(2) << x * x, x * y).finished().sparseView();
  };
  const auto eval_jacobian = [](const double x, const double y) -> Sparse {
    return (Eigen::MatrixXd(2, 1) << 2 * x, y).finished().sparseView();
  };

  TestJacobianFuncHelper2Args<Sparse, sym::Key>(eval_residual, eval_jacobian, {'x'});
}

TEMPLATE_TEST_CASE("Test linearized values", "[factors]", double, float) {
  using Scalar = TestType;

  sym::Key x('x');
  sym::Key y('y');
  sym::Key z('z');

  sym::Values<Scalar> values;
  values.template Set<Scalar>(x, 1.0);
  values.template Set<Scalar>(y, 2.0);
  values.template Set<Scalar>(z, -3.0);

  const auto func = [](Scalar a, Scalar b, Eigen::Matrix<Scalar, 1, 1>* res,
                       Eigen::Matrix<Scalar, 1, 2>* jac) {
    (*res) << a * a + b;

    if (jac) {
      (*jac) << 2 * a, 1;
    }
  };

  // Check with keys {x, y} so a = 1, b = 2
  const sym::Factor<Scalar> factor1 = sym::Factor<Scalar>::Jacobian(func, {x, y});
  const auto linearized1 = factor1.Linearize(values);
  CAPTURE(linearized1);
  CHECK(linearized1.residual[0] == Catch::Approx(3).epsilon(0).margin(1e-3));
  CHECK(linearized1.jacobian(0, 0) == Catch::Approx(2).epsilon(0).margin(1e-3));
  CHECK(linearized1.jacobian(0, 1) == Catch::Approx(1).epsilon(0).margin(1e-3));

  // Check another combination of keys, now a = -3, b = 1
  const sym::Factor<Scalar> factor2 = sym::Factor<Scalar>::Jacobian(func, {z, x});
  const auto linearized2 = factor2.Linearize(values);
  CAPTURE(linearized2);
  CHECK(linearized2.residual[0] == Catch::Approx(10).epsilon(0).margin(1e-3));
  CHECK(linearized2.jacobian(0, 0) == Catch::Approx(-6).epsilon(0).margin(1e-3));
  CHECK(linearized2.jacobian(0, 1) == Catch::Approx(1).epsilon(0).margin(1e-3));

  // Check Linearize with residual
  sym::VectorX<Scalar> residual;
  factor2.Linearize(values, &residual);
  CHECK(residual.isApprox(linearized2.residual));

  // Check Linearize with residual and jacobian
  sym::MatrixX<Scalar> jacobian;
  factor2.Linearize(values, &residual, &jacobian);
  CHECK(residual.isApprox(linearized2.residual));
  CHECK(jacobian.isApprox(linearized2.jacobian));
}

template <typename Scalar>
struct TestJacobianFunctor {
  TestJacobianFunctor() = default;
  TestJacobianFunctor(const TestJacobianFunctor&) {
    copies++;
  }

  TestJacobianFunctor& operator=(const TestJacobianFunctor&) {
    copies++;
    return *this;
  }

  TestJacobianFunctor(TestJacobianFunctor&& rhs) {
    rhs.is_moved = true;
  }
  TestJacobianFunctor& operator=(TestJacobianFunctor&& rhs) {
    rhs.is_moved = true;
    return *this;
  }

  void operator()(Scalar x, sym::Vector1<Scalar>* res, sym::Matrix11<Scalar>* jac) const {
    (*res) << x;
    (*jac) << 1;
  }

  bool is_moved{false};
  static int copies;
};

template <typename Scalar>
int TestJacobianFunctor<Scalar>::copies = 0;

TEMPLATE_TEST_CASE("Test Jacobian functors has minimal copies", "[factors]", double, float) {
  using Scalar = TestType;

  sym::Factor<Scalar>::Jacobian(TestJacobianFunctor<Scalar>(), {'x'});
  sym::Factor<Scalar>::Jacobian(TestJacobianFunctor<Scalar>(), {'x'}, {'x'});
  CHECK(TestJacobianFunctor<Scalar>::copies == 0);

  {
    TestJacobianFunctor<Scalar> f{};
    sym::Factor<Scalar>::Jacobian(std::move(f), {'x'});
    CHECK(f.is_moved);                                // f should be moved
    CHECK(TestJacobianFunctor<Scalar>::copies == 0);  // should not be copied
  }

  {
    TestJacobianFunctor<Scalar> f{};
    sym::Factor<Scalar>::Jacobian(f, {'x'});
    CHECK(!f.is_moved);                               // f should not be moved
    CHECK(TestJacobianFunctor<Scalar>::copies == 1);  // f should be copied once
  }
}

template <typename Scalar>
struct TestHessianFunctor {
  TestHessianFunctor() = default;
  TestHessianFunctor(const TestHessianFunctor&) {
    copies++;
  }

  TestHessianFunctor& operator=(const TestHessianFunctor&) {
    copies++;
    return *this;
  }

  TestHessianFunctor(TestHessianFunctor&& rhs) {
    rhs.is_moved = true;
  }
  TestHessianFunctor& operator=(TestHessianFunctor&& rhs) {
    rhs.is_moved = true;
    return *this;
  }

  void operator()(Scalar x, sym::Vector1<Scalar>* res, sym::Matrix11<Scalar>* jac,
                  sym::Matrix11<Scalar>* hes, sym::Matrix11<Scalar>* rhs) const {
    (*res) << x;
    (*jac) << 1;
    (*hes) << 1;
    (*rhs) << 1;
  }

  bool is_moved{false};
  static int copies;
};

template <typename Scalar>
int TestHessianFunctor<Scalar>::copies = 0;

TEMPLATE_TEST_CASE("Test Hessian functors has minimal copies", "[factors]", double, float) {
  using Scalar = TestType;

  sym::Factor<Scalar>::Hessian(TestHessianFunctor<Scalar>(), {'x'});
  sym::Factor<Scalar>::Hessian(TestHessianFunctor<Scalar>(), {'x'}, {'x'});
  CHECK(TestHessianFunctor<Scalar>::copies == 0);

  {
    TestHessianFunctor<Scalar> f{};
    sym::Factor<Scalar>::Hessian(std::move(f), {'x'});
    CHECK(f.is_moved);                               // f should be moved
    CHECK(TestHessianFunctor<Scalar>::copies == 0);  // should not be copied
  }

  {
    TestHessianFunctor<Scalar> f{};
    sym::Factor<Scalar>::Hessian(f, {'x'});
    CHECK(!f.is_moved);                              // f should not be moved
    CHECK(TestHessianFunctor<Scalar>::copies == 1);  // f should be copied once
  }
}

TEMPLATE_TEST_CASE("Test factor maybe_index_entry_cache", "[factors]", double, float) {
  using Scalar = TestType;

  sym::Values<Scalar> values;
  values.Set('x', Scalar(5));

  const std::vector<sym::Factor<Scalar>> factors{
      sym::Factor<Scalar>::Jacobian(TestJacobianFunctor<Scalar>(), {'x'}),
      sym::Factor<Scalar>::Hessian(TestHessianFunctor<Scalar>(), {'x'})};

  for (const auto& factor : factors) {
    // Able to linearize without passing index_entry_cache
    factor.Linearize(values);

    // Able to linearize with index_entry_cache
    std::vector<sym::index_entry_t> cache = values.CreateIndex({'x'}).entries;
    factor.Linearize(values, &cache);

    // Using the wrong size cache throws
    cache.emplace_back();
    CHECK_THROWS(factor.Linearize(values, &cache));

    cache.clear();
    CHECK_THROWS(factor.Linearize(values, &cache));
  }
}
