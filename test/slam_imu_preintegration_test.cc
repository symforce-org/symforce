/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <limits>
#include <numeric>
#include <random>
#include <thread>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sym/factors/internal/imu_manifold_preintegration_update.h>
#include <sym/factors/internal/imu_manifold_preintegration_update_auto_derivative.h>
#include <sym/rot3.h>
#include <sym/util/epsilon.h>
#include <symforce/slam/imu_preintegration/imu_factor.h>
#include <symforce/slam/imu_preintegration/imu_preintegrator.h>

using Eigen::Ref;
using Eigen::Vector3d;
using sym::Rot3d;

/**
 * Calculates what ImuManifoldPreintegrationUpdate should be for the changes in orientation,
 * velocity, and position.
 */
void UpdateState(const Rot3d& DR, const Ref<const Vector3d>& Dv, const Ref<const Vector3d>& Dp,
                 const Ref<const Vector3d>& accel, const Ref<const Vector3d>& gyro,
                 const Ref<const Vector3d>& accel_bias, const Ref<const Vector3d>& gyro_bias,
                 const double dt, const double epsilon, Rot3d& new_DR, Ref<Vector3d> new_Dv,
                 Ref<Vector3d> new_Dp) {
  const auto corrected_accel = accel - accel_bias;
  new_DR = DR * Rot3d::FromTangent((gyro - gyro_bias) * dt, epsilon);
  new_Dv = Dv + DR * corrected_accel * dt;
  new_Dp = Dp + Dv * dt + DR * corrected_accel * (0.5 * dt * dt);
}

TEST_CASE("Test ImuManifoldPreintegrationUpdate has correct DR, Dv, & Dp", "[slam]") {
  std::mt19937 gen(1804);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Eigen::Matrix<double, 9, 9> nanM99 = Eigen::Matrix<double, 9, 9>::Constant(nan);
  const Eigen::Matrix3d nanM33 = Eigen::Matrix3d::Constant(nan);

  for (int i_ = 0; i_ < 10; i_++) {
    const sym::Rot3d DR = sym::Rot3d::Random(gen);
    const Eigen::Vector3d Dv = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d Dp = sym::Random<Eigen::Vector3d>(gen);

    // Set to NaN because output should not depend on them
    const Eigen::Matrix<double, 9, 9> covariance = nanM99;
    const Eigen::Matrix3d DR_D_gyro_bias = nanM33;
    const Eigen::Matrix3d Dv_D_accel_bias = nanM33;
    const Eigen::Matrix3d Dv_D_gyro_bias = nanM33;
    const Eigen::Matrix3d Dp_D_accel_bias = nanM33;
    const Eigen::Matrix3d Dp_D_gyro_bias = nanM33;

    const Eigen::Vector3d accel = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d gyro = sym::Random<Eigen::Vector3d>(gen);

    const double dt = 1.24;

    const Eigen::Vector3d accel_bias = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d gyro_bias = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d accel_cov = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d gyro_cov = sym::Random<Eigen::Vector3d>(gen);

    sym::Rot3d new_DR;
    Eigen::Vector3d new_Dv;
    Eigen::Vector3d new_Dp;

    sym::ImuManifoldPreintegrationUpdate<double>(
        DR, Dv, Dp, covariance, DR_D_gyro_bias, Dv_D_accel_bias, Dv_D_gyro_bias, Dp_D_accel_bias,
        Dp_D_gyro_bias, accel_bias, gyro_bias, accel_cov, gyro_cov, accel, gyro, dt,
        sym::kDefaultEpsilond, &new_DR, &new_Dv, &new_Dp, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr);

    sym::Rot3d expected_DR;
    Eigen::Vector3d expected_Dv;
    Eigen::Vector3d expected_Dp;

    UpdateState(DR, Dv, Dp, accel, gyro, accel_bias, gyro_bias, dt, sym::kDefaultEpsilond,
                expected_DR, expected_Dv, expected_Dp);

    CHECK(sym::IsClose(new_DR, expected_DR, 1e-14));
    CHECK(sym::IsClose(new_Dv, expected_Dv, 1e-14));
    CHECK(sym::IsClose(new_Dp, expected_Dp, 1e-14));
  }
}

/**
 * Helper class to generate samples from a multi-variate normal distribution with
 * the same covariance as that passed into the constructor.
 */
class MultiVarNormalDist {
 private:
  const Eigen::MatrixXd L;
  std::normal_distribution<double> dist;

 public:
  /**
   * covar is the covariance of the desired normal distribution.
   * Precondition: covar is a symmetric, positive definite matrix.
   */
  explicit MultiVarNormalDist(const Eigen::MatrixXd covar) : L{covar.llt().matrixL()}, dist{} {}

  /**
   * Returns a matrix whose (count) columns are samples from the distribution.
   */
  template <typename Generator>
  Eigen::MatrixXd Sample(Generator& gen, const int count) {
    Eigen::MatrixXd rand =
        Eigen::MatrixXd::NullaryExpr(L.cols(), count, [&]() { return dist(gen); });
    return L * rand;
  }
};

/**
 * Calculates the covariance of the columns of samples.
 */
Eigen::MatrixXd Covariance(const Eigen::MatrixXd& samples) {
  const size_t sample_size = samples.cols();
  const Eigen::VectorXd avg_col = samples.rowwise().sum() / sample_size;
  const Eigen::MatrixXd zero_mean_samples = samples.colwise() - avg_col;
  return (zero_mean_samples * zero_mean_samples.transpose()) / (sample_size - 1);
}

TEST_CASE("Test MultiVarNormalDist generates samples with correct covariance", "[slam]") {
  std::mt19937 gen(170);

  // Create covar, a symmetric positive definite matrix
  // NOTE(brad): This matrix is positive semi-definite because the diagonal entries all
  // have values of at least 2.0, and the max value of an off-diagonal entry is .2. Thus
  // within a row, the sum of the non-diagonal entries is capped by 0.2 * 8 = 1.6. Thus by the
  // Gershgorin circle theorem, any eigen value of covar is at least 0.4 (i.e., positive).
  std::uniform_real_distribution<double> uniform_dist(0.01, 0.1);
  Eigen::MatrixXd covar = Eigen::MatrixXd::NullaryExpr(9, 9, [&]() { return uniform_dist(gen); });
  covar += covar.transpose().eval();
  covar.diagonal().array() += 2.0;

  MultiVarNormalDist dist(covar);

  const Eigen::MatrixXd samples = dist.Sample(gen, 1 << 23);
  const Eigen::Matrix<double, 9, 9> calculated_covar = Covariance(samples);

  for (int col = 0; col < 9; col++) {
    for (int row = 0; row < 9; row++) {
      const double covar_rc = covar(row, col);
      const double difference = covar_rc - calculated_covar(row, col);
      // NOTE(brad): trade-off between sample count (time) and accuracy of calculated_covar
      CHECK(difference * difference < 9e-3 * covar_rc * covar_rc);
    }
  }
}

namespace example {
static const Eigen::Vector3d kAccelBias = {3.4, 1.6, -5.9};
static const Eigen::Vector3d kGyroBias = {1.2, -2.4, 0.5};
static const Eigen::Vector3d kAccelCov(7e-5, 7e-5, 7e-5);
static const Eigen::Vector3d kGyroCov(1e-3, 1e-3, 1e-3);
static const Eigen::Vector3d kTrueAccel = Eigen::Vector3d::Constant(4.3);
static const Eigen::Vector3d kTrueGyro = Eigen::Vector3d::Constant(10.2);
}  // namespace example

TEST_CASE("Test ImuPreintegrator.covariance", "[slam]") {
  // In order to test that the computed covariance is correct, we sample the values many
  // times to numerically calculate the covariance to compare.
  using M99 = Eigen::Matrix<double, 9, 9>;
  // Parameters
  const double dt = 1e-3;
  const int kMeasurementsPerSample = 70;
  const int kSampleCount = 1 << 18;

  // Multithreading
  const int kThreadCount = 1;
  const int kStepsPerThread = kSampleCount / kThreadCount;
  // NOTE(brad): I assume I can equally distribute samples among the threads
  CHECK(kSampleCount % kThreadCount == 0);
  std::vector<std::thread> threads;
  std::array<Eigen::Matrix<double, 9, 9>, kThreadCount> covariance_sums{};
  covariance_sums.fill(M99::Zero());

  Eigen::MatrixXd samples(9, kSampleCount);

  sym::ImuPreintegrator<double> noiseless_integrator(example::kAccelBias, example::kGyroBias);
  for (int k = 0; k < kMeasurementsPerSample; k++) {
    noiseless_integrator.IntegrateMeasurement(
        example::kAccelBias + example::kTrueAccel, example::kGyroBias + example::kTrueGyro,
        example::kAccelCov, example::kGyroCov, dt, sym::kDefaultEpsilond);
  }

  for (int i = 0; i < kThreadCount; i++) {
    threads.emplace_back([&samples, &covariance_sums, &noiseless_integrator, dt, i]() {
      std::mt19937 gen(45816 + i);
      // Each thread needs its own distribution as they are not thread safe
      MultiVarNormalDist accel_dist(example::kAccelCov.asDiagonal() * (1 / dt));
      MultiVarNormalDist gyro_dist(example::kGyroCov.asDiagonal() * (1 / dt));

      Eigen::MatrixXd accel_noise;
      Eigen::MatrixXd gyro_noise;

      for (int j = kStepsPerThread * i; j < kStepsPerThread * (i + 1); j++) {
        sym::ImuPreintegrator<double> integrator(example::kAccelBias, example::kGyroBias);

        accel_noise = accel_dist.Sample(gen, kMeasurementsPerSample);
        gyro_noise = gyro_dist.Sample(gen, kMeasurementsPerSample);

        for (int k = 0; k < kMeasurementsPerSample; k++) {
          integrator.IntegrateMeasurement(
              example::kAccelBias + example::kTrueAccel + accel_noise.col(k),
              example::kGyroBias + example::kTrueGyro + gyro_noise.col(k), example::kAccelCov,
              example::kGyroCov, dt, sym::kDefaultEpsilond);
        }

        const auto& noiseless_delta = noiseless_integrator.PreintegratedMeasurements().delta;
        const auto& delta = integrator.PreintegratedMeasurements().delta;

        samples.col(j).segment(0, 3) = noiseless_delta.DR.LocalCoordinates(delta.DR);
        samples.col(j).segment(3, 3) = delta.Dv;
        samples.col(j).segment(6, 3) = delta.Dp;

        covariance_sums[i] += integrator.Covariance();
      }
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }

  const M99 calculated_covariance =
      (1.0 / kSampleCount) *
      std::accumulate(covariance_sums.begin(), covariance_sums.end(), M99::Zero().eval());

  const Eigen::MatrixXd sampled_covariance = Covariance(samples);
  const double sampled_covariance_max = sampled_covariance.array().abs().maxCoeff();

  const Eigen::MatrixXd weighted_relative_error = sampled_covariance.binaryExpr(
      calculated_covariance, [=](const double x, const double y) -> double {
        // NOTE(brad): 6e-2 is partially arbitrary. Seemed like a reasonable value
        return std::abs((x - y) / (std::abs(x) + 6e-2 * sampled_covariance_max));
      });

  // NOTE(brad): 0.05 is also somewhat arbitrary. Seemed reasonable.
  CHECK(weighted_relative_error.maxCoeff() < 0.05);
}

Eigen::Matrix3d SkewSymmetric(const Ref<const Vector3d>& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

/**
 * The covariance that one integration step of duration dt adds to [DR, Dv, Dp], for white noise
 * with the given spectral densities, held measurements, and the rotation held at DR over the step.
 * This is the closed form documented in manifold_symbolic.py, written out block by block.
 */
Eigen::Matrix<double, 9, 9> AnalyticStepNoiseCovariance(const Rot3d& DR,
                                                        const Ref<const Vector3d>& corrected_accel,
                                                        const Ref<const Vector3d>& accel_cov,
                                                        const Ref<const Vector3d>& gyro_cov,
                                                        const double dt) {
  const Eigen::Matrix3d R = DR.ToRotationMatrix();
  const Eigen::Matrix3d A = -R * SkewSymmetric(corrected_accel);
  const Eigen::Matrix3d Sg = gyro_cov.asDiagonal();
  const Eigen::Matrix3d Sa = R * accel_cov.asDiagonal() * R.transpose();
  const Eigen::Matrix3d ASg = A * Sg;
  const Eigen::Matrix3d ASgAT = ASg * A.transpose();

  Eigen::Matrix<double, 9, 9> Q;
  Q.block<3, 3>(0, 0) = Sg * dt;
  Q.block<3, 3>(3, 0) = ASg * (dt * dt / 2);
  Q.block<3, 3>(6, 0) = ASg * (dt * dt * dt / 6);
  Q.block<3, 3>(3, 3) = Sa * dt + ASgAT * (dt * dt * dt / 3);
  Q.block<3, 3>(6, 3) = Sa * (dt * dt / 2) + ASgAT * (dt * dt * dt * dt / 8);
  Q.block<3, 3>(6, 6) = Sa * (dt * dt * dt / 3) + ASgAT * (dt * dt * dt * dt * dt / 20);
  Q.block<3, 3>(0, 3) = Q.block<3, 3>(3, 0).transpose();
  Q.block<3, 3>(0, 6) = Q.block<3, 3>(6, 0).transpose();
  Q.block<3, 3>(3, 6) = Q.block<3, 3>(6, 3).transpose();
  return Q;
}

TEST_CASE("Test ImuManifoldPreintegrationUpdate adds the analytic step noise covariance",
          "[slam]") {
  using M99 = Eigen::Matrix<double, 9, 9>;
  std::mt19937 gen(2718);

  for (int i_ = 0; i_ < 10; i_++) {
    const sym::Rot3d DR = sym::Rot3d::Random(gen);
    const Eigen::Vector3d Dv = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d Dp = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d accel = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d gyro = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d accel_bias = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d gyro_bias = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d accel_cov = sym::Random<Eigen::Vector3d>(gen).cwiseAbs();
    const Eigen::Vector3d gyro_cov = sym::Random<Eigen::Vector3d>(gen).cwiseAbs();
    const double dt = 0.37;

    // With zero input covariance, the output covariance is exactly the step's noise covariance
    M99 new_covariance;
    sym::ImuManifoldPreintegrationUpdate<double>(
        DR, Dv, Dp, M99::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
        Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), accel_bias,
        gyro_bias, accel_cov, gyro_cov, accel, gyro, dt, sym::kDefaultEpsilond, nullptr, nullptr,
        nullptr, &new_covariance, nullptr, nullptr, nullptr, nullptr, nullptr);

    const M99 expected =
        AnalyticStepNoiseCovariance(DR, accel - accel_bias, accel_cov, gyro_cov, dt);
    const M99 symmetric_new_covariance = new_covariance.selfadjointView<Eigen::Lower>();
    CHECK(sym::IsClose(symmetric_new_covariance, expected, 1e-12 * expected.norm()));
  }
}

TEST_CASE("Test ImuPreintegrator covariance is positive definite after one measurement", "[slam]") {
  using M99 = Eigen::Matrix<double, 9, 9>;
  const double dt = 5e-4;

  sym::ImuPreintegrator<double> integrator(example::kAccelBias, example::kGyroBias);
  integrator.IntegrateMeasurement(example::kAccelBias + example::kTrueAccel,
                                  example::kGyroBias + example::kTrueGyro, example::kAccelCov,
                                  example::kGyroCov, dt, sym::kDefaultEpsilond);

  const M99 covariance = integrator.Covariance().selfadjointView<Eigen::Lower>();
  const Eigen::LLT<M99> llt(covariance);
  CHECK(llt.info() == Eigen::Success);
  const Eigen::SelfAdjointEigenSolver<M99> eigensolver(covariance);
  CHECK(eigensolver.eigenvalues().minCoeff() > 0.0);

  const M99 expected = AnalyticStepNoiseCovariance(sym::Rot3d::Identity(), example::kTrueAccel,
                                                   example::kAccelCov, example::kGyroCov, dt);
  CHECK(sym::IsClose(covariance, expected, 1e-12 * expected.norm()));
}

TEST_CASE("Test ImuPreintegrator covariance does not depend on the step subdivision", "[slam]") {
  // With a held measurement, the integrated noise covariance of a window is the exact solution of
  // the linearized continuous-time model, so it is the same whether the window is integrated as
  // one step or as many. With no rotation rate this holds to machine precision; with rotation,
  // the step's rotation is held at its start, so the results agree to O(|gyro| dt).
  using M99 = Eigen::Matrix<double, 9, 9>;
  const double window = 0.2;
  const Eigen::Vector3d accel_cov(0.06 * 0.06, 0.06 * 0.06, 0.06 * 0.06);
  const Eigen::Vector3d gyro_cov(5e-4 * 5e-4, 5e-4 * 5e-4, 5e-4 * 5e-4);
  const Eigen::Vector3d accel(0.4, -0.7, 9.81);

  const auto integrate = [&](const Eigen::Vector3d& gyro, const int steps) -> M99 {
    sym::ImuPreintegrator<double> integrator(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    for (int k = 0; k < steps; k++) {
      integrator.IntegrateMeasurement(accel, gyro, accel_cov, gyro_cov, window / steps,
                                      sym::kDefaultEpsilond);
    }
    return integrator.Covariance().selfadjointView<Eigen::Lower>();
  };

  {
    const M99 one_step = integrate(Eigen::Vector3d::Zero(), 1);
    const M99 expected =
        AnalyticStepNoiseCovariance(sym::Rot3d::Identity(), accel, accel_cov, gyro_cov, window);
    CHECK(sym::IsClose(one_step, expected, 1e-12 * expected.norm()));
    for (const int steps : {2, 7, 40, 400}) {
      const M99 many_steps = integrate(Eigen::Vector3d::Zero(), steps);
      CAPTURE(steps);
      CHECK(sym::IsClose(many_steps, one_step, 1e-10 * one_step.norm()));
    }
  }

  {
    const Eigen::Vector3d gyro(0.3, -0.5, 0.4);
    const M99 coarse = integrate(gyro, 40);
    const M99 fine = integrate(gyro, 4000);
    CHECK(sym::IsClose(coarse, fine, 1e-6 * fine.norm()));
  }
}

TEST_CASE("Test preintegrated derivatives wrt IMU biases", "[slam]") {
  using M33 = Eigen::Matrix<double, 3, 3>;
  using M96 = Eigen::Matrix<double, 9, 6>;

  const double dt = 1e-3;
  const int iterations = 100;

  sym::ImuPreintegrator<double> integrator(example::kAccelBias, example::kGyroBias);

  for (int k_ = 0; k_ < iterations; k_++) {
    integrator.IntegrateMeasurement(example::kAccelBias + example::kTrueAccel,
                                    example::kGyroBias + example::kTrueGyro, example::kAccelCov,
                                    example::kGyroCov, dt, sym::kDefaultEpsilond);
  }

  Eigen::Matrix<double, 9, 6> state_D_bias =
      (Eigen::Matrix<double, 9, 6>() << integrator.PreintegratedMeasurements().DR_D_gyro_bias,
       M33::Zero(), integrator.PreintegratedMeasurements().Dv_D_gyro_bias,
       integrator.PreintegratedMeasurements().Dv_D_accel_bias,
       integrator.PreintegratedMeasurements().Dp_D_gyro_bias,
       integrator.PreintegratedMeasurements().Dp_D_accel_bias)
          .finished();

  // Perturb inputs and calculate derivatives numerically from them
  const double perturbation = 1e-5;
  M96 numerical_state_D_bias;
  for (int i = 0; i < 6; i++) {
    // Create biases and perturb one of their coefficients
    Eigen::Vector3d perturbed_accel_bias = example::kAccelBias;
    Eigen::Vector3d perturbed_gyro_bias = example::kGyroBias;
    if (i < 3) {
      perturbed_gyro_bias[i] += perturbation;
    } else {
      perturbed_accel_bias[i - 3] += perturbation;
    }

    sym::ImuPreintegrator<double> perturbed_integrator(perturbed_accel_bias, perturbed_gyro_bias);
    for (int k_ = 0; k_ < iterations; k_++) {
      perturbed_integrator.IntegrateMeasurement(
          example::kAccelBias + example::kTrueAccel, example::kGyroBias + example::kTrueGyro,
          example::kAccelCov, example::kGyroCov, dt, sym::kDefaultEpsilond);
    }

    const auto& delta = integrator.PreintegratedMeasurements().delta;
    const auto& perturbed_delta = perturbed_integrator.PreintegratedMeasurements().delta;

    numerical_state_D_bias.col(i).segment(0, 3) =
        delta.DR.LocalCoordinates(perturbed_delta.DR) / perturbation;
    numerical_state_D_bias.col(i).segment(3, 3) = (perturbed_delta.Dv - delta.Dv) / perturbation;
    numerical_state_D_bias.col(i).segment(6, 3) = (perturbed_delta.Dp - delta.Dp) / perturbation;
  }

  const auto passed = numerical_state_D_bias
                          .binaryExpr(state_D_bias,
                                      [](const double x, const double y) -> bool {
                                        if (std::abs(x - y) < 1e-10) {
                                          return true;
                                        }

                                        if (std::abs((x - y) / x) < 1e-6) {
                                          return true;
                                        }

                                        return false;
                                      })
                          .eval();

  CHECK(passed.all());
}

TEST_CASE("Verify handwritten and auto-derivative impls are equivalent", "[slam]") {
  // NOTE(Brad): To explain, some derivatives are needed in order to calculate the expressions
  // for the new covariance and the new derivatives of the state w.r.t. the IMU biases in the
  // symbolic formulation of ImuManifoldPreintegrationUpdate. These derivatives can be
  // be automatically computed using tangent_jacobians, but the expressions can also be obtained
  // by hand. For whatever reason (likely having to do with CSE) the handwritten expressions use
  // fewer ops. To verify that the handwritten implementations are correct, we compare them to
  // the version of ImuManifoldPreintegrationUpdate using the automatic derivatives.
  using M99 = Eigen::Matrix<double, 9, 9>;
  std::mt19937 gen(1104);

  for (int i_ = 0; i_ < 10; i_++) {
    // Generate sample inputs to preintegration update functions
    const sym::Rot3d DR = sym::Rot3d::Random(gen);
    const Eigen::Vector3d Dv = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d Dp = sym::Random<Eigen::Vector3d>(gen);

    const M99 covariance =
        (sym::Random<M99>(gen) + 10 * M99::Identity()).selfadjointView<Eigen::Lower>();
    const Eigen::Matrix3d DR_D_gyro_bias = sym::Random<Eigen::Matrix3d>(gen);
    const Eigen::Matrix3d Dv_D_accel_bias = sym::Random<Eigen::Matrix3d>(gen);
    const Eigen::Matrix3d Dv_D_gyro_bias = sym::Random<Eigen::Matrix3d>(gen);
    const Eigen::Matrix3d Dp_D_accel_bias = sym::Random<Eigen::Matrix3d>(gen);
    const Eigen::Matrix3d Dp_D_gyro_bias = sym::Random<Eigen::Matrix3d>(gen);

    const Eigen::Vector3d accel = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d gyro = sym::Random<Eigen::Vector3d>(gen);

    const double dt = 1.24;

    const Eigen::Vector3d accel_bias = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d gyro_bias = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d accel_cov = sym::Random<Eigen::Vector3d>(gen);
    const Eigen::Vector3d gyro_cov = sym::Random<Eigen::Vector3d>(gen);

    // Calculate outputs of handwritten derivative version
    Eigen::Matrix<double, 9, 9> new_covariance;
    Eigen::Matrix3d new_DR_D_gyro_bias;
    Eigen::Matrix3d new_Dv_D_accel_bias;
    Eigen::Matrix3d new_Dv_D_gyro_bias;
    Eigen::Matrix3d new_Dp_D_accel_bias;
    Eigen::Matrix3d new_Dp_D_gyro_bias;

    sym::ImuManifoldPreintegrationUpdate<double>(
        DR, Dv, Dp, covariance, DR_D_gyro_bias, Dv_D_accel_bias, Dv_D_gyro_bias, Dp_D_accel_bias,
        Dp_D_gyro_bias, accel_bias, gyro_bias, accel_cov, gyro_cov, accel, gyro, dt,
        sym::kDefaultEpsilond, nullptr, nullptr, nullptr, &new_covariance, &new_DR_D_gyro_bias,
        &new_Dv_D_accel_bias, &new_Dv_D_gyro_bias, &new_Dp_D_accel_bias, &new_Dp_D_gyro_bias);

    // Calculate outputs of auto derivative version
    Eigen::Matrix<double, 9, 9> new_covariance_auto;
    Eigen::Matrix3d new_DR_D_gyro_bias_auto;
    Eigen::Matrix3d new_Dv_D_accel_bias_auto;
    Eigen::Matrix3d new_Dv_D_gyro_bias_auto;
    Eigen::Matrix3d new_Dp_D_accel_bias_auto;
    Eigen::Matrix3d new_Dp_D_gyro_bias_auto;

    sym::ImuManifoldPreintegrationUpdateAutoDerivative<double>(
        DR, Dv, Dp, covariance, DR_D_gyro_bias, Dv_D_accel_bias, Dv_D_gyro_bias, Dp_D_accel_bias,
        Dp_D_gyro_bias, accel_bias, gyro_bias, accel_cov, gyro_cov, accel, gyro, dt,
        sym::kDefaultEpsilond, nullptr, nullptr, nullptr, &new_covariance_auto,
        &new_DR_D_gyro_bias_auto, &new_Dv_D_accel_bias_auto, &new_Dv_D_gyro_bias_auto,
        &new_Dp_D_accel_bias_auto, &new_Dp_D_gyro_bias_auto);

    // Verify they're equivalent
    CHECK(sym::IsClose(new_covariance, new_covariance_auto, 1e-8));
    CHECK(sym::IsClose(new_DR_D_gyro_bias, new_DR_D_gyro_bias_auto, 1e-8));
    CHECK(sym::IsClose(new_Dv_D_accel_bias, new_Dv_D_accel_bias_auto, 1e-8));
    CHECK(sym::IsClose(new_Dv_D_gyro_bias, new_Dv_D_gyro_bias_auto, 1e-8));
    CHECK(sym::IsClose(new_Dp_D_accel_bias, new_Dp_D_accel_bias_auto, 1e-8));
    CHECK(sym::IsClose(new_Dp_D_gyro_bias, new_Dp_D_gyro_bias_auto, 1e-8));
  }
}

TEST_CASE("Verify bias-corrected delta", "[slam]") {
  // This test checks that if we integrate the same IMU data with two slightly different imu biases,
  // then we can use the first-order Jacobian to correct the pre-integrated delta.

  Eigen::Vector3d accel_cov{0.1, 0.1, 0.1};
  Eigen::Vector3d gyro_cov{0.1, 0.1, 0.1};

  // True imu biases
  Eigen::Vector3d accel_bias0{0, 0, 0};
  Eigen::Vector3d gyro_bias0{0, 0, 0};

  // Slightly different imu biases than the true one
  Eigen::Vector3d accel_bias1{0.01, 0.01, 0.01};
  Eigen::Vector3d gyro_bias1{0.01, 0.01, 0.01};

  // preint0 will integrate with bias0
  sym::ImuPreintegrator<double> preint0(accel_bias0, gyro_bias0);
  // preint1 will integrate with bias1
  sym::ImuPreintegrator<double> preint1(accel_bias1, gyro_bias1);

  constexpr double dt = 0.005;

  for (int i = 0; i < 100; ++i) {
    Eigen::Vector3d accel = Eigen::Vector3d::Random();
    Eigen::Vector3d gyro = Eigen::Vector3d::Random();

    // Integrate both
    preint0.IntegrateMeasurement(accel, gyro, accel_cov, gyro_cov, dt);
    preint1.IntegrateMeasurement(accel, gyro, accel_cov, gyro_cov, dt);
  }

  const auto& pim0 = preint0.PreintegratedMeasurements();
  const auto& pim1 = preint1.PreintegratedMeasurements();

  // original delta1 (this should be the same as pim1)
  const auto delta1 =
      preint1.PreintegratedMeasurements().GetBiasCorrectedDelta(accel_bias1, gyro_bias1);

  // corrected delta1
  const auto delta1_corrected =
      preint1.PreintegratedMeasurements().GetBiasCorrectedDelta(accel_bias0, gyro_bias0);

  const auto& pim0_delta = pim0.delta;
  const auto& pim1_delta = pim1.delta;

  CAPTURE(pim0_delta.Dp.transpose(), pim0_delta.Dv.transpose(), pim0_delta.DR);
  CAPTURE(pim1_delta.Dp.transpose(), pim1_delta.Dv.transpose(), pim1_delta.DR);
  CAPTURE(delta1.Dp.transpose(), delta1.Dv.transpose(), delta1.DR);
  CAPTURE(delta1_corrected.Dp.transpose(), delta1_corrected.Dv.transpose(), delta1_corrected.DR);

  constexpr double kTol = 1e-4;
  CHECK(pim1_delta.Dp.isApprox(delta1.Dp, kTol));
  CHECK(pim1_delta.Dv.isApprox(delta1.Dv, kTol));
  CHECK(sym::LieGroupOps<sym::Rot3d>::IsClose(pim1_delta.DR, delta1.DR, kTol));

  // Check that pim1_delta is not close to the true pim0_delta
  CHECK(!pim0_delta.Dp.isApprox(delta1.Dp, kTol));
  CHECK(!pim0_delta.Dv.isApprox(delta1.Dv, kTol));
  CHECK(!sym::LieGroupOps<sym::Rot3d>::IsClose(pim0_delta.DR, delta1.DR, kTol));

  // Check that the corrected delta1 is sufficiently close to the true delta (pim0_delta)
  CHECK(pim0_delta.Dp.isApprox(delta1_corrected.Dp, kTol));
  CHECK(pim0_delta.Dv.isApprox(delta1_corrected.Dv, kTol));
  CHECK(sym::LieGroupOps<sym::Rot3d>::IsClose(pim0_delta.DR, delta1_corrected.DR, kTol));
}

TEST_CASE("Test RollForwardState", "[slam]") {
  const double dt = 1e-3;
  const int iterations = 100;

  sym::ImuPreintegrator<double> integrator(example::kAccelBias, example::kGyroBias);

  for (int k_ = 0; k_ < iterations; k_++) {
    integrator.IntegrateMeasurement(example::kAccelBias + example::kTrueAccel,
                                    example::kGyroBias + example::kTrueGyro, example::kAccelCov,
                                    example::kGyroCov, dt, sym::kDefaultEpsilond);
  }

  std::mt19937 gen{42};

  const auto p0 = sym::Pose3d::Random(gen);
  const auto v0 = sym::Random<Vector3d>(gen);

  const auto [p1, v1] =
      integrator.PreintegratedMeasurements().delta.RollForwardState(p0, v0, Vector3d::Zero());

  const auto factor = sym::ImuFactord{integrator};
  sym::Vector9d residual;
  factor(p0, v0, p1, v1, example::kAccelBias, example::kGyroBias, Vector3d::Zero(),
         sym::kDefaultEpsilond, &residual);

  CHECK(residual.norm() < 1e-10);
}

TEMPLATE_TEST_CASE("Test StorageOps", "[slam]", float, double) {
  using Scalar = TestType;

  std::mt19937 gen{42};
  const auto accel_bias = sym::Random<sym::Vector3<Scalar>>(gen);
  const auto gyro_bias = sym::Random<sym::Vector3<Scalar>>(gen);
  sym::PreintegratedImuMeasurements<Scalar> pim(accel_bias, gyro_bias);
  const auto Dt = sym::Random<Scalar>(gen);
  const auto DR = sym::Random<sym::Rot3<Scalar>>(gen);
  const auto Dv = sym::Random<sym::Vector3<Scalar>>(gen);
  const auto Dp = sym::Random<sym::Vector3<Scalar>>(gen);
  pim.delta = typename sym::PreintegratedImuMeasurements<Scalar>::Delta{Dt, DR, Dv, Dp};
  pim.DR_D_gyro_bias = sym::Random<sym::Matrix33<Scalar>>(gen);
  pim.Dv_D_accel_bias = sym::Random<sym::Matrix33<Scalar>>(gen);
  pim.Dv_D_gyro_bias = sym::Random<sym::Matrix33<Scalar>>(gen);
  pim.Dp_D_gyro_bias = sym::Random<sym::Matrix33<Scalar>>(gen);
  pim.Dp_D_accel_bias = sym::Random<sym::Matrix33<Scalar>>(gen);

  {
    static_assert(sym::PreintegratedImuMeasurements<Scalar>::StorageDim() ==
                  sym::StorageOps<sym::PreintegratedImuMeasurements<Scalar>>::StorageDim());
  }

  {
    std::array<Scalar, sym::PreintegratedImuMeasurements<Scalar>::StorageDim()> storage{};
    pim.ToStorage(storage.data());
    const auto pim2 = sym::PreintegratedImuMeasurements<Scalar>::FromStorage(storage.data());

    CHECK(pim2.GetLcmType() == pim.GetLcmType());
  }

  {
    std::array<Scalar, sym::PreintegratedImuMeasurements<Scalar>::StorageDim()> storage{};
    sym::StorageOps<sym::PreintegratedImuMeasurements<Scalar>>::ToStorage(pim, storage.data());
    const auto pim2 =
        sym::StorageOps<sym::PreintegratedImuMeasurements<Scalar>>::FromStorage(storage.data());

    CHECK(pim2.GetLcmType() == pim.GetLcmType());
  }

  {
    sym::Values<Scalar> v;
    v.Set('p', pim);

    const auto pim2 = v.template At<sym::PreintegratedImuMeasurements<Scalar>>('p');
    CHECK(pim2.GetLcmType() == pim.GetLcmType());
  }
}
