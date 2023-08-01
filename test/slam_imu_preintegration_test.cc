/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <limits>
#include <numeric>
#include <random>
#include <thread>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include <sym/factors/internal/imu_manifold_preintegration_update.h>
#include <sym/factors/internal/imu_manifold_preintegration_update_auto_derivative.h>
#include <sym/rot3.h>
#include <sym/util/epsilon.h>
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
  std::array<Eigen::Matrix<double, 9, 9>, kThreadCount> covariance_sums;
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

  const Eigen::MatrixXd relative_error =
      numerical_state_D_bias.binaryExpr(state_D_bias, [](const double x, const double y) -> double {
        return std::abs((x - y) / (std::abs(x) + 1e-10));
      });

  CHECK(relative_error.maxCoeff() < 0.05);
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
