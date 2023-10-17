/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <stdint.h>
#include <sys/time.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sym/atan_camera_cal.h>
#include <sym/double_sphere_camera_cal.h>
#include <sym/equirectangular_camera_cal.h>
#include <sym/linear_camera_cal.h>
#include <sym/ops/lie_group_ops.h>
#include <sym/polynomial_camera_cal.h>
#include <sym/rot3.h>
#include <sym/spherical_camera_cal.h>
#include <sym/unit3.h>
#include <sym/util/epsilon.h>
#include <symforce/opt/util.h>
#include <symforce/opt/values.h>

TEMPLATE_TEST_CASE("Test values", "[values]", double, float) {
  using Scalar = TestType;

  INFO("Testing Values: " << typeid(Scalar).name());

  sym::Values<Scalar> v;
  CHECK(v.Keys().size() == 0);
  CHECK(v.NumEntries() == 0);
  CHECK(v.Items().size() == 0);
  CHECK(v.Data().size() == 0);
  CHECK(!v.Has(sym::Key('F', -1, 3)));

  // Add a key
  sym::Key R1_key('R', 1);
  sym::Rot3<Scalar> R1 = sym::Rot3<Scalar>::FromYawPitchRoll(0.5, -0.2, 0.1);
  const bool is_new = v.Set(R1_key, R1);
  CHECK(is_new);
  CHECK(v.NumEntries() == 1);
  CHECK(v.Keys().size() == 1);
  CHECK(v.Items().size() == 1);
  CHECK(v.Data().size() == R1.StorageDim());
  CHECK(v.Has(R1_key));
  sym::Rot3<Scalar> R1_fetch = v.template At<sym::Rot3<Scalar>>(R1_key);
  CHECK(R1 == R1_fetch);

  // Add a second
  sym::Key z1_key = sym::Key('z', 1);
  Scalar s = 2.0;
  v.Set(z1_key, s);
  CHECK(v.NumEntries() == 2);
  CHECK(v.Data().size() == R1.StorageDim() + 1);
  CHECK(v.Has(z1_key));
  CHECK(v.Has(R1_key));
  CHECK(s == v.template At<Scalar>(z1_key));

  // Modify a key
  const sym::Rot3<Scalar> R1_new = sym::Rot3<Scalar>::FromTangent({1.2, 0.2, 0.0});
  const bool is_new2 = v.Set(R1_key, R1_new);
  CHECK_FALSE(is_new2);
  CHECK(v.NumEntries() == 2);
  CHECK(v.Data().size() == R1.StorageDim() + 1);
  CHECK(R1_new == v.template At<sym::Rot3<Scalar>>(R1_key));

  // Remove nothing
  bool remove_nothing = v.Remove(sym::Key('f'));
  CHECK_FALSE(remove_nothing);

  // Remove z1
  bool remove_z1 = v.Remove(z1_key);
  CHECK(remove_z1);
  CHECK(v.NumEntries() == 1);
  CHECK(v.Data().size() == R1.StorageDim() + 1);
  CHECK_FALSE(v.Has(z1_key));

  // Add some more
  v.Set(sym::Key('f', 1), Scalar(4.2));
  v.Set(sym::Key('f', 2), Scalar(4.3));
  v.Set(sym::Key('d', 1), Scalar(4.3));
  v.Set(sym::Key('v', 1), Eigen::Matrix<Scalar, 1, 1>(0.0));
  v.Set(sym::Key('v', 3), Eigen::Matrix<Scalar, 3, 1>(1.0, 2.0, 3.0));
  v.template Set<Eigen::Matrix<Scalar, 9, 1>>(sym::Key('v', 9),
                                              Eigen::Matrix<Scalar, 9, 1>::Constant(1.2));

  // Right now since we removed a scalar the data array is one longer than the actual storage dim
  CHECK(v.NumEntries() == 7);
  const sym::index_t index_1 = v.CreateIndex(v.Keys());
  CHECK(static_cast<int>(v.Data().size()) == index_1.storage_dim + 1);
  CHECK(index_1.tangent_dim == index_1.storage_dim - 1);

  // Cleanup to get rid of the empty space from the scalar
  v.Cleanup();
  CHECK(v.NumEntries() == 7);
  CHECK(static_cast<int>(v.Data().size()) == index_1.storage_dim);
  const sym::index_t index_2 = v.CreateIndex(v.Keys());
  CHECK(R1_new == v.template At<sym::Rot3<Scalar>>(R1_key));
  CHECK(Scalar(4.2) == v.template At<Scalar>({'f', 1}));
  CHECK(index_2.storage_dim == index_1.storage_dim);
  CHECK(index_2.tangent_dim == index_1.tangent_dim);

  // Test lookups
  const sym::index_entry_t& f2_entry = index_1.entries[1];
  CHECK(v.template At<Scalar>(f2_entry) == v.template At<Scalar>(sym::Key('f', 2)));
  v.Set(f2_entry, Scalar(15.6));
  CHECK(v.template At<Scalar>(sym::Key('f', 2)) == Scalar(15.6));

  // Pack to LCM
  const typename sym::Values<Scalar>::LcmType msg = v.GetLcmType(true /*sort keys*/);
  CHECK(msg.index == index_2);
  CHECK(msg.data.size() == v.Data().size());

  // Recreate another
  sym::Values<Scalar> v2(msg);
  CHECK(v2.NumEntries() == v.NumEntries());
  CHECK(v2.Data() == v.Data());
  CHECK(v.CreateIndex(v.Keys()) == v2.CreateIndex(v2.Keys()));

  // Print
  CAPTURE(v);

  // Clear
  v.RemoveAll();
  CHECK(v.NumEntries() == 0);
  CHECK(v.Data().size() == 0);
}

TEMPLATE_PRODUCT_TEST_CASE("Test storage and retrieval of camera cals", "[values]",
                           (sym::LinearCameraCal, sym::ATANCameraCal, sym::SphericalCameraCal,
                            sym::EquirectangularCameraCal, sym::PolynomialCameraCal,
                            sym::DoubleSphereCameraCal),
                           (double, float)) {
  using T = TestType;
  using Scalar = typename sym::StorageOps<T>::Scalar;

  sym::Values<Scalar> values;

  const T camera_cal = sym::GroupOps<T>::Identity();
  values.Set('a', camera_cal);
  CHECK(camera_cal == values.template At<T>('a'));
}

// TODO(brad): SphericalCameraCal and PolynomialCameraCal should fail this test.
TEMPLATE_PRODUCT_TEST_CASE("Test Retract and LocalCoordinates of camera cals", "[values]",
                           (sym::LinearCameraCal, sym::ATANCameraCal, sym::SphericalCameraCal,
                            sym::EquirectangularCameraCal, sym::PolynomialCameraCal,
                            sym::DoubleSphereCameraCal),
                           (double, float)) {
  using T = TestType;
  using Scalar = typename sym::StorageOps<T>::Scalar;

  sym::Values<Scalar> values;

  const T camera_cal = sym::GroupOps<T>::Identity();
  values.Set('a', camera_cal);
  const sym::index_t index = values.CreateIndex({'a'});
  const auto tangent_vec = sym::LieGroupOps<T>::ToTangent(camera_cal, sym::kDefaultEpsilond);
  values.Retract(index, tangent_vec.data(), sym::kDefaultEpsilond);

  sym::Values<Scalar> values2 = values;
  sym::VectorX<Scalar> vec = values.LocalCoordinates(values2, index, sym::kDefaultEpsilond);
}

TEST_CASE("Test IndexEntryAt", "[values]") {
  sym::Valuesd values;
  const sym::Key k1 = sym::Key('k', 1);
  const sym::Key k2 = sym::Key('k', 2);
  const sym::Key k3 = sym::Key('k', 3);
  values.Set<double>(k1, 1.0);
  values.Set<double>(k2, 2.0);

  const sym::index_entry_t entry2 = values.IndexEntryAt(k2);

  // Can be used with At to access an entry
  CHECK(values.At<double>(entry2) == 2.0);

  // Entry remains valid when a value is added afterwards
  values.Set<double>(k3, 3.0);
  CHECK(values.At<double>(entry2) == 2.0);

  // Entry remains valid when a value added before it is removed
  values.Remove(k1);
  CHECK(values.At<double>(entry2) == 2.0);

  // Entry remains valid when the value is re-set
  values.Set<double>(k2, 4.0);
  CHECK(values.At<double>(entry2) == 4.0);

  // Entry remains valid even when updated with UpdateOrSet
  sym::Valuesd other_values;
  other_values.Set<double>(k1, -1.0);
  other_values.Set<double>(k2, -2.0);
  other_values.Set<double>(k3, -3.0);
  values.UpdateOrSet(other_values.CreateIndex({k1, k2, k3}), other_values);
  CHECK(values.At<double>(entry2) == -2.0);
}

TEST_CASE("Test implicit construction", "[values]") {
  sym::Valuesd values;
  values.Set<double>('x', 1.0);
  values.Set<double>('y', 2.0);
  values.Set<double>('z', -3.0);
  values.Set<sym::Rot3d>({'R', 1}, sym::Rot3d::Identity());
  values.Set<sym::Rot3d>({'R', 2}, sym::Rot3d::FromYawPitchRoll(1.0, 0.0, 0.0));
  values.Set<sym::Pose3d>('P', sym::Pose3d::Identity());
  values.Set<sym::Unit3d>('R', sym::Unit3d::Identity());
  CAPTURE(values);
}

TEST_CASE("Test initializer list construction", "[values]") {
  sym::Valuesd v1;
  v1.Set<double>('x', 1.0);
  v1.Set<double>('y', 2.0);
  v1.Set<double>('z', -3.0);

  sym::Valuesd v2;
  v2.Set<sym::Rot3d>('R', sym::Rot3d::Identity());

  // construct v3 by merging v1 and v2
  sym::Valuesd v3({v1, v2});

  // test data
  CHECK(v3.At<double>('x') == 1.0);
  CHECK(v3.At<double>('y') == 2.0);
  CHECK(v3.At<double>('z') == -3.0);
  CHECK(v3.At<sym::Rot3d>('R') == sym::Rot3d::Identity());

  // test preserving key ordering
  const auto v3_keys = v3.Keys();
  CHECK(v3_keys[0] == 'x');
  CHECK(v3_keys[1] == 'y');
  CHECK(v3_keys[2] == 'z');
  CHECK(v3_keys[3] == 'R');
}

TEST_CASE("Test indexed update", "[values]") {
  // Create some data
  sym::Valuesd values;
  values.Set<double>('x', 1.0);
  values.Set<double>('y', 2.0);
  values.Set<double>('z', -3.0);
  values.Set<sym::Rot3d>({'R', 1}, sym::Rot3d::Identity());
  values.Set<sym::Rot3d>({'R', 2}, sym::Rot3d::FromYawPitchRoll(1.0, 0.0, 0.0));
  values.Set<sym::Pose3d>('P', sym::Pose3d::Identity());

  // Create an index for a subset of keys
  const sym::index_t index = values.CreateIndex({'x', 'y', {'R', 1}});

  // Copy into another Values
  sym::Valuesd values2 = values;

  // Modify some keys in the original
  values.Set<double>('x', 7.7);
  values.Set<sym::Rot3d>({'R', 1}, values.At<sym::Rot3d>({'R', 2}));

  CHECK(values.At<double>('x') == 7.7);
  CHECK(values2.At<double>('x') == 1.0);

  // Efficiently update keys into the new values
  values2.Update(index, values);

  CHECK(values2.At<double>('x') == 7.7);
}

TEST_CASE("Test key update", "[values]") {
  // Create some data
  sym::Valuesd values;
  values.Set<double>('x', 1.0);
  values.Set<double>('y', 2.0);
  values.Set<double>('z', -3.0);
  values.Set<sym::Rot3d>({'R', 1}, sym::Rot3d::Identity());
  values.Set<sym::Rot3d>({'R', 2}, sym::Rot3d::FromYawPitchRoll(1.0, 0.0, 0.0));
  values.Set<sym::Pose3d>('P', sym::Pose3d::Identity());

  // Create an index for a subset of keys (random order should be supported)
  const std::vector<sym::Key> keys = {{'R', 1}, 'x', 'y'};
  const sym::index_t index = values.CreateIndex(keys);

  // Another different structured Values
  sym::Valuesd values2;
  values2.Set<double>('z', 10.0);

  // Update from a different structure
  values2.UpdateOrSet(index, values);

  // Test for update
  CHECK(values2.At<double>('x') == 1.0);
  CHECK(values2.At<double>('y') == 2.0);
  CHECK(values2.At<sym::Rot3d>({'R', 1}) == sym::Rot3d::Identity());

  // Test for not clobbering other field
  CHECK(values2.At<double>('z') == 10.0);

  // Test efficient update with cached index
  const sym::index_t index2 = values2.CreateIndex(keys);
  values.Set<double>('x', -10.0);
  values.Set<double>('y', 20.0);
  values2.Update(index2, index, values);
  CHECK(values2.At<double>('x') == -10.0);
  CHECK(values2.At<double>('y') == 20.0);
}

TEMPLATE_PRODUCT_TEST_CASE("Test lie group ops", "[values]",
                           (sym::Rot2, sym::Pose2, sym::Rot3, sym::Pose3, sym::Unit3, sym::Vector1,
                            sym::Vector3, sym::Vector9, sym::Matrix11, sym::Matrix33, sym::Matrix99,
                            sym::Matrix34),
                           (double, float)) {
  using T = TestType;
  using Scalar = typename sym::StorageOps<T>::Scalar;

  INFO("Testing Values " << typeid(Scalar).name() << " LieGroupOps with " << typeid(T).name());
  constexpr Scalar epsilon = sym::kDefaultEpsilon<Scalar>;
  const Scalar tolerance = epsilon * 1000;
  CAPTURE(epsilon, tolerance);

  // Create a values object that stores an identity element, and an index for it
  sym::Values<Scalar> v1;
  const T element = sym::GroupOps<T>::Identity();
  v1.Set('x', element);
  const sym::index_t index = v1.CreateIndex({'x'});

  // Test a bunch of retractions and local coordinates
  std::mt19937 gen(42);
  for (int i = 0; i < 100; ++i) {
    v1.Set('x', element);
    const sym::Values<Scalar> v2 = v1;

    const T random_element = sym::StorageOps<T>::Random(gen);
    CAPTURE(random_element);
    const auto tangent_vec = sym::LieGroupOps<T>::ToTangent(random_element, epsilon);
    CAPTURE(tangent_vec.transpose());

    // test retraction
    v1.Retract(index, tangent_vec.data(), epsilon);
    const T retracted_element = v1.template At<T>('x');
    CAPTURE(retracted_element);
    CHECK(sym::IsClose(random_element, retracted_element, tolerance));

    // test local coordinates
    const sym::VectorX<Scalar> local_coords = v1.LocalCoordinates(v2, index, epsilon);
    CHECK(sym::IsClose<sym::VectorX<Scalar>>(local_coords, tangent_vec, tolerance));
  }
}

TEST_CASE("Test move operator", "[values]") {
  static_assert(std::is_move_assignable<sym::Values<float>>::value, "");
  sym::Valuesf values;
  values.Set<float>('x', 1.0f);
  values.Set<float>('y', 2.0f);
  values.Set<sym::Rot3f>({'R', 1}, sym::Rot3f::Identity());
  sym::Valuesf values2 = std::move(values);
  CHECK(values2.At<float>('x') == 1.0f);
}

TEST_CASE("Test Set with Eigen expressions", "[values]") {
  sym::Valuesd values;
  values.Set('a', Eigen::Vector3d::Zero());
  values.Set('b', Eigen::Vector3f::Zero().cast<double>());
  values.Set('c', Eigen::Vector3d::Zero() + 2 * Eigen::Vector3d::Ones());
  values.Set('d', Eigen::Vector3d(Eigen::Vector3d::Zero()));
  values.Set('e', sym::Rot3d::FromAngleAxis(1, Eigen::Vector3d::Ones()).ToRotationMatrix());
  CHECK(values.At<Eigen::Vector3d>('a') == Eigen::Vector3d::Zero());
  CHECK(values.At<Eigen::Vector3d>('b') == Eigen::Vector3d::Zero());
  CHECK(values.At<Eigen::Vector3d>('c') == Eigen::Vector3d::Constant(2));
  CHECK(values.At<Eigen::Vector3d>('d') == Eigen::Vector3d::Zero());
  CHECK(values.At<Eigen::Matrix3d>('e') ==
        sym::Rot3d::FromAngleAxis(1, Eigen::Vector3d::Ones()).ToRotationMatrix());

  sym::index_t index = values.CreateIndex({'a', 'b', 'c', 'd', 'e'});
  values.Set(index.entries[4],
             sym::Rot3d::FromAngleAxis(0.3, Eigen::Vector3d::Ones()).ToRotationMatrix());
  values.Set(index.entries[3], Eigen::Vector3d::Zero());
  values.Set(index.entries[2], Eigen::Vector3f::Zero().cast<double>());
  values.Set(index.entries[1], Eigen::Vector3d::Zero() + 2 * Eigen::Vector3d::Ones());
  values.Set(index.entries[0], Eigen::Vector3d(Eigen::Vector3d::Zero()));
  CHECK(values.At<Eigen::Matrix3d>('e') ==
        sym::Rot3d::FromAngleAxis(0.3, Eigen::Vector3d::Ones()).ToRotationMatrix());
  CHECK(values.At<Eigen::Vector3d>('d') == Eigen::Vector3d::Zero());
  CHECK(values.At<Eigen::Vector3d>('c') == Eigen::Vector3d::Zero());
  CHECK(values.At<Eigen::Vector3d>('b') == Eigen::Vector3d::Constant(2));
  CHECK(values.At<Eigen::Vector3d>('a') == Eigen::Vector3d::Zero());
}

TEST_CASE("Test SetNew", "[values]") {
  sym::Valuesd values;
  values.SetNew('a', Eigen::Vector3d::Zero());
  CHECK(values.At<Eigen::Vector3d>('a') == Eigen::Vector3d::Zero());
  CHECK_THROWS_AS(values.SetNew('a', Eigen::Vector3d::Zero()), std::runtime_error);
}

TEST_CASE("Test MaybeIndexEntryAt", "[values]") {
  sym::Valuesd values;
  CHECK(values.MaybeIndexEntryAt('a') == sym::optional<sym::index_entry_t>{});
  values.Set('a', 1.0);
  CHECK(values.MaybeIndexEntryAt('a') != sym::optional<sym::index_entry_t>{});
}
