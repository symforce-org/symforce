#include <iostream>
#include <stdint.h>
#include <sys/time.h>

#include <geo/rot3.h>

#include "../symforce/opt/values.h"

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

template <typename Scalar>
void TestValues() {
  std::cout << "*** Testing Values<" << typeid(Scalar).name() << "> ***" << std::endl;
  const Scalar epsilon = 1e-9;

  sym::Values<Scalar> v;
  assertTrue(v.Keys().size() == 0);
  assertTrue(v.NumEntries() == 0);
  assertTrue(v.Items().size() == 0);
  assertTrue(v.Data().size() == 0);
  assertTrue(!v.Has(sym::Key()));
  assertTrue(!v.Has(sym::Key('F', -1, 3)));

  // Add a key
  sym::Key R1_key('R', 1);
  geo::Rot3<Scalar> R1 = geo::Rot3<Scalar>::FromYawPitchRoll(0.5, -0.2, 0.1);
  const bool is_new = v.Set(R1_key, R1);
  assertTrue(is_new);
  assertTrue(v.NumEntries() == 1);
  assertTrue(v.Keys().size() == 1);
  assertTrue(v.Items().size() == 1);
  assertTrue(v.Data().size() == R1.StorageDim());
  assertTrue(v.Has(R1_key));
  geo::Rot3<Scalar> R1_fetch = v.template At<geo::Rot3<Scalar>>(R1_key);
  assertTrue(R1 == R1_fetch);

  // Add a second
  sym::Key z1_key = sym::Key('z', 1);
  Scalar s = 2.0;
  v.Set(z1_key, s);
  assertTrue(v.NumEntries() == 2);
  assertTrue(v.Data().size() == R1.StorageDim() + 1);
  assertTrue(v.Has(z1_key));
  assertTrue(v.Has(R1_key));
  assertTrue(s == v.template At<Scalar>(z1_key));

  // Modify a key
  const geo::Rot3<Scalar> R1_new = geo::Rot3<Scalar>::FromTangent({1.2, 0.2, 0.0});
  const bool is_new2 = v.Set(R1_key, R1_new);
  assertTrue(!is_new2);
  assertTrue(v.NumEntries() == 2);
  assertTrue(v.Data().size() == R1.StorageDim() + 1);
  assertTrue(R1_new == v.template At<geo::Rot3<Scalar>>(R1_key));

  // Remove nothing
  bool remove_nothing = v.Remove(sym::Key('f'));
  assertTrue(!remove_nothing);

  // Remove z1
  bool remove_z1 = v.Remove(z1_key);
  assertTrue(remove_z1);
  assertTrue(v.NumEntries() == 1);
  assertTrue(v.Data().size() == R1.StorageDim() + 1);
  assertTrue(!v.Has(z1_key));

  // Add some more
  v.Set(sym::Key('f', 1), Scalar(4.2));
  v.Set(sym::Key('f', 2), Scalar(4.3));
  v.Set(sym::Key('d', 1), Scalar(4.3));
  v.Set(sym::Key('v', 1), Eigen::Matrix<Scalar, 1, 1>(0.0));
  v.Set(sym::Key('v', 3), Eigen::Matrix<Scalar, 3, 1>(1.0, 2.0, 3.0));
  v.template Set<Eigen::Matrix<Scalar, 9, 1>>(sym::Key('v', 9),
                                              Eigen::Matrix<Scalar, 9, 1>::Constant(1.2));

  // Right now since we removed a scalar the data array is one longer than the actual storage dim
  assertTrue(v.NumEntries() == 7);
  const sym::index_t index_1 = v.CreateIndex(v.Keys());
  assertTrue(v.Data().size() == index_1.storage_dim + 1);
  assertTrue(index_1.tangent_dim == index_1.storage_dim - 1);

  // Cleanup to get rid of the empty space from the scalar
  size_t num_cleaned = v.Cleanup();
  assertTrue(v.NumEntries() == 7);
  assertTrue(v.Data().size() == index_1.storage_dim);
  const sym::index_t index_2 = v.CreateIndex(v.Keys());
  assertTrue(R1_new == v.template At<geo::Rot3<Scalar>>(R1_key));
  assertTrue(Scalar(4.2) == v.template At<Scalar>({'f', 1}));
  assertTrue(index_2.storage_dim == index_1.storage_dim);
  assertTrue(index_2.tangent_dim == index_1.tangent_dim);

  // Test lookups
  const sym::index_entry_t& f2_entry = index_1.entries[1];
  assertTrue(v.template At<Scalar>(f2_entry) == v.template At<Scalar>(sym::Key('f', 2)));
  v.Set(f2_entry, Scalar(15.6));
  assertTrue(v.template At<Scalar>(sym::Key('f', 2)) == Scalar(15.6));

  // Pack to LCM
  const typename sym::Values<Scalar>::LcmType msg = v.GetLcmType();
  assertTrue(msg.index == index_2);
  assertTrue(msg.data.size() == v.Data().size());

  // Recreate another
  sym::Values<Scalar> v2(msg);
  assertTrue(v2.NumEntries() == v.NumEntries());
  assertTrue(v2.Data() == v.Data());
  assert(v.CreateIndex(v.Keys()) == v2.CreateIndex(v2.Keys()));

  // Print
  std::cout << "v: " << v << std::endl;

  // Clear
  v.RemoveAll();
  assertTrue(v.NumEntries() == 0);
  assertTrue(v.Data().size() == 0);
}

void TestImplicitConstruction() {
  std::cout << "*** Testing Values Implicit Construction ***" << std::endl;

  sym::Valuesd values;
  values.Set<double>('x', 1.0);
  values.Set<double>('y', 2.0);
  values.Set<double>('z', -3.0);
  values.Set<geo::Rot3d>({'R', 1}, geo::Rot3d::Identity());
  values.Set<geo::Rot3d>({'R', 2}, geo::Rot3d::FromYawPitchRoll(1.0, 0.0, 0.0));
  values.Set<geo::Pose3d>('P', geo::Pose3d::Identity());
  std::cout << values << std::endl;
}

void TestInitializerListConstruction() {
  std::cout << "*** Testing Values Initializer List Construction ***" << std::endl;

  sym::Valuesd v1;
  v1.Set<double>('x', 1.0);
  v1.Set<double>('y', 2.0);
  v1.Set<double>('z', -3.0);

  sym::Valuesd v2;
  v2.Set<geo::Rot3d>('R', geo::Rot3d::Identity());

  // construct v3 by merging v1 and v2
  sym::Valuesd v3({v1, v2});

  // test data
  assertTrue(v3.At<double>('x') == 1.0);
  assertTrue(v3.At<double>('y') == 2.0);
  assertTrue(v3.At<double>('z') == -3.0);
  assertTrue(v3.At<geo::Rot3d>('R') == geo::Rot3d::Identity());

  // test preserving key ordering
  const auto v3_keys = v3.Keys();
  assertTrue(v3_keys[0] == 'x');
  assertTrue(v3_keys[1] == 'y');
  assertTrue(v3_keys[2] == 'z');
  assertTrue(v3_keys[3] == 'R');
}

void TestIndexedUpdate() {
  std::cout << "*** Testing Values Indexed Update ***" << std::endl;

  // Create some data
  sym::Valuesd values;
  values.Set<double>('x', 1.0);
  values.Set<double>('y', 2.0);
  values.Set<double>('z', -3.0);
  values.Set<geo::Rot3d>({'R', 1}, geo::Rot3d::Identity());
  values.Set<geo::Rot3d>({'R', 2}, geo::Rot3d::FromYawPitchRoll(1.0, 0.0, 0.0));
  values.Set<geo::Pose3d>('P', geo::Pose3d::Identity());

  // Create an index for a subset of keys
  const sym::index_t index = values.CreateIndex({'x', 'y', {'R', 1}});

  // Copy into another Values
  sym::Valuesd values2 = values;

  // Modify some keys in the original
  values.Set<double>('x', 7.7);
  values.Set<geo::Rot3d>({'R', 1}, values.At<geo::Rot3d>({'R', 2}));

  assertTrue(values.At<double>('x') == 7.7);
  assertTrue(values2.At<double>('x') == 1.0);

  // Efficiently update keys into the new values
  values2.Update(index, values);

  assertTrue(values2.At<double>('x') == 7.7);
}

void TestKeyUpdate() {
  std::cout << "*** Testing Values Key Update ***" << std::endl;

  // Create some data
  sym::Valuesd values;
  values.Set<double>('x', 1.0);
  values.Set<double>('y', 2.0);
  values.Set<double>('z', -3.0);
  values.Set<geo::Rot3d>({'R', 1}, geo::Rot3d::Identity());
  values.Set<geo::Rot3d>({'R', 2}, geo::Rot3d::FromYawPitchRoll(1.0, 0.0, 0.0));
  values.Set<geo::Pose3d>('P', geo::Pose3d::Identity());

  // Create an index for a subset of keys (random order should be supported)
  const std::vector<sym::Key> keys = {{'R', 1}, 'x', 'y'};
  const sym::index_t index = values.CreateIndex(keys);

  // Another different structured Values
  sym::Valuesd values2;
  values2.Set<double>('z', 10.0);

  // Update from a different structure
  values2.UpdateOrSet(index, values);

  // Test for update
  assertTrue(values2.At<double>('x') == 1.0);
  assertTrue(values2.At<double>('y') == 2.0);
  assertTrue(values2.At<geo::Rot3d>({'R', 1}) == geo::Rot3d::Identity());

  // Test for not clobbering other field
  assertTrue(values2.At<double>('z') == 10.0);

  // Test efficient update with cached index
  const sym::index_t index2 = values2.CreateIndex(keys);
  values.Set<double>('x', -10.0);
  values.Set<double>('y', 20.0);
  values2.Update(index2, index, values);
  assertTrue(values2.At<double>('x') == -10.0);
  assertTrue(values2.At<double>('y') == 20.0);
}

template <typename Scalar>
void TestLieGroupOps() {
  std::cout << "*** Testing Values<" << typeid(Scalar).name() << "> LieGroupOps ***" << std::endl;
  const Scalar epsilon = 1e-9;

  // Create a values object that stores an identity rotation, and an index for it
  sym::Values<Scalar> v1;
  const geo::Rot3<Scalar> rot = geo::Rot3<Scalar>::Identity();
  v1.Set('R', rot);
  const sym::index_t index = v1.CreateIndex({'R'});

  // Test a bunch of retractions and local coordinates
  std::mt19937 gen(42);
  for (int i = 0; i < 100; ++i) {
    v1.Set('R', rot);
    const sym::Values<Scalar> v2 = v1;

    const geo::Rot3<Scalar> random_rot = geo::Rot3<Scalar>::Random(gen);
    const Eigen::Matrix<Scalar, 3, 1> tangent_vec =
        geo::LieGroupOps<geo::Rot3<Scalar>>::ToTangent(random_rot, epsilon);

    // test retraction
    v1.Retract(index, tangent_vec.data(), epsilon);
    const geo::Rot3<Scalar> retracted_rot = v1.template At<geo::Rot3<Scalar>>('R');
    assertTrue(random_rot.IsApprox(retracted_rot, 1e-6));

    // test local coordinates
    const sym::VectorX<Scalar> local_coords = v1.LocalCoordinates(v2, index, epsilon);
    assertTrue(local_coords.isApprox(tangent_vec));
  }
}

void TestMoveOperator() {
  static_assert(std::is_move_assignable<sym::Values<float>>::value, "");
  sym::Valuesf values;
  values.Set<float>('x', 1.0f);
  values.Set<float>('y', 2.0f);
  values.Set<geo::Rot3f>({'R', 1}, geo::Rot3f::Identity());
  sym::Valuesf values2 = std::move(values);
  assertTrue(values2.At<float>('x') == 1.0f);
}

int main(int argc, char** argv) {
  TestValues<float>();
  TestValues<double>();

  TestImplicitConstruction();
  TestInitializerListConstruction();

  TestLieGroupOps<float>();
  TestLieGroupOps<double>();

  TestIndexedUpdate();
  TestKeyUpdate();
  TestMoveOperator();
}
