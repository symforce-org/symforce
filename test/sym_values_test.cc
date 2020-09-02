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

// TODO(hayk): Fix this
template <typename Scalar>
void TestRetract() {
  std::cout << "*** Testing Values<" << typeid(Scalar).name() << "> Retract ***" << std::endl;
  const Scalar epsilon = 1e-6;
  const Scalar tolerance = 10 * epsilon;

  // Create a values object that stores an identity rotation
  sym::Values<Scalar> v;
  sym::Key key('k', 1);
  geo::Rot3<Scalar> rot = geo::Rot3<Scalar>::Identity();
  v.Set(key, rot);

  // Perturb the rotation in the tangent space
  // TODO(nathan): Try several random perturbations
  geo::Rot3<Scalar> perturbation = geo::Rot3<Scalar>::FromYawPitchRoll(1.0, 0.0, 0.0);
  Eigen::Matrix<Scalar, 3, 1> tangent_mat =
      geo::LieGroupOps<geo::Rot3<Scalar>>::ToTangent(perturbation, epsilon);

  const sym::index_t index = v.CreateIndex({key});
  const std::vector<Scalar> tangent_vec(tangent_mat.data(),
                                        tangent_mat.data() + tangent_mat.size());
  v.Retract(index, tangent_vec.data(), epsilon);

  geo::Rot3<Scalar> perturbed_rot = v.template At<geo::Rot3<Scalar>>(key);
  assertTrue(perturbation.IsApprox(perturbed_rot, tolerance));
}

int main(int argc, char** argv) {
  TestValues<float>();
  TestValues<double>();

  TestRetract<float>();
  TestRetract<double>();
}
