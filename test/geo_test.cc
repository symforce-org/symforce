/**
 * Tests for C++ geometry types. Mostly checking all the templates compile since
 * the math is tested comprehensively in symbolic form.
 */

#include <iostream>

#include <geo/rot2.h>
#include <geo/pose2.h>
#include <geo/rot3.h>
#include <geo/pose3.h>

void assertTrue(bool a) { assert(a); }

void TestRot3() {
    // Make a random rotation
  const geo::Rot3f rot = geo::Rot3f::Expmap(geo::Rot3f::TangentVec::Random());

  // Cast
  const geo::Rot3d rotd = rot.Cast<double>();
  assertTrue(rotd.IsApprox(rot.Cast<double>(), 1e-6));
  assertTrue(rotd.Cast<float>().IsApprox(rot, 1e-6));

  // Convert to Eigen rotation representations
  const Eigen::Quaternionf quat = rot.Quaternion();
  const Eigen::AngleAxisf aa = rot.AngleAxis();
  const Eigen::Matrix<float, 3, 3> mat = rot.Matrix();
  const Eigen::Matrix<float, 3, 1> ypr = rot.YawPitchRoll();

  // Rotate a point
  const Eigen::Vector3f point = Eigen::Vector3f::Random();
  assertTrue((quat * point).isApprox(aa * point, 1e-6));
  assertTrue((quat * point).isApprox(mat * point, 1e-6));
  assertTrue((quat * point).isApprox(rot * point, 1e-6));

  // Construct back from Eigen rotation representations
  assertTrue(geo::Rot3f(quat).IsApprox(rot, 1e-6));
  assertTrue(geo::Rot3f(aa).IsApprox(rot, 1e-6));
  assertTrue(geo::Rot3f::FromMatrix(mat).IsApprox(rot, 1e-6));
  assertTrue(geo::Rot3f::FromYawPitchRoll(ypr).IsApprox(rot, 1e-6));

  // Make a pose
  geo::Pose3f pose(geo::Rot3f(aa), point);
  assertTrue(pose.Rotation().IsApprox(rot, 1e-6));
  assertTrue(pose.Position() == point);

  const geo::Pose3f pose_inv = pose.Inverse();
  assertTrue(pose_inv.Rotation().IsApprox(rot.Inverse(), 1e-9));

  // Transform a point with a pose
  assertTrue((pose_inv * point).norm() < 1e-6);

  // Check zero comparison
  assertTrue(geo::Rot3f(Eigen::Vector4f::Zero()).IsApprox(geo::Rot3f(Eigen::Vector4f::Zero()), 1e-9));
  assertTrue(!geo::Rot3f().IsApprox(geo::Rot3f(Eigen::Vector4f::Zero()), 1e-9));
}


void TestRot2Pose2() {
  const geo::Rot2f rot = geo::Rot2f::Expmap(geo::Rot2f::TangentVec::Random());
  const Eigen::Vector2f pos = Eigen::Vector2f::Random();

  // Cast
  const geo::Rot2d rotd = rot.Cast<double>();
  assertTrue(rotd.IsApprox(rot.Cast<double>(), 1e-6));
  assertTrue(rotd.Cast<float>().IsApprox(rot, 1e-6));

  // Make a pose
  const geo::Pose2f pose(rot, pos);
  assertTrue(pose.Rotation().IsApprox(rot, 1e-6));
  assertTrue(pose.Position() == pos);

  const geo::Pose2f pose_inv = pose.Inverse();
  assertTrue(pose_inv.Rotation().IsApprox(rot.Inverse(), 1e-9));
}

template <typename T>
void TestStorageOps() {
  using Scalar = typename T::Scalar;

  const T value;
  std::cout << "*** Testing StorageOps: " << value << " ***" << std::endl;

  constexpr int32_t storage_dim = geo::StorageOps<T>::StorageDim();
  assertTrue(value.Storage().rows() == storage_dim);
  assertTrue(value.Storage().cols() == 1);

  std::vector<Scalar> vec;
  value.ToList(&vec);
  assertTrue(vec.size() > 0);
  assertTrue(vec.size() == storage_dim);
  for (int i = 0; i < vec.size(); ++i) {
    assertTrue(vec[i] == value.Storage()[i]);
  }

  const T value2 = geo::StorageOps<T>::FromList(vec);
  assertTrue(value.Storage() == value2.Storage());
  vec[0] = 2.1;
  const T value3 = geo::StorageOps<T>::FromList(vec);
  assertTrue(value.Storage() != value3.Storage());
}

template <typename T>
void TestGroupOps() {
  const T identity;
  std::cout << "*** Testing GroupOps: " << identity << " ***" << std::endl;

  assertTrue(identity.IsApprox(geo::GroupOps<T>::Identity(), 1e-9));
  assertTrue(identity.IsApprox(geo::GroupOps<T>::Compose(identity, identity), 1e-9));
  assertTrue(identity.IsApprox(geo::GroupOps<T>::Inverse(identity), 1e-9));
  assertTrue(identity.IsApprox(geo::GroupOps<T>::Between(identity, identity), 1e-9));
}

template <typename T>
void TestLieGroupOps() {
  using Scalar = typename T::Scalar;
  using TangentVec = Eigen::Matrix<Scalar, geo::LieGroupOps<T>::TangentDim(), 1>;
  const Scalar epsilon = 1e-8;

  const T identity;
  std::cout << "*** Testing LieGroupOps: " << identity << " ***" << std::endl;

  constexpr int32_t tangent_dim = geo::LieGroupOps<T>::TangentDim();
  assertTrue(tangent_dim > 0);
  assertTrue(tangent_dim <= geo::StorageOps<T>::StorageDim());

  const TangentVec pertubation = TangentVec::Random();
  const T value = geo::LieGroupOps<T>::Expmap(pertubation, epsilon);

  const TangentVec recovered_pertubation = geo::LieGroupOps<T>::Logmap(value, epsilon);
  assertTrue(pertubation.isApprox(recovered_pertubation, std::sqrt(epsilon)));

  const T recovered_identity = geo::LieGroupOps<T>::Retract(
    value, -recovered_pertubation, epsilon);
  assertTrue(recovered_identity.IsApprox(identity, std::sqrt(epsilon)));

  const TangentVec pertubation_zero = geo::LieGroupOps<T>::LocalCoordinates(
    identity, recovered_identity, epsilon);
  assertTrue((pertubation_zero - TangentVec::Zero()).norm() < std::sqrt(epsilon));
}

int main(int argc, char** argv) {
  TestStorageOps<geo::Rot2<double>>();
  TestGroupOps<geo::Rot2<double>>();
  TestLieGroupOps<geo::Rot2<double>>();

  TestStorageOps<geo::Pose2<double>>();
  TestGroupOps<geo::Pose2<double>>();
  TestLieGroupOps<geo::Pose2<double>>();

  TestStorageOps<geo::Rot3<double>>();
  TestGroupOps<geo::Rot3<double>>();
  TestLieGroupOps<geo::Rot3<double>>();

  TestStorageOps<geo::Pose3<double>>();
  TestGroupOps<geo::Pose3<double>>();
  TestLieGroupOps<geo::Pose3<double>>();

  TestStorageOps<geo::Rot2<float>>();
  TestGroupOps<geo::Rot2<float>>();
  TestLieGroupOps<geo::Rot2<float>>();

  TestStorageOps<geo::Pose2<float>>();
  TestGroupOps<geo::Pose2<float>>();
  TestLieGroupOps<geo::Pose2<float>>();

  TestStorageOps<geo::Rot3<float>>();
  TestGroupOps<geo::Rot3<float>>();
  TestLieGroupOps<geo::Rot3<float>>();

  TestStorageOps<geo::Pose3<float>>();
  TestGroupOps<geo::Pose3<float>>();
  TestLieGroupOps<geo::Pose3<float>>();

  TestRot3();
  TestRot2Pose2();
}
