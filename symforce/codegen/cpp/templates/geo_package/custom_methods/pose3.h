  // --------------------------------------------------------------------------
  // Handwritten methods for Pose3
  // --------------------------------------------------------------------------

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  Pose3(const Rot3<Scalar>& rotation, const Vector3& position) {
    data_.template head<4>() = rotation.Data();
    data_.template tail<3>() = position;
  }

  Rot3<Scalar> Rotation() const {
      return Rot3<Scalar>(data_.template head<4>());
  }

  Vector3 Position() const {
      return data_.template tail<3>();
  }

  // TODO(hayk): Could codegen this.
  Vector3 Compose(const Vector3& point) const {
      return Rotation() * point + Position();
  }
