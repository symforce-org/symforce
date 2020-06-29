  // --------------------------------------------------------------------------
  // Handwritten methods for Pose2
  // --------------------------------------------------------------------------

  Pose2(const Rot2<Scalar>& rotation, const Eigen::Matrix<Scalar, 2, 1>& position) {
    data_.template head<2>() = rotation.Data();
    data_.template tail<2>() = position;
  }

  Rot2<Scalar> Rotation() const {
      return Rot2<Scalar>(data_.template head<2>());
  }

  Eigen::Matrix<Scalar, 2, 1> Position() const {
      return data_.template tail<2>();
  }
