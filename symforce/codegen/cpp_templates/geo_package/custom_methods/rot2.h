  // --------------------------------------------------------------------------
  // Handwritten methods for Rot2
  // --------------------------------------------------------------------------

  // Generate a random element of SO2 from a number u1 in [0, 1]
  static Rot2 RandomFromUniformSamples(const Scalar u1) {
    const Scalar theta = 2 * M_PI * u1;
    return Rot2(Eigen::Matrix<Scalar, 2, 1>(std::cos(theta), std::sin(theta)));
  }

  // Generate a random element in SO3
  static Rot2 Random(std::mt19937& gen) {
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
    return RandomFromUniformSamples(dist(gen));
  }
