/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <algorithm>
#include <fstream>
#include <vector>

#include <spdlog/spdlog.h>

#include <sym/pose3.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/values.h>

#include "./gen/keys.h"
#include "./gen/snavely_reprojection_factor.h"

/**
 * Calculate median position
 */
Eigen::Vector3d CalculateMedianPosition(const sym::Valuesd& values, int num_cameras) {
  if (num_cameras == 0) {
    return Eigen::Vector3d::Zero();
  }

  std::vector<Eigen::Vector3d> positions;
  positions.reserve(num_cameras);
  for (int i = 0; i < num_cameras; i++) {
    positions.push_back(values.At<sym::Pose3d>(sym::Keys::CAM_T_WORLD.WithSuper(i)).Position());
  }

  std::nth_element(
      positions.begin(), positions.begin() + positions.size() / 2, positions.end(),
      [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return a.norm() < b.norm(); });

  return positions[positions.size() / 2];
}

/**
 * Remove cameras that are far from the median
 * See https://github.com/symforce-org/symforce/issues/387
 */
void FilterCameraOutliers(std::vector<sym::Factord>& factors, sym::Valuesd& values,
                          int& num_cameras, double threshold = 10.0) {
  std::vector<sym::Pose3d> camera_poses;
  std::vector<sym::Factord> filtered_factors;

  Eigen::Vector3d mean = CalculateMedianPosition(values, num_cameras);

  int num_filtered_cameras = 0;
  for (int i = 0; i < num_cameras; i++) {
    sym::Pose3d pose = values.At<sym::Pose3d>(sym::Keys::CAM_T_WORLD.WithSuper(i));
    double distance = (pose.Position() - mean).norm();

    if (distance <= threshold) {
      values.Set(sym::Keys::CAM_T_WORLD.WithSuper(num_filtered_cameras), pose);
      values.Set(sym::Keys::INTRINSICS.WithSuper(num_filtered_cameras),
                 values.At<Eigen::Vector3d>(sym::Keys::INTRINSICS.WithSuper(i)));

      filtered_factors.push_back(factors[i]);

      num_filtered_cameras++;
    }
  }

  num_cameras = num_filtered_cameras;
  factors = std::move(filtered_factors);
}

using namespace sym::Keys;

/**
 * Create a `sym::Factor` for the reprojection residual, attached to the given camera and point
 * variables.  It's also attached to fixed entries in the Values for the pixel measurement and the
 * constant EPSILON.
 */
sym::Factord MakeFactor(int camera, int point, int pixel) {
  return sym::Factord::Hessian(sym::SnavelyReprojectionFactor<double>,
                               /* all_keys = */
                               {
                                   CAM_T_WORLD.WithSuper(camera),
                                   INTRINSICS.WithSuper(camera),
                                   POINT.WithSuper(point),
                                   PIXEL.WithSuper(pixel),
                                   EPSILON,
                               },
                               /* optimized_keys = */
                               {
                                   CAM_T_WORLD.WithSuper(camera),
                                   INTRINSICS.WithSuper(camera),
                                   POINT.WithSuper(point),
                               });
}

/**
 * A struct to represent the problem definition
 */
struct Problem {
  std::vector<sym::Factord> factors;
  sym::Valuesd values;
  int num_cameras;
  int num_points;
  int num_observations;
};

/**
 * Read the problem description from the given path
 *
 * See https://grail.cs.washington.edu/projects/bal/ for file format description
 */
Problem ReadProblem(const std::string& filename) {
  std::ifstream file(filename);

  int num_cameras, num_points, num_observations;
  file >> num_cameras;
  file >> num_points;
  file >> num_observations;

  std::vector<sym::Factord> factors;
  sym::Valuesd values;

  for (int i = 0; i < num_observations; i++) {
    int camera, point;
    file >> camera;
    file >> point;

    double px, py;
    file >> px;
    file >> py;

    factors.push_back(MakeFactor(camera, point, i));
    values.Set(PIXEL.WithSuper(i), Eigen::Vector2d(px, py));
  }

  for (int i = 0; i < num_cameras; i++) {
    double rx, ry, rz, tx, ty, tz, f, k1, k2;
    file >> rx;
    file >> ry;
    file >> rz;
    file >> tx;
    file >> ty;
    file >> tz;
    file >> f;
    file >> k1;
    file >> k2;

    values.Set(CAM_T_WORLD.WithSuper(i),
               sym::Pose3d(sym::Rot3d::FromTangent(Eigen::Vector3d(rx, ry, rz)),
                           Eigen::Vector3d(tx, ty, tz)));
    values.Set(INTRINSICS.WithSuper(i), Eigen::Vector3d(f, k1, k2));
  }

  for (int i = 0; i < num_points; i++) {
    double x, y, z;
    file >> x;
    file >> y;
    file >> z;

    values.Set(POINT.WithSuper(i), Eigen::Vector3d(x, y, z));
  }

  values.Set(EPSILON, sym::kDefaultEpsilond);

  FilterCameraOutliers(factors, values, num_cameras);

  return {std::move(factors), std::move(values), num_cameras, num_points, num_observations};
}

/**
 * Example usage: `bundle_adjustment_in_the_large_example data/problem-21-11315-pre.txt`
 */
int main(int argc, char** argv) {
  SYM_ASSERT_EQ(argc, 2);

  // Read the problem from disk, and create the Values and factors
  const auto problem = ReadProblem(argv[1]);

  // Create a copy of the Values - we'll optimize this one in place
  sym::Valuesd optimized_values = problem.values;

  // Optimize
  auto params = sym::DefaultOptimizerParams();
  params.verbose = true;
  params.lambda_update_type = sym::lambda_update_type_t::DYNAMIC;
  sym::Optimizerd optimizer{params, std::move(problem.factors)};
  const auto stats = optimizer.Optimize(optimized_values);

  spdlog::info("Finished in {} iterations", stats.iterations.size());
}
