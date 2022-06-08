/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <fstream>

#include <sym/pose3.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/internal/logging_configure.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/values.h>

#include "./gen/keys.h"
#include "./gen/snavely_reprojection_factor.h"

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
                                   sym::Key::WithSuper(CAM_T_WORLD, camera),
                                   sym::Key::WithSuper(INTRINSICS, camera),
                                   sym::Key::WithSuper(POINT, point),
                                   sym::Key::WithSuper(PIXEL, pixel),
                                   EPSILON,
                               },
                               /* optimized_keys = */
                               {
                                   sym::Key::WithSuper(CAM_T_WORLD, camera),
                                   sym::Key::WithSuper(INTRINSICS, camera),
                                   sym::Key::WithSuper(POINT, point),
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
    values.Set(sym::Key::WithSuper(PIXEL, i), Eigen::Vector2d(px, py));
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

    values.Set(sym::Key::WithSuper(CAM_T_WORLD, i),
               sym::Pose3d(sym::Rot3d::FromTangent(Eigen::Vector3d(rx, ry, rz)),
                           Eigen::Vector3d(tx, ty, tz)));
    values.Set(sym::Key::WithSuper(INTRINSICS, i), Eigen::Vector3d(f, k1, k2));
  }

  for (int i = 0; i < num_points; i++) {
    double x, y, z;
    file >> x;
    file >> y;
    file >> z;

    values.Set(sym::Key::WithSuper(POINT, i), Eigen::Vector3d(x, y, z));
  }

  values.Set(EPSILON, sym::kDefaultEpsilond);

  return {std::move(factors), std::move(values), num_cameras, num_points, num_observations};
}

/**
 * Example usage: `bundle_adjustment_in_the_large_example data/problem-21-11315-pre.txt`
 */
int main(int argc, char** argv) {
  sym::internal::SetLogLevel("info");

  SYM_ASSERT(argc == 2);

  // Read the problem from disk, and create the Values and factors
  const auto problem = ReadProblem(argv[1]);

  // Create a copy of the Values - we'll optimize this one in place
  sym::Valuesd optimized_values = problem.values;

  // Optimize
  sym::Optimizerd optimizer{sym::DefaultOptimizerParams(), std::move(problem.factors)};
  const auto stats = optimizer.Optimize(&optimized_values);

  spdlog::info("Finished in {} iterations", stats.iterations.size());
}
