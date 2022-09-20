# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Symbolic factor and codegen for the Bundle-Adjustment-in-the-Large problem
"""

from pathlib import Path

import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import values_codegen
from symforce.values import Values


def snavely_reprojection_residual(
    cam_T_world: sf.Pose3,
    intrinsics: sf.V3,
    point: sf.V3,
    pixel: sf.V2,
    epsilon: sf.Scalar,
) -> sf.V2:
    """
    Reprojection residual for the camera model used in the Bundle-Adjustment-in-the-Large dataset, a
    polynomial camera with two distortion coefficients, cx == cy == 0, and fx == fy

    See https://grail.cs.washington.edu/projects/bal/ for more information

    Args:
        cam_T_world: The (inverse) pose of the camera
        intrinsics: Camera intrinsics (f, k1, k2)
        point: The world point to be projected
        pixel: The measured pixel in the camera (with (0, 0) == center of image)

    Returns:
        residual: The reprojection residual
    """
    focal_length, k1, k2 = intrinsics

    # Here we're writing the projection ourselves because this isn't a camera model provided by
    # SymForce.  For cameras in `symforce.cam` we could just create a `sf.PosedCamera` and call
    # `camera.pixel_from_global_point` instead, or we could create a subclass of `sf.CameraCal` and
    # do that.
    point_cam = cam_T_world * point

    p = sf.V2(point_cam[:2]) / sf.Max(-point_cam[2], epsilon)

    r = 1 + k1 * p.squared_norm() + k2 * p.squared_norm() ** 2

    pixel_projected = focal_length * r * p

    return pixel_projected - pixel


def generate(output_dir: Path) -> None:
    """
    Generates the snavely_reprojection_factor into C++, as well as a set of Keys to help construct
    the optimization problem in C++, and puts them into `output_dir`.  This is called by
    `symforce/test/symforce_examples_bundle_adjustment_in_the_large_codegen_test.py` to generate the
    contents of the `gen` folder inside this directory.
    """

    # Generate the residual function (see `gen/snavely_reprojection_factor.h`)
    codegen.Codegen.function(snavely_reprojection_residual, codegen.CppConfig()).with_linearization(
        which_args=["cam_T_world", "intrinsics", "point"]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    # Make a `Values` with variables used in the C++ problem, and generate C++ Keys for them (see
    # `gen/keys.h`)
    values = Values(
        cam_T_world=sf.Pose3(),
        intrinsics=sf.V3(),
        point=sf.V3(),
        pixel=sf.V2(),
        epsilon=sf.Scalar(),
    )

    values_codegen.generate_values_keys(values, output_dir, skip_directory_nesting=True)
