# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Symbolic factor and codegen for the Bundle-Adjustment-in-the-Large problem
"""

from pathlib import Path

from symforce import codegen
from symforce.codegen import values_codegen
from symforce import geo
from symforce import sympy as sm
from symforce import typing as T
from symforce.values import Values


def snavely_reprojection_residual(
    cam_T_world: geo.Pose3,
    intrinsics: geo.V3,
    point: geo.V3,
    pixel: geo.V2,
    epsilon: T.Scalar,
) -> geo.V2:
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

    point_cam = cam_T_world * point

    p = geo.V2(point_cam[:2]) / sm.Max(-point_cam[2], epsilon)

    r = 1 + k1 * p.squared_norm() + k2 * p.squared_norm() ** 2

    pixel_projected = focal_length * r * p

    return pixel_projected - pixel


def generate(output_dir: Path) -> None:
    codegen.Codegen.function(snavely_reprojection_residual, codegen.CppConfig()).with_linearization(
        which_args=["cam_T_world", "intrinsics", "point"]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    values = Values(
        cam_T_world=geo.Pose3(),
        intrinsics=geo.V3(),
        point=geo.V3(),
        pixel=geo.V2(),
        epsilon=T.Scalar(),
    )

    values_codegen.generate_values_keys(values, output_dir, skip_directory_nesting=True)
