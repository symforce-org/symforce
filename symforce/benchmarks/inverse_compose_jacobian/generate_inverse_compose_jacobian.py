# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from pathlib import Path

from symforce import geo
from symforce import codegen


def generate(output_dir: Path) -> None:
    from sympy.simplify import cse_opts

    config = codegen.CppConfig(
        force_no_inline=True, cse_optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)]
    )

    def pose_inverse_compose_point(pose: geo.Pose3, point: geo.V3) -> geo.V3:
        return pose.inverse() * point

    codegen.Codegen.function(pose_inverse_compose_point, config=config).with_jacobians(
        which_args=["pose"]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    def pose_inverse(pose: geo.Pose3) -> geo.Pose3:
        return pose.inverse()

    codegen.Codegen.function(pose_inverse, config=config).with_jacobians().generate_function(
        output_dir=output_dir, skip_directory_nesting=True
    )

    def pose_compose_point(pose: geo.Pose3, point: geo.V3) -> geo.V3:
        return pose * point

    codegen.Codegen.function(pose_compose_point, config=config).with_jacobians(
        which_args=["pose"]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)
