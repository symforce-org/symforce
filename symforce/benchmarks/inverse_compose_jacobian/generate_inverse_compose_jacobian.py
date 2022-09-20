# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from pathlib import Path

import symforce.symbolic as sf
from symforce import codegen


def generate(output_dir: Path) -> None:
    from sympy.simplify import cse_opts

    config = codegen.CppConfig(
        force_no_inline=True, cse_optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)]
    )

    def pose_inverse_compose_point(pose: sf.Pose3, point: sf.V3) -> sf.V3:
        return pose.inverse() * point

    codegen.Codegen.function(pose_inverse_compose_point, config=config).with_jacobians(
        which_args=["pose"]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    def pose_inverse(pose: sf.Pose3) -> sf.Pose3:
        return pose.inverse()

    codegen.Codegen.function(pose_inverse, config=config).with_jacobians().generate_function(
        output_dir=output_dir, skip_directory_nesting=True
    )

    def pose_compose_point(pose: sf.Pose3, point: sf.V3) -> sf.V3:
        return pose * point

    codegen.Codegen.function(pose_compose_point, config=config).with_jacobians(
        which_args=["pose"]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)
