# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import functools

from symforce import codegen
from symforce import typing as T
from symforce.slam.imu_preintegration.manifold_symbolic import imu_manifold_preintegration_update
from symforce.slam.imu_preintegration.manifold_symbolic import internal_imu_residual


def generate_manifold_imu_preintegration(
    config: codegen.CodegenConfig, output_dir: T.Openable
) -> None:
    """
    Generate the on-manifold IMU preintegration update and residual functions.
    """

    update_output_names = [
        "new_DR",
        "new_Dv",
        "new_Dp",
        "new_covariance",
        "new_DR_D_gyro_bias",
        "new_Dv_D_accel_bias",
        "new_Dv_D_gyro_bias",
        "new_Dp_D_accel_bias",
        "new_Dp_D_gyro_bias",
    ]

    codegen_update = codegen.Codegen.function(
        functools.partial(imu_manifold_preintegration_update, use_handwritten_derivatives=True),
        config=config,
        output_names=update_output_names,
    )
    codegen_update.generate_function(output_dir=output_dir, skip_directory_nesting=True)

    codegen_update_auto_derivative = codegen.Codegen.function(
        functools.partial(imu_manifold_preintegration_update, use_handwritten_derivatives=False),
        name="imu_manifold_preintegration_update_auto_derivative",
        docstring="""
    Alternative to ImuManifoldPreintegrationUpdate that uses auto-derivatives. Exists only to
    help verify correctness of ImuManifoldPreintegrationUpdate. Should have the same output.
    Since this function is more expensive, there is no reason to use it instead.
        """
        + (
            imu_manifold_preintegration_update.__doc__
            if imu_manifold_preintegration_update.__doc__ is not None
            else ""
        ),
        config=config,
        output_names=update_output_names,
    )
    codegen_update_auto_derivative.generate_function(
        output_dir=output_dir, skip_directory_nesting=True
    )

    codegen_residual = codegen.Codegen.function(
        internal_imu_residual,
        config=config,
    ).with_linearization(
        which_args=[
            "pose_i",
            "vel_i",
            "pose_j",
            "vel_j",
            "accel_bias_i",
            "gyro_bias_i",
        ]
    )
    codegen_residual.generate_function(output_dir=output_dir, skip_directory_nesting=True)
