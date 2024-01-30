# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import functools

import symforce.symbolic as sf
from symforce import codegen
from symforce import typing as T
from symforce.slam.imu_preintegration.manifold_symbolic import imu_manifold_preintegration_update
from symforce.slam.imu_preintegration.manifold_symbolic import internal_imu_residual
from symforce.slam.imu_preintegration.manifold_symbolic import roll_forward_state


def internal_imu_unit_gravity_residual(
    pose_i: sf.Pose3,
    vel_i: sf.V3,
    pose_j: sf.Pose3,
    vel_j: sf.V3,
    accel_bias_i: sf.V3,
    gyro_bias_i: sf.V3,
    # Preintegrated measurements: state
    DR: sf.Rot3,
    Dv: sf.V3,
    Dp: sf.V3,
    # Other pre-integrated quantities
    sqrt_info: sf.M99,
    DR_D_gyro_bias: sf.M33,
    Dv_D_accel_bias: sf.M33,
    Dv_D_gyro_bias: sf.M33,
    Dp_D_accel_bias: sf.M33,
    Dp_D_gyro_bias: sf.M33,
    # other
    accel_bias_hat: sf.V3,
    gyro_bias_hat: sf.V3,
    gravity_direction: sf.Unit3,
    gravity_norm: sf.Scalar,
    dt: T.Scalar,
    epsilon: T.Scalar,
) -> sf.V9:
    return internal_imu_residual(
        pose_i=pose_i,
        vel_i=vel_i,
        pose_j=pose_j,
        vel_j=vel_j,
        accel_bias_i=accel_bias_i,
        gyro_bias_i=gyro_bias_i,
        DR=DR,
        Dv=Dv,
        Dp=Dp,
        sqrt_info=sqrt_info,
        DR_D_gyro_bias=DR_D_gyro_bias,
        Dv_D_accel_bias=Dv_D_accel_bias,
        Dv_D_gyro_bias=Dv_D_gyro_bias,
        Dp_D_accel_bias=Dp_D_accel_bias,
        Dp_D_gyro_bias=Dp_D_gyro_bias,
        accel_bias_hat=accel_bias_hat,
        gyro_bias_hat=gyro_bias_hat,
        gravity=gravity_direction.to_unit_vector() * gravity_norm,
        dt=dt,
        epsilon=epsilon,
    )


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

    codegen.Codegen.function(internal_imu_residual, config=config).with_linearization(
        which_args=[
            "pose_i",
            "vel_i",
            "pose_j",
            "vel_j",
            "accel_bias_i",
            "gyro_bias_i",
        ]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    codegen.Codegen.function(
        internal_imu_residual, name="internal_imu_with_gravity_residual", config=config
    ).with_linearization(
        which_args=[
            "pose_i",
            "vel_i",
            "pose_j",
            "vel_j",
            "accel_bias_i",
            "gyro_bias_i",
            "gravity",
        ]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    codegen.Codegen.function(internal_imu_unit_gravity_residual, config=config).with_linearization(
        which_args=[
            "pose_i",
            "vel_i",
            "pose_j",
            "vel_j",
            "accel_bias_i",
            "gyro_bias_i",
            "gravity_direction",
        ]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    codegen.Codegen.function(roll_forward_state, config=config).generate_function(
        output_dir=output_dir, skip_directory_nesting=True
    )
