# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import tempfile
from pathlib import Path

from symforce import codegen
from symforce import logger
from symforce import typing as T
from symforce.slam.imu_preintegration.symbolic import imu_preintegration_update


def generate_imu_preintegration(
    config: codegen.CodegenConfig, output_dir: T.Openable = None
) -> Path:
    """
    Generate the IMU preintegration update function.
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(
            prefix=f"sf_codegen_{type(config).__name__.lower()}_", dir="/tmp"
        )
        logger.debug(f"Creating temp directory: {output_dir}")

    cg = codegen.Codegen.function(
        imu_preintegration_update,
        config=config,
        output_names=[
            "new_state",
            "new_state_cov",
            "new_state_D_accel_bias",
            "new_state_D_gyro_bias",
        ],
    )

    cg.generate_function(output_dir=output_dir, skip_directory_nesting=True)

    return Path(output_dir)
